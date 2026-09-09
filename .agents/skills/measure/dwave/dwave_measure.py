"""D-wave pair-pair correlation: GPU measurement stage.

Measures the unique bond-pair terms F(b_L, b_R) = <B†_{b_L} B_{b_R}> and
hopping terms G_s(i,j) = <c†_{is} c_{js}> defined in dwave_pairing.md, for one
reference site r.

Why stay on GPU for the whole run: converting a term's `Oloc` result to numpy
as soon as it's computed (e.g. `np.mean(np.real(op.Oloc(...)))` inside the
per-term loop) is merely wasteful on a single GPU, but on a sharded
multi-GPU/multi-node run every such conversion forces a cross-device
collective to materialize the sharded array on the host. Doing that ~200
times per sweep (one per unique term) multiplies communication overhead for
no benefit, since none of the individual term values are needed until the
whole run is done — so every reduction below uses `jnp.mean`/`jnp.real` (not
`np`) and every sweep's values are `jnp.stack`ed and kept as a JAX array; the
per-sweep arrays are themselves collected in a plain Python list (no
computation forces materialization) and only stacked and pulled to host once,
after the sweep loop, via `qtx.utils.to_replicated_numpy` — the multi-host-safe
gather, unlike plain `np.asarray` which only sees the local shard when there's
more than one host. The accumulated data is tiny (nsweeps x nterms floats —
even 10 000 sweeps is a few MB), so this is safe unless nterms is pushed into
the hundreds of thousands.

Output (dwave_terms.npz):
- F, G: (nsweeps, nterms) raw per-sweep sweep-means, not yet assembled.
- F_keys, G_keys: the bond/site tuples identifying each column, saved as
  object arrays (load with allow_pickle=True).
- LX, LY, R: lattice extent and the reference site, needed by
  dwave_assembly.py to rebuild the same bond-keying scheme.

Run dwave_assembly.py (CPU, numpy) separately to jackknife and assemble
P(dr) and P_c(dr) — that step is pure numpy over a couple hundred numbers
and would otherwise waste GPU-node hours.

Adapting to production: swap the ED `state` for a trained `Variational` (set
`max_parallel`), adjust NSAMPLES/NSWEEPS to the target statistics, and change
the lattice. The per-sweep array is tiny (nterms floats), so holding all
NSWEEPS of them in host memory is cheap even for long production runs — no
need to stream to disk incrementally.
"""

import numpy as np
import jax.numpy as jnp
import quantax as qtx
from quantax.sampler import Samples
from quantax.operator import (
    Operator,
    create_u,
    create_d,
    annihilate_u,
    annihilate_d,
    number_u,
    number_d,
)

qtx.set_default_dtype(jnp.float64)

LX, LY = 4, 3
lattice = qtx.sites.Grid(
    (LX, LY), particle_type=qtx.PARTICLE_TYPE.spinful_fermion, Nparticles=(5, 5)
)
N = lattice.Nsites

# --- state preparation: replace with your trained Variational (set max_parallel!) ---
H = qtx.operator.Hubbard(U=8.0)
E, wf = H.diagonalize()
state = qtx.state.DenseState(wf)

R = (0, 0)  # reference site r
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
SITES = [(x, y) for x in range(LX) for y in range(LY)]


def wrap(site):
    return (site[0] % LX, site[1] % LY)


def canon_bond(site, d):
    """Canonical (x, y, axis) key of the NN bond from `site` along `d`.

    B†(a,b) = B†(b,a), so each bond is keyed by its lower endpoint and axis.
    Must match the identical helper in dwave_assembly.py. Wrapping assumes
    PBC; with APBC the wrap sign must be tracked.
    """
    if d[0] + d[1] < 0:
        site = (site[0] + d[0], site[1] + d[1])
    x, y = wrap(site)
    return (x, y, 0 if d[0] != 0 else 1)


def bond_sites(bond):
    x, y, axis = bond
    return (x, y), wrap((x + 1, y) if axis == 0 else (x, y + 1))


def pair_create(bond) -> Operator:
    """Bond-singlet creation B† = (c†_{a↑} c†_{b↓} - c†_{a↓} c†_{b↑}) / sqrt(2)."""
    a, b = bond_sites(bond)
    Bd = create_u(*a) @ create_d(*b) - create_d(*a) @ create_u(*b)
    return Bd / np.sqrt(2)


def hop(i, j, spin) -> Operator:
    """G_s(i,j) = c†_{is} c_{js}."""
    if i == j:
        return number_u(*i) if spin == "u" else number_d(*i)
    if spin == "u":
        return create_u(*i) @ annihilate_u(*j)
    return create_d(*i) @ annihilate_d(*j)


# ---- unique terms: 4 x 2N pair terms F and 5 x N x 2 hopping terms G ----
left_bonds = sorted({canon_bond(R, d) for d in DIRS})
all_bonds = [(x, y, ax) for (x, y) in SITES for ax in (0, 1)]
left_sites = sorted({s for b in left_bonds for s in bond_sites(b)})

F_keys = [(bl, br) for bl in left_bonds for br in all_bonds]
G_keys = [(i, j, s) for i in left_sites for j in SITES for s in ("u", "d")]
F_ops = [pair_create(bl) @ pair_create(br).H for bl, br in F_keys]
G_ops = [hop(i, j, s) for i, j, s in G_keys]
print(f"unique terms: {len(F_ops)} pair + {len(G_ops)} hopping")

# ---- Monte Carlo sampling and measurement ----
NSAMPLES = 4096
NSWEEPS = 32
sampler = qtx.sampler.ParticleHop(
    state, NSAMPLES, thermal_steps=50 * N, sweep_steps=5 * N
)

F_sweeps, G_sweeps = [], []
for n in range(NSWEEPS):
    samples = sampler.sweep()
    # compute psi (and RefModel internals) once; all Oloc calls below reuse it
    if state.use_ref:
        psi, internal = state.init_internal(samples.spins)
    else:
        psi, internal = state(samples.spins), None
    rw = 1.0 if samples.reweight_factor is None else samples.reweight_factor
    samples = Samples(samples.spins, psi, internal, samples.reweight_factor)

    # correlations are real for a time-reversal symmetric state; stay in JAX
    # (jnp, not np) for every term and every sweep — no host sync yet.
    F_sweeps.append(
        jnp.stack([jnp.mean(jnp.real(op.Oloc(state, samples)) * rw) for op in F_ops])
    )
    G_sweeps.append(
        jnp.stack([jnp.mean(jnp.real(op.Oloc(state, samples)) * rw) for op in G_ops])
    )

# one host sync for the entire run, gathering correctly even across hosts
F_series = qtx.utils.to_replicated_numpy(jnp.stack(F_sweeps))
G_series = qtx.utils.to_replicated_numpy(jnp.stack(G_sweeps))

def to_object_array(keys):
    """1D array of tuples; plain np.array(keys, dtype=object) would flatten
    the uniform tuple-of-tuples shape into a multi-dimensional array instead."""
    arr = np.empty(len(keys), dtype=object)
    arr[:] = keys
    return arr


np.savez(
    "dwave_terms.npz",
    F=F_series,
    F_keys=to_object_array(F_keys),
    G=G_series,
    G_keys=to_object_array(G_keys),
    LX=LX,
    LY=LY,
    R=np.array(R),
)
print("saved dwave_terms.npz")
