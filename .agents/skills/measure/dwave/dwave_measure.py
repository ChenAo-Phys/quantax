"""D-wave pair-pair correlation: GPU measurement stage.

Measures the unique bond-pair terms F(b_L, b_R) = <B†_{b_L} B_{b_R}> and
hopping terms G_s(i,j) = <c†_{is} c_{js}> defined in dwave_pairing.md, for one
reference site r.

Why keep the term values on the GPU: converting a term's `Oloc` result to
numpy as soon as it's computed (e.g. `np.mean(np.real(op.Oloc(...)))` inside
the per-term loop) is merely wasteful on a single GPU, but on a sharded
multi-GPU/multi-node run every such conversion forces a cross-device
collective to materialize the sharded array on the host, once per term. So
every reduction below uses `jnp.mean`/`jnp.real` (not `np`), every sweep's
values are `jnp.stack`ed and kept as a JAX array in a plain Python list, and
the series is pulled to host only at the periodic saves, via
`qtx.utils.to_replicated_numpy` — the multi-host-safe gather, unlike plain
`np.asarray` which only sees the local shard when there's more than one host.
This saves a gather per term, but every `Oloc` call still does one blocking
host sync (`_get_conn_size(...).item()`) per update-mode group, so the ~18N
terms cost >= ~18N syncs per sweep. To remove that overhead, pack the terms
into one connected-config pass and forward (see "Off-diagonal" in SKILL.md).

Output (dwave_terms.npz, rewritten every SAVE_EVERY sweeps and at the end):
- F, G: (nsweeps_done, nterms) raw per-sweep sweep-means, not yet assembled.
- F_keys, G_keys: the bond/site tuples identifying each column, saved as
  object arrays (load with allow_pickle=True).
- LX, LY, R: lattice extent and the reference site, needed by
  dwave_assembly.py to rebuild the same bond-keying scheme.

Run dwave_assembly.py (CPU, numpy) separately to jackknife and assemble
P(dr) and P_c(dr) — that step is pure numpy over a couple hundred numbers
and would otherwise waste GPU-node hours.

Adapting to production: swap the ED `state` for a trained `Variational`,
remove the `set_default_dtype(float64)` line, adjust NSAMPLES/NSWEEPS to the
target statistics, and change the lattice. Every F/G term has <= 4 connected
configs per sample, so each per-term `Oloc` costs at least one full forward
chunk per device: set `max_parallel` to ~NSAMPLES // ndevices here, not the
large chunk suited to a Hamiltonian (under `symm` the chunk counts unexpanded
configs, so don't multiply it by nsymm). At large lattices, pack the terms
instead: one `apply_off_diag` pass and one forward over all F (or G) terms,
split per term, cross-checked against per-term `Oloc` on sweep 0. The
periodic saves make a long multi-node run that dies or hits its walltime keep
every completed sweep; the per-sweep arrays themselves are tiny (nterms
floats).
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
import quantax as qtx
from quantax.operator import (
    Operator,
    create_u,
    create_d,
    annihilate_u,
    annihilate_d,
    number_u,
    number_d,
)

# ED reference only: turns on the process-global jax_enable_x64 (never reset).
# Remove when measuring a trained fp32 Variational.
qtx.set_default_dtype(jnp.float64)

is_main = jax.process_index() == 0

LX, LY = 4, 3
lattice = qtx.sites.Grid(
    (LX, LY), particle_type=qtx.PARTICLE_TYPE.spinful_fermion, Nparticles=(5, 5)
)
N = lattice.Nsites

# --- state preparation: replace with your trained Variational (max_parallel, see
# the docstring) ---
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
if is_main:
    print(f"unique terms: {len(F_ops)} pair + {len(G_ops)} hopping")

# ---- Monte Carlo sampling and measurement ----
NSAMPLES = 4096
NSWEEPS = 32
sampler = qtx.sampler.ParticleHop(
    state, NSAMPLES, thermal_steps=50 * N, sweep_steps=5 * N
)

SAVE_EVERY = 8  # sweeps between saves of the partial series


def to_object_array(keys):
    """1D array of tuples; plain np.array(keys, dtype=object) would flatten
    the uniform tuple-of-tuples shape into a multi-dimensional array instead."""
    arr = np.empty(len(keys), dtype=object)
    arr[:] = keys
    return arr


def save(F_sweeps, G_sweeps):
    """Gather the series so far (collective: call on every process) and write it
    from process 0 only, atomically replacing the previous file."""
    F_series = qtx.utils.to_replicated_numpy(jnp.stack(F_sweeps))
    G_series = qtx.utils.to_replicated_numpy(jnp.stack(G_sweeps))
    if is_main:
        np.savez(
            "dwave_terms.tmp.npz",
            F=F_series,
            F_keys=to_object_array(F_keys),
            G=G_series,
            G_keys=to_object_array(G_keys),
            LX=LX,
            LY=LY,
            R=np.array(R),
        )
        os.replace("dwave_terms.tmp.npz", "dwave_terms.npz")
        print(f"saved dwave_terms.npz ({len(F_sweeps)} sweeps)")


F_sweeps, G_sweeps = [], []
for n in range(NSWEEPS):
    samples = sampler.sweep()
    # compute the RefModel internals once, which all Oloc calls below reuse. psi is
    # needed for the reweighting factor before the first Oloc call, in case the
    # sampler doesn't return it.
    if state.use_ref:
        samples.psi, samples.state_internal = state.init_internal(samples.spins)
    elif samples.psi is None:
        samples.psi = state(samples.spins)
    samples.update_r()
    rw = 1.0 if samples.r is None else samples.r

    # correlations are real for a time-reversal symmetric state; stay in JAX
    # (jnp, not np) for every term, so there is no gather per term (each Oloc
    # still syncs once with the host)
    F_sweeps.append(
        jnp.stack([jnp.mean(jnp.real(op.Oloc(state, samples)) * rw) for op in F_ops])
    )
    G_sweeps.append(
        jnp.stack([jnp.mean(jnp.real(op.Oloc(state, samples)) * rw) for op in G_ops])
    )
    if (n + 1) % SAVE_EVERY == 0 or n + 1 == NSWEEPS:
        save(F_sweeps, G_sweeps)
