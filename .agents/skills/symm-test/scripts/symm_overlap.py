"""Symmetry overlap test on the 4x4 Heisenberg model (see ../SKILL.md).

Loads a trained ResConv (from tutorials/J1J2.ipynb) from a weight file, then
measures O_T = <psi|T|psi><psi|T^-1|psi>/<psi|psi>^2 for each group element T.
"""

import numpy as np
import jax
import jax.numpy as jnp
import quantax as qtx
from quantax.symmetry import Translation, Rotation, Flip, SpinInverse
from quantax.symmetry.symmetry import _permutation_sign


def get_overlap(state, samples, symm):
    """Overlap <psi|T|psi><psi|T^-1|psi>/<psi|psi>^2 per group element of symm.

    Returns (overlap, error) arrays ordered as symm._perm; if
    symm.Z2_inversion != 0 (spin systems), the same permutations composed with
    s -> -s follow. Assumes the trivial sector (characters ignored).
    samples.psi must be state(samples.spins); samples.r, if present, reweights
    samples not drawn from |psi|^2.
    """
    spins = samples.spins
    nsamples = spins.shape[0]
    perm = jnp.asarray(symm._perm)  # (ngroup, Nmodes)
    ngroup = perm.shape[0]

    # T^-1 is the group element whose permutation inverts perm[g]. Reuse its sign
    # column and forward: under anti-periodic boundaries the fermion sign of T^-1
    # comes from its own perm_sign row, not from perm_sign[g].
    perm_np = np.asarray(symm._perm, dtype=np.int64)  # argsort returns int64
    lookup = {row.tobytes(): i for i, row in enumerate(perm_np)}
    try:
        inv_idx = [lookup[row.tobytes()] for row in np.argsort(perm_np, axis=1)]
    except KeyError:
        raise ValueError("symm._perm is not closed under inversion") from None

    if symm.is_fermion:
        if symm.Z2_inversion != 0:
            raise NotImplementedError("particle-hole sign not implemented")
        # _permutation_sign is vmapped over group elements; vmap over samples
        sign_fn = jax.vmap(_permutation_sign, in_axes=(0, None, None))
        sign_symm = sign_fn(spins, perm, symm._perm_sign)  # (nsamples, ngroup)
    else:
        sign_symm = jnp.ones((nsamples, ngroup))

    r = 1.0 if samples.r is None else samples.r  # normalized to mean 1
    z2_flips = (1,) if symm.Z2_inversion == 0 else (1, -1)

    overlap = []
    error = []
    for z2 in z2_flips:
        # sigma_T(s) psi(Ts) / psi(s), ratio formed before leaving LogArray
        xs = []
        for g in range(ngroup):
            ratio = state(z2 * spins[:, perm[g]]) / samples.psi
            xs.append(sign_symm[:, g] * jnp.asarray(ratio))
        for g in range(ngroup):
            x, y = xs[g], xs[inv_idx[g]]
            xmean = jnp.mean(r * x)
            ymean = jnp.mean(r * y)
            # covariance matrix of the two sample means (same samples -> correlated),
            # r * (x - xmean) is the delta-method term of the reweighted mean
            dxy = jnp.stack([r * (x - xmean), r * (y - ymean)])
            cov = jnp.cov(dxy) / nsamples
            var = (
                ymean**2 * cov[0, 0]
                + xmean**2 * cov[1, 1]
                + 2 * xmean * ymean * cov[0, 1]
            )
            overlap.append((xmean * ymean).real)
            error.append(jnp.sqrt(jnp.abs(var)).real)

    return jnp.array(overlap), jnp.array(error)


if __name__ == "__main__":
    lattice = qtx.sites.Square(4, Nparticles=(8, 8))

    # the model must match the architecture used to produce the weight file
    model = qtx.model.ResConv(nblocks=2, channels=8, kernel_size=3)
    nsamples = 1024
    # Every state(...) call in get_overlap evaluates exactly nsamples configs, so a
    # per-device chunk of nsamples // ndevices gives one unpadded chunk and a single
    # executable. Unlike Oloc (measure skill) there are no connected configs, so a
    # larger chunk only pads. max_parallel counts unexpanded configs: under
    # Variational(symm=...) do NOT multiply it by nsymm.
    max_parallel = nsamples // jax.device_count()
    state = qtx.state.Variational(
        model, param_file="params.eqx", max_parallel=max_parallel
    )
    # random start + 20 * Nmodes thermal steps by default; strongly ordered states
    # may need initial_spins=<training chains> or a larger thermal_steps
    sampler = qtx.sampler.SpinExchange(state, nsamples=nsamples)

    s = sampler.sweep()
    # recompute psi with state(): numerators and denominator share one executable
    samples = qtx.sampler.Samples(s.spins, psi=state(s.spins), prob=s.prob)
    samples.update_r()

    symmetries = {
        "trans(1,0)": Translation((1, 0)),  # exact by CNN -> 1.0000 +- 0.0000
        "rot90": Rotation(np.pi / 2),
        "flip0": Flip(axis=0),  # default center = lattice mean, bond-centred here
        "spin_inv": SpinInverse(),  # Z2 inversion for spin systems, not a perm
        "flip0*spin_inv": Flip(axis=0) @ SpinInverse(),
        "flip0_all_centers": Flip(axis=0) @ Translation((1, 0)),
    }

    for name, symm in symmetries.items():
        overlap, error = get_overlap(state, samples, symm)
        print(f"=== {name} ===")
        for g, (o, e) in enumerate(zip(overlap, error)):
            print(f"  element {g}: {o:.4f} +- {e:.4f}")
