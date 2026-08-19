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
    """
    spins = samples.spins
    nsamples = spins.shape[0]
    perm = jnp.asarray(symm._perm)  # (ngroup, Nmodes)
    perm_sign = jnp.asarray(symm._perm_sign)
    perm_inv = jnp.argsort(perm, axis=1)
    ngroup = perm.shape[0]

    if symm.is_fermion:
        if symm.Z2_inversion != 0:
            raise NotImplementedError("particle-hole sign not implemented")
        # _permutation_sign is vmapped over group elements; vmap over samples
        sign_fn = jax.vmap(_permutation_sign, in_axes=(0, None, None))
        sign_symm = sign_fn(spins, perm, perm_sign)  # (nsamples, ngroup)
        sign_inv = sign_fn(spins, perm_inv, perm_sign)
    else:
        sign_symm = sign_inv = jnp.ones((nsamples, ngroup))

    z2_flips = (1,) if symm.Z2_inversion == 0 else (1, -1)

    overlap = []
    error = []
    for z2 in z2_flips:
        for g in range(ngroup):
            psi_symm = state(z2 * spins[:, perm[g]])
            psi_inv = state(z2 * spins[:, perm_inv[g]])
            x = sign_symm[:, g] * (psi_symm / samples.psi).value()
            y = sign_inv[:, g] * (psi_inv / samples.psi).value()
            xmean = jnp.mean(x)
            ymean = jnp.mean(y)
            # covariance matrix of the two sample means (same samples -> correlated)
            cov = jnp.cov(jnp.stack([x, y])) / nsamples
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
    # state(s) pads the input batch to max_parallel, so a large training value
    # wastes compute here. Two options:
    #   max_parallel=None — no padding, but batch-size changes may trigger re-jit
    #   max_parallel=<training value> — exactly the training computation, but
    #     possibly much larger than needed for measurement
    state = qtx.state.Variational(model, param_file="params.eqx", max_parallel=None)
    sampler = qtx.sampler.SpinExchange(state, nsamples=1024)

    # attach psi so every get_overlap call reuses it
    samples = sampler.sweep()
    samples = qtx.sampler.Samples(samples.spins, state(samples.spins))

    symmetries = {
        "trans(1,0)": Translation((1, 0)),  # exact by CNN -> 1.0000 +- 0.0000
        "rot90": Rotation(np.pi / 2),
        "flip0": Flip(axis=0),
        "spin_inv": SpinInverse(),  # Z2 inversion for spin systems, not a perm
    }

    for name, symm in symmetries.items():
        overlap, error = get_overlap(state, samples, symm)
        print(f"=== {name} ===")
        for g, (o, e) in enumerate(zip(overlap, error)):
            print(f"  element {g}: {o:.4f} +- {e:.4f}")
