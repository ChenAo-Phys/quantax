"""
Direct evaluation (``__call__``) of the mean-field models in
``quantax.model.fermion_mf`` against closed-form equivalent models, on every
configuration of small fixed-particle-number sectors, and on a spin chain without
:math:`S^z` conservation.
"""

import itertools
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from quantax.sites import Chain
from quantax.global_defs import PARTICLE_TYPE
from quantax.model import GeneralDet, GeneralPf, SingletPair, MultiPf, PartialPair

from _dtype import use_dtype
from _fermion import pfaffian


def _sector_configs(L, N):
    """All +-1 configurations of N fermions on L modes."""
    configs = []
    for occ in itertools.combinations(range(L), N):
        s = -np.ones(L)
        s[list(occ)] = 1
        configs.append(s)
    return np.stack(configs)


def _spinful_sector_configs(L, Nup, Ndn):
    up, dn = _sector_configs(L, Nup), _sector_configs(L, Ndn)
    return np.concatenate(
        [np.repeat(up, len(dn), axis=0), np.tile(dn, (len(up), 1))], axis=1
    )


def _amplitudes(model, spins):
    return np.asarray(jax.jit(jax.vmap(lambda s: model(s).value()))(spins))


def test_pf_models_match_pfaffian_formula(x64):
    r"""
    ``GeneralPf`` and ``MultiPf`` are :math:`\sum_k \mathrm{pf}(F_k[n, n])` times
    the QuSpin reordering sign :math:`(-1)^{N(N-1)/2}`, which is odd for N = 6. The
    Fock-space tests of the mean-field states rely on this formula.
    """
    use_dtype(jnp.complex128)
    L, N = 8, 6
    Chain(L, boundary=1, particle_type=PARTICLE_TYPE.spinless_fermion, Nparticles=N)
    spins = _sector_configs(L, N)  # 28 configurations
    rng = np.random.default_rng(2)
    A = rng.standard_normal((2, L, L)) + 1j * rng.standard_normal((2, L, L))
    F = A - np.swapaxes(A, 1, 2)

    def formula(Fs):
        out = []
        for s in spins:
            idx = np.flatnonzero(s > 0)
            out.append(-sum(pfaffian(Fk[np.ix_(idx, idx)]) for Fk in Fs))
        return np.asarray(out)

    psi = _amplitudes(GeneralPf(F=jnp.asarray(F[0])), jnp.asarray(spins))
    np.testing.assert_allclose(psi, formula(F[:1]), rtol=1e-10)
    psi = _amplitudes(MultiPf(npfs=2, F=jnp.asarray(F)), jnp.asarray(spins))
    np.testing.assert_allclose(psi, formula(F), rtol=1e-10)


def test_partialpair_limits_are_pfaffian_and_determinant(x64):
    r"""
    ``PartialPair`` evaluates pf([[Up J Up^T, U], [-U^T, 0]]) on the occupied rows.
    Without unpaired orbitals it is the Pfaffian of F = Up J Up^T, and with all
    particles unpaired the paired block drops out,
    pf([[A, U], [-U^T, 0]]) = (-1)^{n(n-1)/2} det(U) for a square U, so it is
    proportional to the determinant of the unpaired orbitals. Complex parameters
    tell the transposes from adjoints; under a real default dtype, they also pin that
    the parameter dtype comes from the given matrices.
    """
    use_dtype(jnp.float64)
    L, N = 6, 4
    Chain(L, boundary=1, particle_type=PARTICLE_TYPE.spinless_fermion, Nparticles=N)
    spins = jnp.asarray(_sector_configs(L, N))  # 15 configurations
    rng = np.random.default_rng(0)

    def cplx(shape):
        return jnp.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))

    U = cplx((L, L))
    paired = PartialPair(0, U=U, J=cplx((L, L)))
    pf = GeneralPf(F=paired.F_full)
    np.testing.assert_allclose(_amplitudes(paired, spins), _amplitudes(pf, spins))

    # the paired block drops out, so J is left to its default; the dtype is taken
    # from the complex U, which the real default dtype would truncate
    unpaired = PartialPair(N, U=U)
    assert unpaired.holomorphic
    det = GeneralDet(U=U[:, :N])
    ratio = _amplitudes(unpaired, spins) / _amplitudes(det, spins)
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-8)
    assert abs(ratio[0]) > 1e-3

    # a real U with a complex J is not truncated either; F is built from the inputs
    Ur, J = U.real, cplx((L, L))
    mixed = PartialPair(0, U=Ur, J=J)
    assert mixed.holomorphic
    pf = GeneralPf(F=Ur @ ((J - J.T) / 2) @ Ur.T)
    np.testing.assert_allclose(_amplitudes(mixed, spins), _amplitudes(pf, spins))


def test_singletpair_matches_block_generalpf(x64):
    r"""
    The singlet pairing det(F[up, dn]) is the Pfaffian of [[0, F], [-F^T, 0]], since
    pf([[0, B], [-B^T, 0]]) = (-1)^{n(n-1)/2} det(B). This pins the reordering sign
    of ``SingletPair``, which its low-rank updates must reproduce.
    """
    use_dtype(jnp.complex128)
    L = 4
    Chain(L, boundary=1, particle_type=PARTICLE_TYPE.spinful_fermion, Nparticles=(2, 2))
    # 36 configurations, n = 2 has an odd reordering sign
    spins = jnp.asarray(_spinful_sector_configs(L, 2, 2))
    rng = np.random.default_rng(1)
    F = rng.standard_normal((L, L)) + 1j * rng.standard_normal((L, L))
    Fpf = np.block([[np.zeros((L, L)), F], [-F.T, np.zeros((L, L))]])

    psi_sp = _amplitudes(SingletPair(F=jnp.asarray(F)), spins)
    psi_pf = _amplitudes(GeneralPf(F=jnp.asarray(Fpf)), spins)
    np.testing.assert_allclose(psi_sp, psi_pf, rtol=1e-10)


@pytest.mark.parametrize("model_cls", [GeneralDet, GeneralPf])
def test_spin_amplitude_is_site_ordered_projection(model_cls, x64):
    r"""
    On a spin lattice, the mean-field amplitudes with the reordering sign of the
    spin-up partons are the Gutzwiller projection with one parton per site created
    in site order, times the single-site gauge sign
    :math:`(-1)^{\sum_{i \in \uparrow} i}`. So they span the projected mean-field
    states also without :math:`S^z` conservation, which an :math:`S^z`-dependent
    sign (e.g. dropping the reordering sign for spins) would break.
    """
    use_dtype(jnp.float64)
    L = 4
    Chain(L, boundary=1)  # no Sz conservation, 16 configurations
    spins = np.array(list(itertools.product([1.0, -1.0], repeat=L)))
    rng = np.random.default_rng(3)
    if model_cls is GeneralDet:
        U = rng.standard_normal((2 * L, L))
        model = GeneralDet(U=jnp.asarray(U))
        amplitude = lambda rows: np.linalg.det(U[rows])
    else:
        A = rng.standard_normal((2 * L, 2 * L))
        F = A - A.T
        model = GeneralPf(F=jnp.asarray(F))
        amplitude = lambda rows: pfaffian(F[np.ix_(rows, rows)])

    expected = []
    for s in spins:
        rows = [i if s[i] > 0 else L + i for i in range(L)]
        expected.append((-1) ** np.sum(np.flatnonzero(s > 0)) * amplitude(rows))
    psi = _amplitudes(model, jnp.asarray(spins))
    np.testing.assert_allclose(psi, expected, rtol=1e-10, atol=1e-12)
