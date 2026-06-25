"""
Tests for ``quantax.model.jastrow``: the traditional two-body ``Jastrow`` factor and
the ``GeneralJastrow`` wrapper that multiplies a network factor onto a reference state.

These pin the pieces that are easy to break silently:

- the sublattice-invariant parametrization of ``W`` (parameter sharing, symmetry, and
  the number of free parameters), mirroring the index logic of the mean-field models;
- the *different* forward formulas for spin and fermion systems -- spins couple the
  :math:`\\pm 1` configuration while fermions couple the :math:`0/1` occupations
  (density-density / Gutzwiller form);
- ``GeneralJastrow`` reducing to the plain product ``net * mf`` when no translation
  symmetrization is requested, the inherited ``sublattice`` / ``holomorphic`` flags, and
  the accelerated ``ref_forward`` path agreeing with ``__call__``.
"""

import numpy as np
import jax.numpy as jnp
import pytest
from quantax.sites import Square, Chain
from quantax.global_defs import PARTICLE_TYPE
from quantax.symmetry import TransND
from quantax.model import Jastrow, GeneralJastrow, GeneralPf
from quantax.model.jastrow import _get_jastrow_index


# --------------------------------------------------------------------------- #
# helpers (one lattice per test; conftest resets Sites._SITES between tests)
# --------------------------------------------------------------------------- #
def _spin_square(L=2):
    return Square(L, boundary=1)  # spin: Nmodes = Nsites = L*L, ncells = L*L


def _spinless_chain(L=4, N=2):
    return Chain(
        L, boundary=1, particle_type=PARTICLE_TYPE.spinless_fermion, Nparticles=N
    )


def _spinful_chain(L=4, Nparticles=(2, 2)):
    return Chain(
        L,
        boundary=1,
        particle_type=PARTICLE_TYPE.spinful_fermion,
        Nparticles=Nparticles,
    )


def _occupation(s):
    return np.where(np.asarray(s) > 0, 1.0, 0.0)


# --------------------------------------------------------------------------- #
# _get_jastrow_index: parameter-sharing structure
# --------------------------------------------------------------------------- #
def test_get_jastrow_index_free_is_full_grid():
    # Without a sublattice every entry of the Nmodes x Nmodes matrix is independent.
    _spin_square(2)  # Nmodes = 4
    idx = _get_jastrow_index(None)
    assert idx.shape == (4, 4)
    assert np.array_equal(idx, np.arange(16).reshape(4, 4))


def test_get_jastrow_index_translation_ties_entries():
    # With full translation symmetry the coupling only depends on the displacement,
    # so a 4x4 matrix on a 2x2 lattice keeps one parameter per cell (4 of them).
    _spin_square(2)
    idx = _get_jastrow_index(TransND())
    assert idx.shape == (4, 4)
    assert int(idx.max()) + 1 == 4


# --------------------------------------------------------------------------- #
# Jastrow: parameter count and W structure
# --------------------------------------------------------------------------- #
def test_jastrow_param_count_spin():
    _spin_square(2)  # Nmodes = 4, ncells = 4
    assert Jastrow(dtype=jnp.float32).W.size == 16  # free: Nmodes**2
    assert Jastrow(sublattice=TransND(), dtype=jnp.float32).W.size == 4  # per cell


def test_jastrow_param_count_spinful_fermion_splits_spin_channels():
    # Spinful fermions get distinct up/down channels, so a translation-invariant W has
    # (2 spin channels)**2 * ncells = 4 * 4 = 16 parameters, not just ncells.
    _spinful_chain(4, (2, 2))  # Nmodes = 8, ncells = 4
    assert Jastrow(dtype=jnp.float32).W.size == 64  # free: Nmodes**2
    assert Jastrow(sublattice=TransND(), dtype=jnp.float32).W.size == 16


def test_jastrow_W_full_symmetric_and_translation_invariant():
    _spin_square(2)
    J = Jastrow(sublattice=TransND(), dtype=jnp.float32)
    W = np.asarray(J.W_full)
    assert W.shape == (4, 4)
    np.testing.assert_allclose(W, W.T, atol=1e-6)
    # invariant under every lattice translation: W[Pg, Pg] == W
    for p in np.asarray(TransND()._perm):
        np.testing.assert_allclose(W[np.ix_(p, p)], W, atol=1e-6)


# --------------------------------------------------------------------------- #
# Jastrow: dtype / holomorphic
# --------------------------------------------------------------------------- #
def test_jastrow_real_dtype_not_holomorphic():
    _spin_square(2)
    J = Jastrow(dtype=jnp.float32)
    assert J.holomorphic is False
    assert J.W.dtype == jnp.float32


def test_jastrow_complex_dtype_holomorphic():
    _spin_square(2)
    J = Jastrow(dtype=jnp.complex64)
    assert J.holomorphic is True
    assert J.W.dtype == jnp.complex64


# --------------------------------------------------------------------------- #
# Jastrow: forward formula (spin uses s_i, fermion uses occupations n_i)
# --------------------------------------------------------------------------- #
def test_jastrow_spin_forward_equals_spin_quadratic_form():
    _spin_square(2)
    J = Jastrow(dtype=jnp.float32)
    s = jnp.array([1, -1, 1, -1], dtype=jnp.float32)
    W = np.asarray(J.W_full)
    vec = np.asarray(s)
    expected = 0.5 * vec @ W @ vec  # log|psi| for a real Jastrow
    np.testing.assert_allclose(np.asarray(J(s).logabs), expected, rtol=1e-5)


def test_jastrow_spinless_fermion_forward_equals_density_form():
    # Fermions couple occupations n_i in {0, 1}, not the +-1 configuration.
    _spinless_chain(4, 2)
    J = Jastrow(dtype=jnp.float32)
    s = jnp.array([1, -1, 1, -1], dtype=jnp.float32)  # occupations [1, 0, 1, 0]
    W = np.asarray(J.W_full)
    n = _occupation(s)
    expected = 0.5 * n @ W @ n
    np.testing.assert_allclose(np.asarray(J(s).logabs), expected, rtol=1e-5)


def test_jastrow_spinful_fermion_forward_equals_density_form():
    _spinful_chain(4, (2, 2))  # Nmodes = 8: [up(4), dn(4)]
    J = Jastrow(dtype=jnp.float32)
    s = jnp.array([1, -1, 1, -1, -1, 1, -1, 1], dtype=jnp.float32)
    W = np.asarray(J.W_full)
    n = _occupation(s)
    expected = 0.5 * n @ W @ n
    np.testing.assert_allclose(np.asarray(J(s).logabs), expected, rtol=1e-5)


def test_jastrow_fermion_ignores_empty_modes():
    # Density-density form: emptying an occupied mode removes all its couplings, while
    # the +-1 (spin) form would still contribute. This distinguishes the two.
    _spinless_chain(4, 2)
    J = Jastrow(dtype=jnp.float32)
    W = np.asarray(J.W_full)
    s = jnp.array([1, 1, -1, -1], dtype=jnp.float32)
    n = _occupation(s)
    np.testing.assert_allclose(np.asarray(J(s).logabs), 0.5 * n @ W @ n, rtol=1e-5)
    # not equal to the spin quadratic form (guards against using s instead of n)
    spin_form = 0.5 * np.asarray(s) @ W @ np.asarray(s)
    assert not np.isclose(float(np.asarray(J(s).logabs)), float(spin_form))


def test_jastrow_amplitude_translation_invariant():
    # sublattice=TransND() makes the amplitude itself invariant under translations.
    _spin_square(2)
    J = Jastrow(sublattice=TransND(), dtype=jnp.float32)
    s = jnp.array([1, 1, -1, -1], dtype=jnp.float32)
    base = complex(J(s).value())
    for p in np.asarray(TransND()._perm):
        np.testing.assert_allclose(complex(J(s[p]).value()), base, rtol=1e-5)


# --------------------------------------------------------------------------- #
# GeneralJastrow: construction
# --------------------------------------------------------------------------- #
def test_generaljastrow_sublattice_none_without_trans_symm():
    _spin_square(2)
    gj = GeneralJastrow(Jastrow(dtype=jnp.float32), GeneralPf(), trans_symm=None)
    assert gj.sublattice is None


def test_generaljastrow_inherits_sublattice_from_reference():
    _spin_square(2)
    pf = GeneralPf(sublattice=TransND())
    gj = GeneralJastrow(Jastrow(dtype=jnp.float32), pf, trans_symm=None)
    assert gj.sublattice == (1, 1)


def test_generaljastrow_holomorphic_requires_both_complex():
    _spin_square(2)
    real_net, cplx_net = Jastrow(dtype=jnp.float32), Jastrow(dtype=jnp.complex64)
    real_mf, cplx_mf = GeneralPf(dtype=jnp.float32), GeneralPf(dtype=jnp.complex64)
    assert GeneralJastrow(real_net, real_mf, None).holomorphic is False
    assert GeneralJastrow(cplx_net, real_mf, None).holomorphic is False
    assert GeneralJastrow(real_net, cplx_mf, None).holomorphic is False
    assert GeneralJastrow(cplx_net, cplx_mf, None).holomorphic is True


# --------------------------------------------------------------------------- #
# GeneralJastrow: forward and accelerated update
# --------------------------------------------------------------------------- #
def test_generaljastrow_forward_is_product_without_symmetrization():
    # With trans_symm=None the wavefunction is simply net(s) * mf(s).
    _spin_square(2)
    pf = GeneralPf()
    jas = Jastrow(sublattice=TransND(), dtype=jnp.float32)
    gj = GeneralJastrow(jas, pf, trans_symm=None)
    s = jnp.array([1, 1, -1, -1], dtype=jnp.float32)
    expected = (pf(s) * jas(s)).value()
    np.testing.assert_allclose(complex(gj(s).value()), complex(expected), rtol=1e-5)


def test_generaljastrow_ref_forward_matches_call():
    # The accelerated low-rank update must reproduce a direct evaluation after a
    # two-spin exchange (nflips=2).
    _spin_square(2)
    gj = GeneralJastrow(
        Jastrow(sublattice=TransND(), dtype=jnp.float32), GeneralPf(), trans_symm=None
    )
    s_old = jnp.array([1, 1, -1, -1], dtype=jnp.float32)
    s = jnp.array([1, -1, -1, 1], dtype=jnp.float32)  # exchange modes 1 and 3
    _, internal = gj.init_internal(s_old)
    psi_ref = gj.ref_forward(s, s_old, {"nflips": 2}, internal)
    psi_direct = gj(s)
    np.testing.assert_allclose(
        complex(psi_ref.value()), complex(psi_direct.value()), rtol=1e-5
    )
