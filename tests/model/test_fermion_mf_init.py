"""
Construction-level tests for the mean-field models in ``quantax.model.fermion_mf``.

These complement the physics-level checks in ``tests/state/test_fermion_mf.py``
(Wick contractions / exact dense expectations) by pinning the parameter-init logic:
orbital/pairing-matrix shapes, the spin block structure, dtype handling, and the
input-validation error paths. They also guard the two shared initializers
``_init_det_orbs`` / ``_init_pf_orbs`` that ``GeneralDet``/``MultiDet`` and
``GeneralPf``/``MultiPf`` build on, so the determinant and Pfaffian families cannot
silently diverge again.
"""

import numpy as np
import jax.numpy as jnp
import pytest
from quantax.sites import Chain
from quantax.global_defs import PARTICLE_TYPE
from quantax.model import (
    GeneralDet,
    RestrictedDet,
    UnrestrictedDet,
    MultiDet,
    GeneralPf,
    SingletPair,
    MultiPf,
    PartialPair,
)
from quantax.model.fermion_mf import _init_det_orbs, _init_pf_orbs


# --------------------------------------------------------------------------- #
# helpers (one lattice per test; conftest resets Sites._SITES between tests)
# --------------------------------------------------------------------------- #
def _spinless_chain(L=6, N=3):
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


def _is_antisym(F, atol=1e-6):
    F = np.asarray(F)
    return np.abs(F + np.swapaxes(F, -1, -2)).max() < atol


# --------------------------------------------------------------------------- #
# _init_det_orbs: shape (Nfmodes, Ntotal) and spin block-diagonal structure
# --------------------------------------------------------------------------- #
def test_init_det_orbs_spinless_shape():
    sites = _spinless_chain(6, 3)
    U = _init_det_orbs(jnp.float32)
    assert U.shape == (sites.Nfmodes, sites.Ntotal) == (6, 3)


@pytest.mark.parametrize(
    "Nparticles, Nup, Ndn",
    [((2, 2), 2, 2), ((2, 1), 2, 1), (4, 2, 2), (3, 1, 2)],
    ids=["tuple-eq", "tuple-neq", "int-even", "int-odd"],
)
def test_init_det_orbs_spinful_shape_and_block_diagonal(Nparticles, Nup, Ndn):
    # For spinful systems the orbital matrix is block-diagonal in spin: the spin-up
    # rows carry no weight on the spin-down columns and vice versa. The int case
    # splits Ntotal as (Ntotal // 2, Ntotal - Ntotal // 2), matching GeneralDet.
    sites = _spinful_chain(4, Nparticles)
    L = sites.Nsites
    U = np.asarray(_init_det_orbs(jnp.float32))
    assert U.shape == (sites.Nfmodes, sites.Ntotal) == (8, Nup + Ndn)
    # up rows (0:L) must be zero on the down columns (Nup:)
    assert np.abs(U[:L, Nup:]).max() == 0.0
    # down rows (L:) must be zero on the up columns (:Nup)
    assert np.abs(U[L:, :Nup]).max() == 0.0


# --------------------------------------------------------------------------- #
# GeneralDet construction
# --------------------------------------------------------------------------- #
def test_generaldet_default_shape_spinless():
    sites = _spinless_chain(6, 3)
    assert GeneralDet().U.shape == (sites.Nfmodes, sites.Ntotal)


def test_generaldet_default_shape_spinful_odd_total():
    # Odd integer total: Ntotal columns must still be produced (1 up + 2 down here).
    sites = _spinful_chain(4, 3)
    assert GeneralDet().U.shape == (sites.Nfmodes, sites.Ntotal) == (8, 3)


def test_generaldet_rejects_wrong_U_shape():
    _spinless_chain(6, 3)
    with pytest.raises(ValueError, match="Expected U to have shape"):
        GeneralDet(U=jnp.zeros((6, 2)))


def test_generaldet_real_dtype_complex_out_dtype_storage():
    # When dtype is real and out_dtype complex, U stacks (real, imag) on a leading
    # axis but U_full materializes the complex matrix of shape (Nfmodes, Ntotal).
    sites = _spinless_chain(6, 3)
    det = GeneralDet(dtype=jnp.float32, out_dtype=jnp.complex64)
    assert det.U.shape == (2, sites.Nfmodes, sites.Ntotal)
    assert det.U.dtype == jnp.float32
    assert det.U_full.shape == (sites.Nfmodes, sites.Ntotal)
    assert det.U_full.dtype == jnp.complex64


def test_generaldet_requires_fixed_particle_number():
    Chain(
        4, boundary=1, particle_type=PARTICLE_TYPE.spinless_fermion
    )  # Nparticles=None
    with pytest.raises(ValueError, match="fixed amount of particles"):
        GeneralDet()


# --------------------------------------------------------------------------- #
# MultiDet construction (regression: odd spinful total must not collapse a column)
# --------------------------------------------------------------------------- #
def test_multidet_default_shape_spinless():
    sites = _spinless_chain(6, 3)
    md = MultiDet(ndets=3)
    assert md.U.shape == (3, sites.Nfmodes, sites.Ntotal)
    assert md.coeffs.shape == (3,)


def test_multidet_odd_spinful_total_keeps_all_columns():
    # Regression: previously the spinful int branch split Nparticles as (Nhalf, Nhalf),
    # dropping a column for odd totals and crashing slogdet on a non-square matrix.
    sites = _spinful_chain(4, 3)
    md = MultiDet(ndets=2)
    assert md.U.shape == (2, sites.Nfmodes, sites.Ntotal) == (2, 8, 3)
    # a valid 3-particle spinful configuration evaluates to a finite amplitude
    s = jnp.array([1, -1, -1, -1, -1, 1, -1, 1], dtype=jnp.float32)
    assert np.isfinite(complex(md(s).value()))


def test_multidet_provided_2d_U_is_broadcast_over_dets():
    sites = _spinless_chain(6, 3)
    U = jnp.zeros((sites.Nfmodes, sites.Ntotal))
    md = MultiDet(ndets=4, U=U)
    assert md.U.shape == (4, sites.Nfmodes, sites.Ntotal)


def test_multidet_rejects_wrong_U_shape():
    _spinless_chain(6, 3)
    with pytest.raises(ValueError, match="Expected U to have shape"):
        MultiDet(ndets=2, U=jnp.zeros((2, 6, 2)))


# --------------------------------------------------------------------------- #
# Restricted / Unrestricted determinant validation
# --------------------------------------------------------------------------- #
def test_restricteddet_requires_equal_spin_populations():
    _spinful_chain(4, (2, 1))
    with pytest.raises(ValueError, match="equal number of spin-up and spin-down"):
        RestrictedDet()


def test_restricteddet_default_shape():
    sites = _spinful_chain(4, (2, 2))
    assert RestrictedDet().U.shape == (sites.Nsites, 2)


def test_unrestricteddet_requires_spin_resolved_particle_number():
    # An integer (total-only) particle number is rejected; the message names the class.
    _spinful_chain(4, 4)
    with pytest.raises(ValueError, match="UnrestrictedDet requires"):
        UnrestrictedDet()


def test_unrestricteddet_default_shapes():
    sites = _spinful_chain(4, (2, 1))
    Uup, Udn = UnrestrictedDet().U
    assert Uup.shape == (sites.Nsites, 2)
    assert Udn.shape == (sites.Nsites, 1)


# --------------------------------------------------------------------------- #
# _init_pf_orbs: antisymmetric F of shape (Nfmodes, Nfmodes), optionally stacked
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex64])
def test_init_pf_orbs_spinless_antisymmetric(dtype):
    sites = _spinless_chain(6, 3)
    F = _init_pf_orbs(dtype)
    assert F.shape == (sites.Nfmodes, sites.Nfmodes)
    assert _is_antisym(F)


def test_init_pf_orbs_spinful_antisymmetric():
    sites = _spinful_chain(4, (2, 2))
    F = _init_pf_orbs(jnp.float32)
    assert F.shape == (sites.Nfmodes, sites.Nfmodes) == (8, 8)
    assert _is_antisym(F)


def test_init_pf_orbs_batched_shape_and_antisymmetric():
    # The npfs argument stacks independent antisymmetric matrices; this is the only
    # difference between the GeneralPf and MultiPf initialization paths.
    sites = _spinful_chain(4, (2, 2))
    F = _init_pf_orbs(jnp.float32, npfs=3)
    assert F.shape == (3, sites.Nfmodes, sites.Nfmodes)
    assert _is_antisym(F)


# --------------------------------------------------------------------------- #
# GeneralPf / MultiPf construction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "make_sites", [_spinless_chain, _spinful_chain], ids=["spinless", "spinful"]
)
def test_generalpf_default_F_full_antisymmetric(make_sites):
    sites = make_sites()
    F = GeneralPf().F_full
    assert F.shape == (sites.Nfmodes, sites.Nfmodes)
    assert _is_antisym(F)


def test_generalpf_rejects_wrong_F_shape():
    sites = _spinless_chain(6, 3)
    with pytest.raises(ValueError, match="Expected F to have shape"):
        GeneralPf(F=jnp.zeros((sites.Nfmodes, sites.Nfmodes + 1)))


def test_multipf_default_F_full_antisymmetric():
    sites = _spinless_chain(6, 3)
    F = MultiPf(npfs=2).F_full
    assert F.shape == (2, sites.Nfmodes, sites.Nfmodes)
    assert _is_antisym(F)


def test_multipf_rejects_wrong_F_shape():
    sites = _spinless_chain(6, 3)
    with pytest.raises(ValueError, match="Expected F to have shape"):
        MultiPf(npfs=2, F=jnp.zeros((3, sites.Nfmodes, sites.Nfmodes)))


# --------------------------------------------------------------------------- #
# SingletPair / PartialPair construction
# --------------------------------------------------------------------------- #
def test_singletpair_requires_spinful():
    _spinless_chain(6, 3)
    with pytest.raises(ValueError, match="SingletPair only works for spinful"):
        SingletPair()


def test_singletpair_default_F_shape():
    sites = _spinful_chain(4, (2, 2))
    assert SingletPair().F_full.shape == (sites.Nsites, sites.Nsites)


def test_partialpair_rejects_wrong_J_shape():
    # Nmodes = Nfmodes - Nunpaired = 6 - 2 = 4, so J must be (4, 4).
    _spinless_chain(6, 3)
    with pytest.raises(ValueError, match="Expected J to have shape"):
        PartialPair(Nunpaired=2, J=jnp.zeros((3, 3)))


def test_partialpair_default_F_full_antisymmetric():
    sites = _spinless_chain(6, 3)
    pp = PartialPair(Nunpaired=2)
    assert pp.U.shape == (sites.Nfmodes, sites.Nfmodes)
    assert _is_antisym(pp.F_full)


def test_partialpair_rejects_odd_unpaired():
    _spinless_chain(6, 3)
    with pytest.raises(NotImplementedError):
        PartialPair(Nunpaired=1)
