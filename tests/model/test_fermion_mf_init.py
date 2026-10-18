"""
Construction-level tests for the mean-field models in ``quantax.model.fermion_mf``.

These complement the physics-level checks in ``tests/state/test_fermion_mf.py``
(Wick contractions / exact dense expectations) by pinning the parameter-init logic:
the Fermi-sea default orbitals and their spin block structure, the storage of complex
parameters as real numbers, the sublattice-constrained pairing matrices, and the
input-validation error paths. They also guard the shared initializers
``_init_det_orbs`` / ``_init_pf_orbs`` that ``GeneralDet``/``MultiDet`` and
``GeneralPf``/``MultiPf`` build on (on spinful and spinless systems), so the
determinant and Pfaffian families cannot silently diverge again.
"""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
from quantax.sites import Chain, Sites
from quantax.symmetry import Translation
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
from quantax.model.fermion_mf import _init_det_orbs, _init_paired_orbs


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


def _hopping(L):
    """Nearest-neighbour hopping matrix of a periodic chain, h_ij = -1."""
    h = np.zeros((L, L))
    i = np.arange(L)
    h[i, (i + 1) % L] = h[(i + 1) % L, i] = -1.0
    return h


def _orbital_energy(U, h):
    """Hopping energy tr(h rho) of the Slater determinant of the columns of U."""
    U = np.asarray(U)
    rho = U @ np.linalg.inv(U.conj().T @ U) @ U.conj().T
    return np.trace(h @ rho).real


# --------------------------------------------------------------------------- #
# Fermi-sea default orbitals
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "Nparticles, Nup, Ndn",
    [((2, 1), 2, 1), (4, 2, 2), (3, 1, 2)],
    ids=["tuple", "int-even", "int-odd"],
)
def test_init_det_orbs_spinful_shape_and_block_diagonal(Nparticles, Nup, Ndn):
    # For spinful systems the orbital matrix is block-diagonal in spin: the spin-up
    # rows carry no weight on the spin-down columns and vice versa. The int case
    # splits Ntotal as (Ntotal // 2, Ntotal - Ntotal // 2), matching GeneralDet.
    # on a non-lattice Sites the spinless orbitals are random instead of a Fermi sea
    sites = Sites(4, particle_type=PARTICLE_TYPE.spinful_fermion, Nparticles=Nparticles)
    L = sites.Nsites
    U = np.asarray(_init_det_orbs(jnp.float32))
    assert U.shape == (sites.Nfmodes, sites.Ntotal) == (8, Nup + Ndn)
    # up rows (0:L) must be zero on the down columns (Nup:)
    assert np.abs(U[:L, Nup:]).max() == 0.0
    # down rows (L:) must be zero on the up columns (:Nup)
    assert np.abs(U[L:, :Nup]).max() == 0.0


# (model, spinful Nparticles or the number of spinless fermions)
_FERMI_SEA_CASES = {
    "generaldet-spinless": (GeneralDet, 3),
    "generaldet-spinful": (GeneralDet, (3, 1)),
    "restricteddet": (RestrictedDet, (3, 3)),
    "unrestricteddet": (UnrestrictedDet, (3, 1)),
    "multidet": (lambda: MultiDet(ndets=3), 3),
    # 2 unpaired fermions and one pair
    "partialpair-spinless": (lambda: PartialPair(Nunpaired=2), 4),
}


@pytest.mark.parametrize("case", _FERMI_SEA_CASES)
def test_default_orbitals_fill_the_fermi_sea(case):
    # The default orbitals are the lowest tight-binding orbitals plus 10% noise, so
    # the hopping energy is close to the exact Fermi sea. The 6-site chain is closed
    # shell for 3 and 1 fermions (levels -2 | -1, -1 | 1, 1, 2): a wrong orbital
    # choice costs >= 1 (>= 2 for 3 fermions), while the noise costs < 0.14 (60
    # seeds).
    make, Nparticles = _FERMI_SEA_CASES[case]
    L = 6
    h = _hopping(L)
    E0 = np.cumsum(np.linalg.eigvalsh(h))  # E0[n - 1]: Fermi sea of n fermions
    if isinstance(Nparticles, int):
        _spinless_chain(L, Nparticles)
    else:
        _spinful_chain(L, Nparticles)

    U = make().U_full
    # (orbitals, hopping matrix, Fermi-sea energy) of each determinant or spin block
    if case == "generaldet-spinful":
        blocks = [(U, np.kron(np.eye(2), h), E0[2] + E0[0])]
    elif case == "unrestricteddet":
        # unequal populations tell the spin-up and spin-down orbitals apart
        blocks = [(U[0], h, E0[2]), (U[1], h, E0[0])]
    elif case == "multidet":
        blocks = [(u, h, E0[2]) for u in U]
    elif case == "partialpair-spinless":
        # the unpaired orbitals are the lowest ones, the -1 level is degenerate;
        # slicing in numpy saves eager compilations
        blocks = [(np.asarray(U)[:, :2], h, E0[1])]
    else:  # 3 spinless fermions, or each spin of RestrictedDet
        blocks = [(U, h, E0[2])]

    # the lower bound is variational, up to the float32 rounding of the orbitals
    for orbitals, hopping, E_sea in blocks:
        assert E_sea - 1e-6 < _orbital_energy(orbitals, hopping) < E_sea + 0.5


# --------------------------------------------------------------------------- #
# complex parameters stored as real numbers (real dtype, complex out_dtype)
# --------------------------------------------------------------------------- #
def _cplx(rng, shape):
    return jnp.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


# model factories on a (2, 2) spinful 4-site chain: Nfmodes = 8, Ntotal = 4
_REAL_TO_CPL_CASES = {
    "generaldet": lambda rng, **kw: GeneralDet(U=_cplx(rng, (8, 4)), **kw),
    "restricteddet": lambda rng, **kw: RestrictedDet(U=_cplx(rng, (4, 2)), **kw),
    "unrestricteddet": lambda rng, **kw: UnrestrictedDet(
        U=(_cplx(rng, (4, 2)), _cplx(rng, (4, 2))), **kw
    ),
    "multidet": lambda rng, **kw: MultiDet(ndets=2, U=_cplx(rng, (2, 8, 4)), **kw),
    "generalpf": lambda rng, **kw: GeneralPf(F=_cplx(rng, (8, 8)), **kw),
    "singletpair": lambda rng, **kw: SingletPair(F=_cplx(rng, (4, 4)), **kw),
    "multipf": lambda rng, **kw: MultiPf(npfs=2, F=_cplx(rng, (2, 8, 8)), **kw),
    "partialpair": lambda rng, **kw: PartialPair(
        2, U=_cplx(rng, (8, 8)), J=jnp.asarray(rng.standard_normal((6, 6))), **kw
    ),
}


@pytest.mark.parametrize("case", _REAL_TO_CPL_CASES)
def test_real_dtype_complex_out_dtype_stores_real_and_imag(case):
    # With a real dtype and a complex out_dtype the complex parameters are stored as
    # stacked (real, imag) parts; the full matrices must reproduce the complex input
    # exactly, and the model is not holomorphic in its real parameters.
    _spinful_chain(4, (2, 2))
    make = _REAL_TO_CPL_CASES[case]

    @eqx.filter_jit
    def build(**dtypes):
        # with explicit parameters the construction is a pure function, so it is
        # jitted: one compilation instead of a dozen eager ones
        model = make(np.random.default_rng(0), **dtypes)
        names = [name for name in ("U_full", "F_full") if hasattr(model, name)]
        return model, [getattr(model, name) for name in names]

    cpl, cpl_full = build(dtype=jnp.complex64)
    split, split_full = build(dtype=jnp.float32, out_dtype=jnp.complex64)
    assert cpl.holomorphic and not split.holomorphic

    params = jax.tree.leaves(eqx.filter(split, eqx.is_inexact_array))
    assert all(p.dtype == jnp.float32 for p in params)
    assert len(jax.tree.leaves(split_full)) > 0
    for x, y in zip(jax.tree.leaves(cpl_full), jax.tree.leaves(split_full)):
        assert y.dtype == jnp.complex64
        np.testing.assert_allclose(y, x, rtol=1e-6)


# --------------------------------------------------------------------------- #
# input validation
# --------------------------------------------------------------------------- #
# wrong parameter shapes on a (2, 2) spinful 4-site chain: Nfmodes = 8, Ntotal = 4
_WRONG_SHAPES = {
    "generaldet": (lambda: GeneralDet(U=jnp.zeros((8, 3))), "Expected U"),
    "restricteddet": (lambda: RestrictedDet(U=jnp.zeros((4, 3))), "Expected U"),
    "unrestricteddet-up": (
        lambda: UnrestrictedDet(U=(jnp.zeros((4, 3)), jnp.zeros((4, 2)))),
        "Expected Uup",
    ),
    "unrestricteddet-dn": (
        lambda: UnrestrictedDet(U=(jnp.zeros((4, 2)), jnp.zeros((4, 3)))),
        "Expected Udn",
    ),
    "multidet": (lambda: MultiDet(ndets=2, U=jnp.zeros((2, 8, 3))), "Expected U"),
    "generalpf": (lambda: GeneralPf(F=jnp.zeros((8, 9))), "Expected F"),
    "singletpair": (lambda: SingletPair(F=jnp.zeros((4, 5))), "Expected F"),
    "multipf": (lambda: MultiPf(npfs=2, F=jnp.zeros((3, 8, 8))), "Expected F"),
    "partialpair-U": (lambda: PartialPair(2, U=jnp.zeros((8, 7))), "Expected U"),
    # Nmodes = Nfmodes - Nunpaired = 6, so J must be (6, 6)
    "partialpair-J": (
        lambda: PartialPair(2, U=jnp.zeros((8, 8)), J=jnp.zeros((5, 5))),
        "Expected J",
    ),
}


@pytest.mark.parametrize("case", _WRONG_SHAPES)
def test_rejects_wrong_parameter_shape(case):
    _spinful_chain(4, (2, 2))
    make, match = _WRONG_SHAPES[case]
    with pytest.raises(ValueError, match=match):
        make()


# unsupported systems: (lattice, constructor, error, message)
_UNSUPPORTED = {
    "generaldet-no-fixed-N": (
        lambda: Chain(4, particle_type=PARTICLE_TYPE.spinless_fermion),
        GeneralDet,
        ValueError,
        "fixed amount of particles",
    ),
    "multidet-no-fixed-N": (
        lambda: Chain(4, particle_type=PARTICLE_TYPE.spinless_fermion),
        MultiDet,
        ValueError,
        "fixed amount of particles",
    ),
    "restricteddet-spinless": (_spinless_chain, RestrictedDet, ValueError, "spinful"),
    "restricteddet-unequal": (
        lambda: _spinful_chain(4, (2, 1)),
        RestrictedDet,
        ValueError,
        "equal number of spin-up and spin-down",
    ),
    "unrestricteddet-spinless": (
        _spinless_chain,
        UnrestrictedDet,
        ValueError,
        "spinful",
    ),
    # an integer (total-only) particle number doesn't fix the spin populations
    "unrestricteddet-total-N": (
        lambda: _spinful_chain(4, 4),
        UnrestrictedDet,
        ValueError,
        "UnrestrictedDet requires",
    ),
    "singletpair-spinless": (
        _spinless_chain,
        SingletPair,
        ValueError,
        "SingletPair only works for spinful",
    ),
    "partialpair-odd-unpaired": (
        _spinless_chain,
        lambda: PartialPair(Nunpaired=1),
        NotImplementedError,
        None,
    ),
}


@pytest.mark.parametrize("case", _UNSUPPORTED)
def test_rejects_unsupported_system(case):
    make_sites, make_model, error, match = _UNSUPPORTED[case]
    make_sites()
    with pytest.raises(error, match=match):
        make_model()


# --------------------------------------------------------------------------- #
# determinant families: provided orbitals, odd spinful totals
# --------------------------------------------------------------------------- #
def test_multidet_odd_spinful_total_keeps_all_columns():
    # Regression: previously the spinful int branch split Nparticles as (Nhalf, Nhalf),
    # dropping a column for odd totals and crashing slogdet on a non-square matrix.
    sites = Sites(4, particle_type=PARTICLE_TYPE.spinful_fermion, Nparticles=3)
    md = MultiDet(ndets=2)
    assert md.U.shape == (2, sites.Nfmodes, sites.Ntotal) == (2, 8, 3)


def test_multidet_provided_2d_U_is_broadcast_over_dets():
    sites = _spinless_chain(6, 3)
    U = np.random.default_rng(0).standard_normal((sites.Nfmodes, sites.Ntotal))
    md = MultiDet(ndets=4, U=jnp.asarray(U, jnp.float32))
    np.testing.assert_array_equal(
        md.U, np.broadcast_to(U.astype(np.float32), md.U.shape)
    )


# --------------------------------------------------------------------------- #
# Pfaffian families: default pairing, sublattice constraint
# --------------------------------------------------------------------------- #
def test_default_spinful_pairing_is_a_singlet():
    # The default pairings are singlets, F = [[0, F0], [-F0^T, 0]] with a symmetric
    # F0: the same-spin blocks and the antisymmetric part of F0 carry only the 10%
    # noise. PartialPair pairs the up and down copies of each paired orbital.
    # MultiPf draws independent noise for its Pfaffians, so it does not start as npfs
    # identical ones, and SingletPair starts from the same F0 as the Pfaffians. The
    # ratios stay below 0.14 (40 seeds), and pairing up_k with down_{k+1} instead
    # gives an antisymmetric part >= 0.9.
    # The pairs carry zero momentum, so F0 is translation invariant up to the noise
    # (relative change < 0.19, 40 seeds).
    L = 4
    _spinful_chain(L, (2, 2))
    models = [MultiPf(npfs=2), PartialPair(Nunpaired=2), SingletPair()]
    # one compilation instead of many eager ones
    F_full = eqx.filter_jit(lambda ms: [m.F_full for m in ms])(models)
    F_multi, F_partial, F_singlet = map(np.asarray, F_full)
    for F in [*F_multi, F_partial]:
        singlet = np.abs(F[:L, L:]).mean()
        assert np.abs(F[:L, :L]).mean() < 0.3 * singlet
        assert np.abs(F[L:, L:]).mean() < 0.3 * singlet
    T = (np.arange(L) + 1) % L  # translation by one site
    for F0 in [*F_multi[:, :L, L:], F_partial[:L, L:], F_singlet]:
        assert np.linalg.norm(F0 - F0.T) < 0.3 * np.linalg.norm(F0 + F0.T)
        assert np.linalg.norm(F0[T][:, T] - F0) < 0.5 * np.linalg.norm(F0)

    assert np.abs(F_multi[0] - F_multi[1]).max() > 1e-3 * np.abs(F_multi).max()
    F0 = F_multi[0, :L, L:]
    assert np.linalg.norm(F_singlet - F0) < 0.3 * np.linalg.norm(F0)

    # Complex orbitals are plane waves, so zero momentum needs phi_k(up) paired with
    # phi_k^*(down), i.e. k with -k; pairing phi_k(up) with phi_k(down) has momentum
    # 2k and gives F0 = delta_{x+y,0}. The noise-free F0 is checked to save the
    # compilations of the complex models.
    F0 = np.asarray(_init_paired_orbs(jnp.complex64))
    np.testing.assert_allclose(F0, F0.T, atol=1e-6)
    np.testing.assert_allclose(F0[T][:, T], F0, atol=1e-6)


def test_default_spinless_pairing_is_random():
    # Spinless fermions have no singlets to pair, so the default Pfaffians are
    # random, with an independent draw for each MultiPf component.
    sites = _spinless_chain(6, 3)
    M = sites.Nfmodes
    F_general, F_multi = eqx.filter_jit(lambda ms: [m.F_full for m in ms])(
        [GeneralPf(), MultiPf(npfs=2)]
    )
    assert F_general.shape == (M, M)
    assert F_multi.shape == (2, M, M)
    assert np.abs(F_general).max() > 0
    assert np.abs(F_multi[0] - F_multi[1]).max() > 0.1 * np.abs(F_multi).max()


def _assert_translation_invariant(F_full, L, shift_sites=2):
    """F_full(T i, T j) = F_full(i, j) for T a shift by two sites, but not by one."""
    F_full = np.asarray(F_full)
    M = F_full.shape[0]

    def shift(d):  # i -> i + d within each spin block
        return (np.arange(M) % L + d) % L + np.arange(M) // L * L

    T = shift(shift_sites)
    np.testing.assert_array_equal(F_full[T][:, T], F_full)
    # still generic within the orbit structure: one-site shifts are no symmetry
    T1 = shift(1)
    assert not np.allclose(F_full[T1][:, T1], F_full)


def test_sublattice_pairing_is_translation_invariant():
    # A sublattice (2,) shares the pairing amplitudes between pairs related by a
    # translation of two sites, with one parameter per orbit of site pairs. The spin
    # label is an extra sublattice index of GeneralPf (4 x 4 x 3 orbits), while
    # SingletPair pairs up with down sites (2 x 2 x 3 orbits).
    L = 6
    _spinful_chain(L, (3, 3))
    rng = np.random.default_rng(1)
    F = rng.standard_normal((2 * L, 2 * L))
    F = jnp.asarray(F - F.T)
    F_sp = jnp.asarray(rng.standard_normal((L, L)))
    sublattice = Translation(np.diag((2,)))

    # the construction from explicit parameters is jitted, see above
    @eqx.filter_jit
    def build():
        pf = GeneralPf(F=F, sublattice=sublattice)
        sp = SingletPair(F=F_sp, sublattice=sublattice)
        return pf.F, pf.F_full, sp.F, sp.F_full

    pf_params, pf_full, sp_params, sp_full = build()
    assert pf_params.size == 4 * 4 * 3
    _assert_translation_invariant(pf_full, L)
    assert sp_params.size == 2 * 2 * 3
    _assert_translation_invariant(sp_full, L)

    # a tuple is the shape of the unit cell
    np.testing.assert_array_equal(GeneralPf(F=F, sublattice=(2,)).F, pf_params)
    with pytest.raises(ValueError, match="Invalid sublattice type"):
        GeneralPf(F=F, sublattice=2)
