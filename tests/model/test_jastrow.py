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
  the accelerated ``ref_forward`` path agreeing with ``__call__``;
- the translation-symmetrized ``GeneralJastrow`` being the symmetry projection of the
  product state ``J_0 * mf``, with characters and fermion signs (and without them for
  spins), also when the sum is restricted to the sublattice of the reference state,
  and its accelerated update of the reference states of all translated configurations.

Model evaluations are jitted and vmapped over configurations: eager evaluation compiles
every primitive separately, which costs several times more. Every check jits a fresh
closure, because the models read the global lattice while tracing: a cached trace from
a test on another lattice with equal array shapes would silently be reused.
"""

from itertools import product
from types import SimpleNamespace

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
from quantax.sites import Square, Chain
from quantax.global_defs import PARTICLE_TYPE, get_sites
from quantax.symmetry import TransND, Translation
from quantax.model import Jastrow, GeneralJastrow, GeneralPf
from _dtype import use_dtype


# --------------------------------------------------------------------------- #
# helpers (one lattice per test; conftest resets Sites._SITES between tests)
# --------------------------------------------------------------------------- #
def _spin_square():
    return Square(2, boundary=1)  # spin: Nmodes = Nsites = 4, ncells = 4


def _spin_chain():
    return Chain(4, boundary=1)  # spin: Nmodes = ncells = 4


def _spinless_chain():
    return Chain(
        4, boundary=1, particle_type=PARTICLE_TYPE.spinless_fermion, Nparticles=2
    )


def _spinful_chain():
    return Chain(
        4, boundary=1, particle_type=PARTICLE_TYPE.spinful_fermion, Nparticles=(2, 2)
    )


_LATTICES = {
    "spin": _spin_square,
    "spin chain": _spin_chain,
    "spinless": _spinless_chain,
    "spinful": _spinful_chain,
}


def _all_configs():
    """Every +-1 configuration of the modes (particle numbers are not imposed)."""
    return jnp.array(list(product([1, -1], repeat=get_sites().Nmodes)), jnp.float32)


def _spinful_configs():
    """All 36 configurations of the spinful chain with (2, 2) fermions."""
    s = np.asarray(_all_configs()) > 0
    keep = (s[:, :4].sum(axis=1) == 2) & (s[:, 4:].sum(axis=1) == 2)
    return _all_configs()[keep]


def _spinful_hops():
    """A chain of single hops on the spinful chain, an up hop and a down hop 0 -> 3,
    both across the periodic boundary. The amplitudes stay >= 2e-3 of the largest
    one, while the near-free-fermion reference is tiny on many other configs."""
    occs = [((0, 1), (0, 1)), ((1, 3), (0, 1)), ((1, 3), (1, 3))]
    configs = -np.ones((3, 8), np.float32)
    for s, (up, dn) in zip(configs, occs):
        s[list(up)] = 1
        s[4 + np.asarray(dn)] = 1
    return jnp.asarray(configs)


def _spin_hops():
    """A chain of 2-spin exchanges on the spin chain, 1100 -> 1010 -> 0011, the last
    one across the periodic boundary."""
    s = [[1, 1, -1, -1], [1, -1, 1, -1], [-1, -1, 1, 1]]
    return jnp.asarray(s, jnp.float32)


def _rows(configs, *states):
    """Positions of the states in configs, to read their direct amplitudes from one
    vmapped evaluation instead of compiling a separate call for each."""
    configs = np.asarray(configs)
    return [np.flatnonzero((configs == np.asarray(s)).all(axis=1))[0] for s in states]


def _values(model, configs):
    return eqx.filter_jit(lambda m, x: jax.vmap(m)(x).value())(model, configs)


# --------------------------------------------------------------------------- #
# Jastrow: parameter sharing and W structure
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "lattice, n_free, n_trans",
    [
        # spin 2x2: Nmodes**2 = 16 free entries, one per displacement with translations
        ("spin", 16, 4),
        # spinful fermions get distinct up/down channels, so a translation-invariant W
        # has (2 spin channels)**2 * ncells = 16 parameters, not just ncells = 4
        ("spinful", 64, 16),
    ],
)
def test_jastrow_param_sharing(lattice, n_free, n_trans):
    _LATTICES[lattice]()
    assert Jastrow(dtype=jnp.float32).W.size == n_free
    J = Jastrow(sublattice=TransND(), dtype=jnp.float32)
    assert J.W.size == n_trans
    W = np.asarray(J.W_full)
    Nmodes = get_sites().Nmodes
    assert W.shape == (Nmodes, Nmodes)
    np.testing.assert_allclose(W, W.T, atol=1e-6)
    # invariant under every lattice translation (acting on both spin blocks), which
    # makes the amplitude exp(x W x / 2) translation invariant as well
    for p in np.asarray(TransND()._perm):
        np.testing.assert_allclose(W[np.ix_(p, p)], W, atol=1e-6)


# --------------------------------------------------------------------------- #
# Jastrow: forward formula (spin uses s_i, fermion uses occupations n_i)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "lattice, dtype",
    [
        ("spin", jnp.float32),
        ("spin", jnp.complex64),
        ("spinless", jnp.float32),
        ("spinful", jnp.float32),
    ],
)
def test_jastrow_forward_matches_quadratic_form(lattice, dtype):
    # psi = exp(x W x / 2) on every configuration, with x = s for spins and the
    # occupations n = (s + 1) / 2 for fermions. Using s for fermions would differ on
    # every config with an empty mode. A complex W gives a holomorphic factor whose
    # phase must be kept by the LogArray output.
    _LATTICES[lattice]()
    J = Jastrow(dtype=dtype)
    complex_ = jnp.issubdtype(dtype, jnp.complexfloating)
    assert J.holomorphic == complex_
    assert J.W.dtype == dtype
    configs = _all_configs()
    x = np.asarray(configs, dtype=np.float64)
    if get_sites().is_fermion:
        x = (x + 1) / 2
    W = np.asarray(J.W_full).astype(np.complex128)
    expected = np.exp(0.5 * np.einsum("bi,ij,bj->b", x, W, x))
    if complex_:  # a genuine, configuration-dependent phase
        assert np.ptp(np.angle(expected)) > 1e-3
    np.testing.assert_allclose(np.asarray(_values(J, configs)), expected, rtol=1e-5)


# --------------------------------------------------------------------------- #
# GeneralJastrow: flags inherited from the net and the reference state
# --------------------------------------------------------------------------- #
def test_generaljastrow_holomorphic_requires_both_complex():
    # Only the `holomorphic` attributes are read; a missing one counts as False.
    _spin_square()
    real, cplx = SimpleNamespace(holomorphic=False), SimpleNamespace(holomorphic=True)
    assert GeneralJastrow(cplx, cplx).holomorphic is True
    assert GeneralJastrow(real, cplx).holomorphic is False
    assert GeneralJastrow(cplx, real).holomorphic is False
    assert GeneralJastrow(cplx, SimpleNamespace()).holomorphic is False


def test_generaljastrow_sublattice_from_reference():
    # The symmetrization sum is restricted to the sublattice of the reference state;
    # without one it covers the whole lattice when trans_symm is given.
    _spin_square()
    net = SimpleNamespace()
    no_subl = SimpleNamespace()
    assert GeneralJastrow(net, no_subl, trans_symm=None).sublattice is None
    assert GeneralJastrow(net, no_subl, trans_symm=TransND()).sublattice == (2, 2)
    mf = SimpleNamespace(sublattice=TransND())  # unit vectors along both axes
    assert GeneralJastrow(net, mf, trans_symm=None).sublattice == (1, 1)
    # a tilted sublattice vector can't be expressed as per-axis sublattice sizes
    tilted = SimpleNamespace(sublattice=Translation([[1, 1]]))
    with pytest.raises(ValueError, match="along one axis"):
        GeneralJastrow(net, tilted, trans_symm=TransND())


# --------------------------------------------------------------------------- #
# GeneralJastrow without symmetrization: product form and accelerated update
# --------------------------------------------------------------------------- #
# The GeneralJastrow checks below share the spinful chain (and its TransND and
# GeneralPf constructions) with the Jastrow tests above, since every new lattice costs
# a fresh set of eager compilations.
def _product_and_ref_forward(gj, jas, pf, configs, s0, s1, s2):
    rows = _rows(configs, s0, s1, s2)

    @eqx.filter_jit
    def check(gj, jas, pf, configs):
        psi = jax.vmap(gj)(configs).value()
        product_ = jax.vmap(lambda s: pf(s) * jas(s))(configs).value()
        psi0, internal = gj.init_internal(s0)
        psi1, internal1 = gj.ref_forward(s1, s0, {"nflips": 2}, internal, True)
        psi2 = gj.ref_forward(s2, s1, {"nflips": 2}, internal1, False)
        ref = jnp.stack([x.value() for x in (psi0, psi1, psi2)])
        return psi, product_, ref, psi[np.array(rows)]

    return map(np.asarray, check(gj, jas, pf, configs))


def test_generaljastrow_without_trans_symm_is_product():
    # With trans_symm=None the wavefunction is simply net(s) * mf(s), and the low-rank
    # update of the reference reproduces it along a chain of hops.
    _spinful_chain()
    pf = GeneralPf()
    jas = Jastrow(sublattice=TransND(), dtype=jnp.float32)
    gj = GeneralJastrow(jas, pf, trans_symm=None)
    out = _product_and_ref_forward(gj, jas, pf, _spinful_configs(), *_spinful_hops())
    psi, product_, ref, direct = out
    np.testing.assert_allclose(psi, product_, rtol=1e-5)
    # float32 low-rank ratios lose ~1e-7 |psi_old / psi_new| relative accuracy, so the
    # tolerance is set by the largest amplitude (the chain stays above 2e-3 of it)
    np.testing.assert_allclose(ref, direct, rtol=0, atol=1e-5 * np.max(np.abs(psi)))


# --------------------------------------------------------------------------- #
# GeneralJastrow with translation symmetrization
# --------------------------------------------------------------------------- #
class _CellNet(eqx.Module):
    """Translation-covariant factors J_{c,r}(s) = exp(sum_{sigma k} w_{c sigma k}
    n_{sigma, r+k}) for channels c and cells r, like a convolutional net built with
    ``trans_symm=Identity()``. GeneralJastrow averages them over the channels."""

    w: jax.Array  # (channels, spin blocks, kernel size)

    def __call__(self, s):
        n = jnp.where(s > 0, 1.0, 0.0).reshape(self.w.shape[1], -1)
        x = sum(
            self.w[:, :, k, None] * jnp.roll(n, -k, axis=1)
            for k in range(self.w.shape[2])
        )
        return jnp.exp(x.sum(axis=1))  # (channels, cells)


# (channels, spin blocks, kernel size) weights of the _CellNet factors
_CELL_W = {
    "spinful": [
        [[0.3, -0.5, 0.2], [0.1, 0.4, -0.3]],
        [[-0.2, 0.1, 0.4], [0.5, -0.1, 0.2]],
    ],
    "spin chain": [[[0.3, -0.5, 0.2]], [[-0.2, 0.1, 0.4]]],
}


def _symmetrized_jastrow(lattice, sublattice):
    # k = pi on the 4-site chain: real characters +-1 compatible with the period-2
    # sublattice. For fermions, the pairs of same-spin fermions pick up signs when
    # translated across the boundary; spins have no such sign. The k = pi sectors of
    # the 36 spinful / 16 spin configurations are multi-dimensional, so
    # proportionality to the reference is a real constraint.
    _LATTICES[lattice]()
    symm = TransND(sector=2)
    net = _CellNet(jnp.array(_CELL_W[lattice]))
    pf = GeneralPf(sublattice=sublattice)
    return GeneralJastrow(net, pf, trans_symm=symm), net, pf, symm


def _symmetrized_checks(gj, net, pf, symm, configs, s0, s1, s2):
    rows = _rows(configs, s0, s1, s2)

    def projected(s, factor):
        # P [factor * mf](s), the factor being J_0 or 1
        s_symm = symm.get_symm_spins(s)
        x = jax.vmap(lambda t: pf(t) * factor(t))(s_symm)
        return symm.symmetrize(x, s).value()

    @eqx.filter_jit
    def check(gj, configs):
        psi = jax.vmap(gj)(configs).value()
        proj = jax.vmap(lambda s: projected(s, lambda t: net(t).mean(0)[0]))(configs)
        proj_mf = jax.vmap(lambda s: projected(s, lambda t: 1.0))(configs)
        # the reference states of all translated configurations are updated together
        psi0, internal = gj.init_internal(s0)
        psi1, internal1 = gj.ref_forward(s1, s0, {"nflips": 2}, internal, True)
        psi2 = gj.ref_forward(s2, s1, {"nflips": 2}, internal1, False)
        ref = jnp.stack([x.value() for x in (psi0, psi1, psi2)])
        return psi, proj, proj_mf, ref, psi[np.array(rows)]

    return map(np.asarray, check(gj, configs))


def _fit_residual(psi, ref):
    """Largest deviation of psi from its least-squares multiple of ref."""
    c = np.vdot(ref, psi) / np.vdot(ref, ref)
    return np.max(np.abs(psi - c * ref))


@pytest.mark.parametrize(
    "lattice, sublattice",
    [("spinful", None), ("spinful", (2,)), ("spin chain", None)],
)
def test_generaljastrow_trans_symm_projects_product_state(x64, lattice, sublattice):
    # J_r(s) = J_0(T_r s) for a covariant net (J_r being the channel mean), so
    # psi(s) = sum_r chi_r sign_r(s) J_r(s) mf(T_r s) is the momentum projection of
    # J_0 * mf, with sign_r = 1 for spins. With a period-2 reference only 2 of the 4
    # translations are summed, with J averaged over the equivalent cells, which is the
    # same projection up to a constant factor.
    # Double precision: the updated references include those of all translated
    # configurations, and some of them are ill-conditioned for some seeds. In float32
    # the spin-chain errors reached 9e-6 max|psi| for these exchanges (5e-4 for
    # others), even with O(1) amplitudes on the chain.
    use_dtype(jnp.float64)
    gj, net, pf, symm = _symmetrized_jastrow(lattice, sublattice)
    assert gj.sublattice == ((4,) if sublattice is None else sublattice)
    assert gj.required_update_modes == pf.required_update_modes
    if lattice == "spinful":
        configs, hops = _spinful_configs(), _spinful_hops()
    else:
        configs, hops = _all_configs(), _spin_hops()
    out = _symmetrized_checks(gj, net, pf, symm, configs, *hops)
    psi, proj, proj_mf, ref, direct = out
    # The k = pi projection has nodes, so the tolerances are set by the largest
    # amplitude (errors are <~1e-13 of it).
    scale = np.max(np.abs(psi))
    assert _fit_residual(psi, proj) < 1e-10 * scale
    # The sector is multi-dimensional: the projection of mf alone is a different state.
    assert _fit_residual(psi, proj_mf) > 1e-2 * scale
    np.testing.assert_allclose(ref, direct, rtol=0, atol=1e-10 * scale)


def test_generaljastrow_scalar_invariant_net_projects_product(x64):
    # A scalar factor is shared by all translations, J_T(s) = J(s), so for a
    # translation-invariant J the symmetrized product is the projection of J * mf,
    # J(s) sum_T chi_T sign_T(s) mf(T s) / |G|. A real factor and reference in the
    # complex k = 1 sector also pin the dtype: the projection is complex.
    use_dtype(jnp.complex128)
    _spinful_chain()
    symm = TransND(sector=1)
    net = Jastrow(sublattice=TransND(), dtype=jnp.float64)
    net = eqx.tree_at(lambda m: m.W, net, net.W * 30)  # J(s) far from 1
    pf = GeneralPf(dtype=jnp.float64)
    gj = GeneralJastrow(net, pf, trans_symm=symm)
    configs = _spinful_configs()
    s0, s1, _ = _spinful_hops()

    @jax.jit
    def check(configs):
        def projected(s):
            x = jax.vmap(lambda t: (net(t) * pf(t)).value())(symm.get_symm_spins(s))
            return symm.symmetrize(x.astype(jnp.complex128), s)

        psi = jax.vmap(lambda s: gj(s).value())(configs)
        J = jax.vmap(lambda s: jax.vmap(net)(symm.get_symm_spins(s)).value())(configs)
        psi0, internal = gj.init_internal(s0)
        psi1 = gj.ref_forward(s1, s0, {"nflips": 2}, internal, False)
        return psi, jax.vmap(projected)(configs), J, psi0.value(), psi1.value()

    psi, proj, J, psi0, psi1 = map(np.asarray, check(configs))
    scale = np.max(np.abs(psi))
    assert np.max(np.abs(J - J[:, :1])) < 1e-12 * np.max(np.abs(J))  # J is invariant
    np.testing.assert_allclose(psi, proj, rtol=0, atol=1e-12 * scale)
    assert np.max(np.abs(psi.imag)) > 1e-2 * scale
    direct = psi[np.array(_rows(configs, s0, s1))]
    np.testing.assert_allclose([psi0, psi1], direct, rtol=0, atol=1e-12 * scale)


def test_generaljastrow_rejects_non_invariant_jastrow():
    # a scalar Jastrow factor is shared by all translations, which only symmetrizes
    # the product if W is translation invariant; Jastrow() leaves every entry free
    _spinful_chain()
    with pytest.raises(ValueError, match="translation invariant"):
        GeneralJastrow(Jastrow(), GeneralPf(), trans_symm=TransND())
    GeneralJastrow(Jastrow(sublattice=TransND()), GeneralPf(), trans_symm=TransND())
