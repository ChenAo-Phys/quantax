"""
Tests for ``quantax.model.backflow``: the neural-network ``DetBackflow`` and
``PfBackflow`` models that add a configuration-dependent correction ``U_1(n)`` to a
mean-field reference and evaluate a determinant / Pfaffian.

These pin the pieces that are easy to break silently:

- the class docstrings actually being attached as ``__doc__`` (they are discarded if
  placed after the equinox field annotations, which kills the rendered docs);
- the forward formula (row slicing and sign of the determinant / Pfaffian) against
  numpy for a constant backflow, which makes the backflowed orbitals fixed;
- the *default* mean-field reference being non-singular -- for spinful systems the
  clean Fermi-sea block structure with the ``skew_eye`` pairing makes
  ``U_0 J_0 U_0^T`` singular unless the degeneracy is broken at init;
- the accelerated ``ref_forward`` path (both the ``return_update`` and the combined
  branch, chaining one update into the next) agreeing with a direct ``__call__``, for
  spin / spinless / spinful systems and for the ``use_ref`` low-rank path as well as
  the ``use_ref=False`` fallback. The second update moves several fermions of one
  sector after the first one left ``internal.idx`` unsorted, a flipped spin is one
  fermion hop, spin flips changing :math:`S^z` change the reordering sign, and the
  update modes are upper bounds with unused slots;
- ``init_internal`` returning the full backflow wavefunction, not the mean-field
  reference of the low-rank updates kept in the internal state;
- net outputs that are not plain ``jax.Array`` (e.g. ``LogArray``) being accepted, in
  ``__call__`` and in ``ref_forward``.

The numerical checks run in double precision (``x64`` fixture, float64 models as by
default), where the accelerated updates agree with ``__call__`` to ~1e-13 relative, so
that a tolerance of 1e-9 can be used. Without x64, as on GPU by default, the models
compute in float32 and the chained updates reach ~1e-5 relative error; one float32
case covers that path with a looser tolerance.

Runtime is dominated by compilation, so model evaluations are jitted (eager evaluation
compiles every primitive separately), nets are initialized with numpy, and all
lattices have Nfmodes = 8 so that the eager compilations of the model constructors are
shared between tests. Every check jits a fresh closure, because the models read the
global lattice while tracing: a cached trace from a test on the other lattice with
equal array shapes would silently be reused.
"""

from itertools import combinations

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from quantax.sites import Chain
from quantax.global_defs import PARTICLE_TYPE, get_sites
from quantax.model import DetBackflow, PfBackflow
from quantax.utils import LogArray


# --------------------------------------------------------------------------- #
# helpers (one lattice per test; conftest resets Sites._SITES between tests)
# --------------------------------------------------------------------------- #
def _spinless_chain():
    # Ntotal = 6 leaves room for a Pfaffian backflow of rank 2 * d_eff = 4 < Ntotal
    # with d_eff = 2, where the quadratic term W^T J_0 W of the update is nonzero
    return Chain(
        8, boundary=1, particle_type=PARTICLE_TYPE.spinless_fermion, Nparticles=6
    )


def _spinful_chain():
    return Chain(
        4, boundary=1, particle_type=PARTICLE_TYPE.spinful_fermion, Nparticles=(2, 2)
    )


def _spin_chain():
    # a spin maps to two fermion modes (up and down), Nfmodes = 8 and Ntotal = 4
    return Chain(4, boundary=1)


_LATTICES = {
    "spinless": _spinless_chain,
    "spinful": _spinful_chain,
    "spin": _spin_chain,
    "spin-flip": _spin_chain,
}


class _Net(eqx.Module):
    """Tiny linear backflow net mapping the raw config to a (d_eff * Nfmodes) vector."""

    weight: jax.Array

    def __call__(self, s):
        return self.weight @ s.astype(self.weight.dtype)


class _LogArrayNet(eqx.Module):
    """Wraps a net so it returns a ``LogArray`` instead of a plain ``jax.Array``."""

    inner: _Net

    def __call__(self, s):
        return LogArray.from_value(self.inner(s))


class _ConstNet(eqx.Module):
    """Configuration-independent output, so U_0 + U_1 is a fixed orbital matrix."""

    x: jax.Array

    def __call__(self, s):
        return self.x


def _normal(shape, seed=0):
    # numpy init: an eager jax.random draw compiles anew for every shape and dtype
    return jnp.asarray(np.random.default_rng(seed).normal(size=shape))


def _make_net(d):
    # The model reshapes net(s) to (d_eff, Nfmodes); d is halved internally for spin.
    sites = get_sites()
    d_eff = d // 2 if sites.is_spinful else d
    shape = (d_eff * sites.Nfmodes, sites.Nmodes)
    return _Net(_normal(shape) / np.sqrt(sites.Nmodes))


def _config(occ_up, occ_dn=None):
    """Build a +-1 config with the listed occupied sites (and spin-down block)."""
    sites = get_sites()
    s = np.full(sites.Nmodes, -1)
    s[list(occ_up)] = 1
    if occ_dn is not None:
        s[sites.Nsites + np.asarray(occ_dn, dtype=int)] = 1
    return jnp.array(s, dtype=jnp.float32)


def _spinful_configs():
    """All 36 configurations of the spinful chain."""
    L = get_sites().Nsites
    pairs = list(combinations(range(L), 2))
    return jnp.stack([_config(up, dn) for up in pairs for dn in pairs])


def _assert_close(a, b, rtol=1e-9):
    a, b = np.atleast_1d(np.asarray(a)), np.atleast_1d(np.asarray(b))
    np.testing.assert_array_less(np.abs(a - b), rtol * np.abs(b) + 1e-30)


# --------------------------------------------------------------------------- #
# docstrings are attached (regression: must precede the eqx field annotations)
# --------------------------------------------------------------------------- #
def test_docstrings_are_attached():
    assert DetBackflow.__doc__ is not None
    assert "Determinant backflow" in DetBackflow.__doc__
    assert PfBackflow.__doc__ is not None
    assert "Pfaffian backflow" in PfBackflow.__doc__


# --------------------------------------------------------------------------- #
# forward formula: a constant backflow gives fixed orbitals U_0 + U_1
# --------------------------------------------------------------------------- #
def _pf4(A):
    """Pfaffian of a 4x4 antisymmetric matrix."""
    return A[0, 1] * A[2, 3] - A[0, 2] * A[1, 3] + A[0, 3] * A[1, 2]


@pytest.mark.parametrize("kind", ["det", "pf"])
def test_constant_backflow_matches_formula(x64, kind):
    # With a configuration-independent net output X of shape (d_eff, Nfmodes), the
    # backflowed orbitals U = U_0 + X^T W^T are fixed, so the model must be the
    # documented det(n * U) resp. pf(n * U J_0 U^T) on every config, with the rows
    # sliced in mode order. The QuSpin-ordering sign (-1)^(N(N-1)/2) that the models
    # share with GeneralDet / GeneralPf is +1 for N = 4.
    _spinful_chain()
    sites = get_sites()
    # d = 4 is halved for spinful fermions; d_eff = 2 > 1 makes the layout matter
    X = _normal((2, sites.Nfmodes), seed=1)
    if kind == "det":
        # an explicit U0 also exercises the user-provided reference path
        U0 = _normal((sites.Nfmodes, sites.Ntotal), seed=2)
        model = DetBackflow(_ConstNet(X), d=4, U0=U0)
    else:
        # an explicit pairing matrix is stored by its upper triangle
        J0 = np.asarray(_normal((sites.Nfmodes, sites.Nfmodes), seed=2))
        J0 = J0 - J0.T
        model = PfBackflow(_ConstNet(X), d=4, J0=jnp.asarray(J0))
        np.testing.assert_array_equal(model.J0_full, J0)
    U = np.asarray(model.U0 + X.T @ model.W.T)
    # Net outputs that aren't plain jax.Arrays (LogArray) must give the same result.
    wrapped = eqx.tree_at(lambda m: m.net, model, _LogArrayNet(model.net))
    configs = _spinful_configs()

    @eqx.filter_jit
    def values(model, wrapped):
        return jax.vmap(model)(configs).value(), jax.vmap(wrapped)(configs).value()

    psi, psi_wrapped = values(model, wrapped)
    expected = []
    for s in np.asarray(configs):
        Us = U[np.flatnonzero(s > 0)]
        expected.append(np.linalg.det(Us) if kind == "det" else _pf4(Us @ J0 @ Us.T))
    expected = np.asarray(expected)
    assert np.all(np.abs(expected) > 1e-6)
    _assert_close(psi, expected)
    _assert_close(psi_wrapped, expected)


def test_backflow_rejects_wrong_reference_shapes(x64):
    _spinful_chain()  # Nfmodes = 8, Ntotal = 4
    net = _ConstNet(jnp.zeros(8))  # never called, the checks run at construction
    with pytest.raises(ValueError, match="U0"):
        DetBackflow(net, d=2, U0=jnp.ones((8, 8)))
    with pytest.raises(ValueError, match="U0"):
        PfBackflow(net, d=2, U0=jnp.ones((8, 4)))
    with pytest.raises(ValueError, match="J0"):
        PfBackflow(net, d=2, U0=jnp.eye(8), J0=jnp.zeros((4, 4)))


# --------------------------------------------------------------------------- #
# default mean-field reference is non-singular (regression for spinful Pfaffian)
# --------------------------------------------------------------------------- #
def test_pfbackflow_spinful_reference_not_singular(x64):
    # Clean Fermi-sea + skew_eye pairing gives a singular U0 J0 U0^T on half the
    # configs; the init must break that degeneracy so init_internal can invert it.
    _spinful_chain()
    pf = PfBackflow(_make_net(2), d=2)
    J0 = np.asarray(pf.J0_full)
    U0 = np.asarray(pf.U0)
    conds = []
    for s in np.asarray(_spinful_configs()):
        U = U0[np.flatnonzero(s > 0)]
        conds.append(np.linalg.cond(U @ J0 @ U.T))
    assert np.max(conds) < 1e6  # would be inf without the symmetry-breaking noise


# --------------------------------------------------------------------------- #
# ref_forward (low-rank path) matches __call__: both branches and chaining
# --------------------------------------------------------------------------- #
def _ref_forward_chain(model, s0, s1, s2, nflips):
    # s0 -> s1 and s1 -> s2 with the update modes {"nflips": n} for n in ``nflips``,
    # the second one starting from the internal state returned by the first update.
    mode1, mode2 = ({"nflips": n} for n in nflips)

    @eqx.filter_jit
    def check(model):
        psi0, internal = model.init_internal(s0)
        psi1 = model.ref_forward(s1, s0, mode1, internal, False)
        psi1u, internal1 = model.ref_forward(s1, s0, mode1, internal, True)
        psi2 = model.ref_forward(s2, s1, mode2, internal1, False)
        psi2u, _ = model.ref_forward(s2, s1, mode2, internal1, True)
        ref = [psi0, psi1, psi1u, psi2, psi2u]
        ref = jnp.stack([psi.value() for psi in ref])
        # one vmapped direct evaluation compiles faster than one call per state
        direct = jax.vmap(model)(jnp.stack([s0, s1, s1, s2, s2])).value()
        return ref, direct

    return check(model)


# The first update leaves internal.idx unsorted, as its rows are replaced in place,
# and the second one then moves several fermions of one sector, whose rows must be
# matched by value (fermion_mf._occupied_positions), not by position.
# spinless: the hop 0 -> 6 (internal.idx = [6, 1, 2, 3, 4, 5]), declared in an update
# mode with an unused slot, then the double hop 1 -> 0, 6 -> 7.
_SPINLESS_HOPS = ((0, 1, 2, 3, 4, 5), (1, 2, 3, 4, 5, 6), (0, 2, 3, 4, 5, 7))
# spinful: the up hop 0 -> 2 (internal.idx = [2, 1 | 4, 5]), then the up hops
# 1 -> 0, 2 -> 3 and the down hop 0 -> 2 at once. s0 is one of the configs that are
# singular for the clean Fermi sea + skew_eye pairing.
_SPINFUL_HOPS = (((0, 1), (0, 1)), ((1, 2), (0, 1)), ((0, 3), (1, 2)))
# spin (up sites listed): every flipped spin hops a fermion between the two modes of
# its site, so an exchange is two hops. The exchange 0 <-> 3, declared in an update
# mode with unused slots, gives internal.idx = [3, 1 | 6, 4]; then two exchanges.
_SPIN_HOPS = ((0, 1), (1, 3), (0, 2))
# spin flips without Sz conservation change the number of up spins counted by
# fermion_inverse_sign: one flip, declared with an unused slot, then two flips
_SPIN_FLIPS = ((0, 1), (0,), (0, 1, 2))
_HOPS = {
    "spinless": _SPINLESS_HOPS,
    "spinful": _SPINFUL_HOPS,
    "spin": _SPIN_HOPS,
    "spin-flip": _SPIN_FLIPS,
}
_NFLIPS = {"spinless": (4, 4), "spinful": (2, 6), "spin": (4, 4), "spin-flip": (2, 2)}


@pytest.mark.parametrize(
    "cls, d, lattice",
    [
        (DetBackflow, 2, "spinless"),  # rank 2 < 6
        (PfBackflow, 2, "spinless"),  # rank 4 < 6, d_eff = 2
        (DetBackflow, 4, "spinful"),  # rank 2 < 4
        (PfBackflow, 2, "spinful"),  # rank 2 < 4
        (DetBackflow, 2, "spin"),  # rank 1 < 4
        (PfBackflow, 2, "spin"),  # rank 2 < 4
        (DetBackflow, 2, "spin-flip"),
        (PfBackflow, 2, "spin-flip"),
    ],
    ids=[
        "det-spinless",
        "pf-spinless",
        "det-spinful",
        "pf-spinful",
        "det-spin",
        "pf-spin",
        "det-spin-flip",
        "pf-spin-flip",
    ],
)
def test_ref_forward_matches_call(x64, cls, d, lattice):
    _LATTICES[lattice]()
    net = _make_net(d)
    if lattice == "spinful":
        # the accelerated path calls the net as well, so it must accept LogArrays
        net = _LogArrayNet(net)
    if cls is DetBackflow and lattice == "spinless":
        # A random reference skips the costly compilation of the default orbitals,
        # which the other cases cover (spinless and spinful PfBackflow initialize
        # them differently).
        model = cls(net, d=d, U0=_normal((8, 6), seed=3))
    else:
        model = cls(net, d=d)
    assert model.use_ref
    assert model.required_update_modes == ("nflips",)
    if lattice == "spinful":
        s0, s1, s2 = (_config(up, dn) for up, dn in _SPINFUL_HOPS)
    else:
        s0, s1, s2 = (_config(occ) for occ in _HOPS[lattice])
    _assert_close(*_ref_forward_chain(model, s0, s1, s2, _NFLIPS[lattice]))


def test_ref_forward_matches_call_float32():
    # The float32 path, as on GPU without x64 (the float64 models are float32 there).
    # A random reference skips the compilation of the default orbitals in float32.
    _spinful_chain()
    U0 = _normal((8, 8), seed=3)
    model = PfBackflow(_LogArrayNet(_make_net(2)), d=2, U0=U0, dtype=jnp.float32)
    assert model.U0.dtype == jnp.float32
    s0, s1, s2 = (_config(up, dn) for up, dn in _SPINFUL_HOPS)
    _assert_close(*_ref_forward_chain(model, s0, s1, s2, _NFLIPS["spinful"]), rtol=1e-4)


# --------------------------------------------------------------------------- #
# use_ref=False fallback: ref_forward defers to __call__ and stores no internal
# --------------------------------------------------------------------------- #
def _fallback(model, s0, s1):
    @eqx.filter_jit
    def check(model):
        psi0, internal = model.init_internal(s0)
        psi1 = model.ref_forward(s1, s0, {"nflips": 2}, internal, False)
        psi1u, internal1 = model.ref_forward(s1, s0, {"nflips": 2}, internal, True)
        ref = jnp.stack([psi0.value(), psi1.value(), psi1u.value()])
        direct = jax.vmap(model)(jnp.stack([s0, s1, s1])).value()
        return ref, direct, internal, internal1

    return check(model)


@pytest.mark.parametrize("cls, d", [(DetBackflow, 8), (PfBackflow, 4)])
def test_use_ref_false_fallback(x64, cls, d):
    _spinful_chain()
    model = cls(_make_net(d), d=d)  # rank 4 >= Ntotal = 4
    assert not model.use_ref
    s0 = _config((0, 1), (0, 1))
    s1 = _config((0, 2), (0, 1))
    ref, direct, internal, internal1 = _fallback(model, s0, s1)
    assert internal is None and internal1 is None
    _assert_close(ref, direct)
