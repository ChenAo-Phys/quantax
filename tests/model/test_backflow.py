"""
Tests for ``quantax.model.backflow``: the neural-network ``DetBackflow`` and
``PfBackflow`` models that add a configuration-dependent correction ``U_1(n)`` to a
mean-field reference and evaluate a determinant / Pfaffian.

These pin the pieces that are easy to break silently:

- the class docstrings actually being attached as ``__doc__`` (they are discarded if
  placed after the equinox field annotations, which kills the rendered docs);
- the *default* mean-field reference being non-singular -- for spinful systems the
  clean Fermi-sea block structure with the ``skew_eye`` pairing makes
  ``U_0 J_0 U_0^T`` singular unless the degeneracy is broken at init;
- the accelerated ``ref_forward`` path (both the ``return_update`` and the combined
  branch, and chaining one update into the next) agreeing with a direct ``__call__``,
  for spinless / spinful systems and for the ``use_ref`` low-rank path as well as the
  ``use_ref=False`` fallback;
- net outputs that are not plain ``jax.Array`` (e.g. ``LogArray`` / ``ScaleArray``)
  being accepted -- the model materializes them via ``jnp.asarray`` before reshaping.
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import pytest

from quantax.sites import Chain
from quantax.global_defs import PARTICLE_TYPE, get_sites
from quantax.model import DetBackflow, PfBackflow
from quantax.nn import fermion_idx
from quantax.utils import LogArray, ScaleArray


# --------------------------------------------------------------------------- #
# helpers (one lattice per test; conftest resets Sites._SITES between tests)
# --------------------------------------------------------------------------- #
def _spinless_chain(L=6, N=4):
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


class _Net(eqx.Module):
    """Tiny backflow net mapping the raw config to a (d_eff * Nfmodes) vector."""

    lin: eqx.nn.Linear

    def __init__(self, in_size, out_size, key):
        self.lin = eqx.nn.Linear(in_size, out_size, key=key)

    def __call__(self, s):
        return self.lin(s.astype(jnp.float32))


class _LogArrayNet(eqx.Module):
    """Wraps a net so it returns a ``LogArray`` instead of a plain ``jax.Array``."""

    inner: _Net

    def __call__(self, s):
        return LogArray.from_value(self.inner(s))


class _ScaleArrayNet(eqx.Module):
    """Wraps a net so it returns a ``ScaleArray`` instead of a plain ``jax.Array``."""

    inner: _Net

    def __call__(self, s):
        x = self.inner(s)
        return ScaleArray(x, jnp.zeros((), dtype=x.dtype))


def _make_net(d):
    # The model reshapes net(s) to (d_eff, Nfmodes); d is halved internally for spin.
    sites = get_sites()
    d_eff = d // 2 if sites.is_spinful else d
    return _Net(sites.Nfmodes, d_eff * sites.Nfmodes, jr.key(0))


def _config(occ_up, occ_dn=None):
    """Build a +-1 config with the listed occupied sites (and spin-down block)."""
    sites = get_sites()
    if occ_dn is None:
        s = np.full(sites.Nsites, -1)
        s[list(occ_up)] = 1
        return jnp.array(s, dtype=jnp.float32)
    up = np.full(sites.Nsites, -1)
    up[list(occ_up)] = 1
    dn = np.full(sites.Nsites, -1)
    dn[list(occ_dn)] = 1
    return jnp.array(np.concatenate([up, dn]), dtype=jnp.float32)


def _value(psi):
    return complex(psi.value())


def _rel_err(a, b):
    return abs(_value(a) - _value(b)) / (abs(_value(b)) + 1e-30)


# --------------------------------------------------------------------------- #
# docstrings are attached (regression: must precede the eqx field annotations)
# --------------------------------------------------------------------------- #
def test_docstrings_are_attached():
    assert DetBackflow.__doc__ is not None
    assert "Determinant backflow" in DetBackflow.__doc__
    assert PfBackflow.__doc__ is not None
    assert "Pfaffian backflow" in PfBackflow.__doc__


# --------------------------------------------------------------------------- #
# default mean-field reference is non-singular (regression for spinful Pfaffian)
# --------------------------------------------------------------------------- #
def test_pfbackflow_spinful_reference_not_singular():
    # Clean Fermi-sea + skew_eye pairing gives a singular U0 J0 U0^T on half the
    # configs; the init must break that degeneracy so init_internal can invert it.
    from itertools import combinations

    sites = _spinful_chain(4, (2, 2))
    pf = PfBackflow(_make_net(2), d=2, dtype=jnp.float32)
    J0 = np.asarray(pf.J0_full)
    U0 = np.asarray(pf.U0)
    conds = []
    for up in combinations(range(4), 2):
        for dn in combinations(range(4), 2):
            s = _config(up, dn)
            idx = np.asarray(fermion_idx(s))
            F = U0[idx, :] @ J0 @ U0[idx, :].T
            conds.append(np.linalg.cond(F))
    assert np.max(conds) < 1e6  # would be inf without the symmetry-breaking noise


def test_pfbackflow_init_internal_finite_spinful():
    _spinful_chain(4, (2, 2))
    pf = PfBackflow(_make_net(2), d=2, dtype=jnp.float32)
    # this config is one of the singular ones for the clean Fermi sea
    s = _config((0, 1), (0, 1))
    psi, internal = pf.init_internal(s)  # internal holds the mean-field inverse
    assert np.all(np.isfinite(np.asarray(internal.inv)))
    assert np.isfinite(_value(psi))
    assert np.isfinite(_value(pf(s)))


# --------------------------------------------------------------------------- #
# ref_forward (low-rank path) matches __call__: both branches and chaining
# --------------------------------------------------------------------------- #
def _check_ref_forward(model, s0, s1, s2, nflips=2):
    _, internal = model.init_internal(s0)

    # combined branch (no internal update) at s1
    psi1 = model.ref_forward(s1, s0, {"nflips": nflips}, internal, False)
    assert _rel_err(psi1, model(s1)) < 1e-4

    # return_update branch at s1, then chain the returned internal into s2
    psi1u, internal1 = model.ref_forward(s1, s0, {"nflips": nflips}, internal, True)
    assert _rel_err(psi1u, model(s1)) < 1e-4
    psi2 = model.ref_forward(s2, s1, {"nflips": nflips}, internal1, False)
    assert _rel_err(psi2, model(s2)) < 1e-4


def test_detbackflow_ref_forward_spinless():
    _spinless_chain(6, 4)
    model = DetBackflow(_make_net(2), d=2, dtype=jnp.float32)  # rank 2 < 4
    assert model.use_ref
    s0 = _config((0, 1, 2, 3))
    s1 = _config((0, 1, 2, 4))  # hop 3 -> 4
    s2 = _config((0, 1, 2, 5))  # hop 4 -> 5
    _check_ref_forward(model, s0, s1, s2)


def test_detbackflow_ref_forward_spinful():
    _spinful_chain(4, (2, 2))
    model = DetBackflow(_make_net(2), d=2, dtype=jnp.float32)  # rank 1 < 4
    assert model.use_ref
    s0 = _config((0, 1), (0, 1))
    s1 = _config((0, 2), (0, 1))  # up hop 1 -> 2
    s2 = _config((0, 2), (0, 2))  # dn hop 1 -> 2
    _check_ref_forward(model, s0, s1, s2)


def test_pfbackflow_ref_forward_spinless():
    _spinless_chain(6, 4)
    model = PfBackflow(_make_net(1), d=1, dtype=jnp.float32)  # rank 2 < 4
    assert model.use_ref
    s0 = _config((0, 1, 2, 3))
    s1 = _config((0, 1, 2, 4))
    s2 = _config((0, 1, 2, 5))
    _check_ref_forward(model, s0, s1, s2)


def test_pfbackflow_ref_forward_spinful():
    # The regression case: a singular default reference made the chained update wrong.
    _spinful_chain(4, (2, 2))
    model = PfBackflow(_make_net(2), d=2, dtype=jnp.float32)  # rank 2 < 4
    assert model.use_ref
    s0 = _config((0, 1), (0, 1))
    s1 = _config((0, 2), (0, 1))
    s2 = _config((0, 2), (0, 2))
    _check_ref_forward(model, s0, s1, s2)


# --------------------------------------------------------------------------- #
# use_ref=False fallback: ref_forward defers to __call__ and stores no internal
# --------------------------------------------------------------------------- #
def test_use_ref_false_fallback():
    _spinless_chain(6, 4)
    det = DetBackflow(_make_net(6), d=6, dtype=jnp.float32)  # rank 6 >= 4
    pf = PfBackflow(_make_net(4), d=4, dtype=jnp.float32)  # rank 8 >= 4
    assert not det.use_ref and not pf.use_ref
    s0 = _config((0, 1, 2, 3))
    s1 = _config((0, 1, 2, 4))
    for model in (det, pf):
        psi, internal = model.init_internal(s0)
        assert internal is None
        assert _rel_err(psi, model(s0)) < 1e-4
        psi1 = model.ref_forward(s1, s0, {"nflips": 2}, internal, False)
        assert _rel_err(psi1, model(s1)) < 1e-4


# --------------------------------------------------------------------------- #
# net outputs that aren't plain jax.Arrays (LogArray / ScaleArray) are accepted
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("wrap", [_LogArrayNet, _ScaleArrayNet], ids=["log", "scale"])
def test_detbackflow_accepts_wrapped_net_output(wrap):
    _spinless_chain(6, 4)
    base = DetBackflow(_make_net(2), d=2, dtype=jnp.float32)
    # Same parameters, only the net output type differs: the wavefunction must match,
    # and the accelerated path (which also calls the net) must still work.
    wrapped = eqx.tree_at(lambda m: m.net, base, wrap(base.net))
    s0 = _config((0, 1, 2, 3))
    s1 = _config((0, 1, 2, 4))
    assert _rel_err(wrapped(s0), base(s0)) < 1e-4
    _, internal = wrapped.init_internal(s0)
    psi1 = wrapped.ref_forward(s1, s0, {"nflips": 2}, internal, False)
    assert _rel_err(psi1, wrapped(s1)) < 1e-4


@pytest.mark.parametrize("wrap", [_LogArrayNet, _ScaleArrayNet], ids=["log", "scale"])
def test_pfbackflow_accepts_wrapped_net_output(wrap):
    _spinful_chain(4, (2, 2))
    base = PfBackflow(_make_net(2), d=2, dtype=jnp.float32)
    wrapped = eqx.tree_at(lambda m: m.net, base, wrap(base.net))
    s0 = _config((0, 1), (0, 1))
    s1 = _config((0, 2), (0, 1))
    assert _rel_err(wrapped(s0), base(s0)) < 1e-4
    _, internal = wrapped.init_internal(s0)
    psi1, _ = wrapped.ref_forward(s1, s0, {"nflips": 2}, internal, True)
    assert _rel_err(psi1, wrapped(s1)) < 1e-4
