import dataclasses
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from quantax.sampler import Samples
from quantax.utils import LogArray


def _samples(ns: int = 8, nmodes: int = 4) -> Samples:
    """A fully-populated `Samples` with distinct, identifiable array leaves."""
    spins = jnp.arange(ns * nmodes, dtype=jnp.float32).reshape(ns, nmodes)
    psi = jnp.arange(ns, dtype=jnp.float32)
    state_internal = {"a": jnp.arange(ns, dtype=jnp.float32) + 100.0}
    reweight = jnp.ones(ns, dtype=jnp.float32)
    return Samples(spins, psi, state_internal, reweight)


# --- construction ---


def test_defaults_are_none():
    # spins is the only required field; the rest default to None.
    s = Samples(jnp.ones((4, 2)))
    assert s.psi is None
    assert s.state_internal is None
    assert s.reweight_factor is None


def test_is_frozen():
    s = Samples(jnp.ones((4, 2)))
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.spins = jnp.zeros((4, 2))


def test_positional_field_order_matches_constructor():
    # Guards the documented order (spins, psi, state_internal, reweight_factor):
    # callers across the package build `Samples` positionally as
    # `Samples(spins, psi, None, reweight)`.
    internal = {"k": jnp.zeros(4)}
    reweight = jnp.full(4, 0.5)
    s = Samples(jnp.ones((4, 2)), jnp.arange(4.0), internal, reweight)
    assert s.state_internal is internal
    assert np.allclose(np.asarray(s.reweight_factor), 0.5)


# --- nsamples ---


def test_nsamples_is_leading_axis():
    assert _samples(ns=8).nsamples == 8
    assert Samples(jnp.ones((3, 7))).nsamples == 3


# --- __getitem__ ---


def test_getitem_returns_samples():
    assert isinstance(_samples()[:3], Samples)


def test_getitem_slices_every_array_leaf():
    sub = _samples(ns=8)[:3]
    assert np.asarray(sub.spins).shape == (3, 4)
    assert np.asarray(sub.psi).shape == (3,)
    assert np.asarray(sub.reweight_factor).shape == (3,)
    # state_internal is a nested pytree; its array leaves are sliced too.
    assert np.asarray(sub.state_internal["a"]).shape == (3,)


def test_getitem_preserves_none_fields():
    s = Samples(jnp.arange(8.0).reshape(4, 2))
    sub = s[:2]
    assert sub.psi is None
    assert sub.state_internal is None
    assert sub.reweight_factor is None
    assert np.asarray(sub.spins).shape == (2, 2)


def test_getitem_values_are_correct():
    s = _samples(ns=8)
    idx = jnp.array([0, 2, 4])
    sub = s[idx]
    assert np.array_equal(np.asarray(sub.psi), [0.0, 2.0, 4.0])
    assert np.array_equal(np.asarray(sub.state_internal["a"]), [100.0, 102.0, 104.0])


def test_getitem_recurses_into_psiarray():
    # `psi` may itself be a PyTree (LogArray / ScaleArray); indexing must reach
    # its inner leaves rather than treating it as an opaque leaf.
    psi = LogArray.from_value(jnp.array([1.0, -2.0, 3.0, -4.0]))
    s = Samples(jnp.ones((4, 2)), psi)
    sub = s[:2]
    assert isinstance(sub.psi, LogArray)
    assert np.array_equal(np.asarray(sub.psi.value()), [1.0, -2.0])


# --- pytree behaviour ---


def test_pytree_roundtrip_preserves_structure():
    s = _samples()
    leaves, treedef = jax.tree_util.tree_flatten(s)
    s2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(s2, Samples)
    assert np.array_equal(np.asarray(s2.spins), np.asarray(s.spins))
    assert np.array_equal(np.asarray(s2.psi), np.asarray(s.psi))
    assert np.array_equal(
        np.asarray(s2.state_internal["a"]), np.asarray(s.state_internal["a"])
    )
    assert np.array_equal(np.asarray(s2.reweight_factor), np.asarray(s.reweight_factor))


def test_pytree_roundtrip_with_none_children():
    # None children flatten to zero leaves and must come back as None.
    s = Samples(jnp.ones((4, 2)))
    leaves, treedef = jax.tree_util.tree_flatten(s)
    assert len(leaves) == 1  # only `spins`
    s2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert s2.psi is None and s2.state_internal is None and s2.reweight_factor is None


def test_tree_map_applies_to_all_array_leaves():
    s = _samples(ns=4)
    doubled = jax.tree.map(lambda x: x * 2, s)
    assert np.array_equal(np.asarray(doubled.psi), np.asarray(s.psi) * 2)
    assert np.array_equal(
        np.asarray(doubled.state_internal["a"]), np.asarray(s.state_internal["a"]) * 2
    )


def test_jittable_as_argument_and_return():
    s = _samples(ns=8)

    @jax.jit
    def f(samples: Samples) -> Samples:
        # Consume it as an argument and produce a new Samples as output.
        return samples[:4]

    out = f(s)
    assert isinstance(out, Samples)
    assert out.nsamples == 4
    assert np.array_equal(np.asarray(out.psi), [0.0, 1.0, 2.0, 3.0])
