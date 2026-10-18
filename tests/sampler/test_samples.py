import dataclasses
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
from quantax.sampler import Samples
from quantax.sampler.samples import SamplesTree
from quantax.utils import LogArray


def _samples(ns: int = 8, nmodes: int = 4) -> Samples:
    """A fully-populated `Samples` with distinct, identifiable array leaves."""
    spins = jnp.arange(ns * nmodes, dtype=jnp.float32).reshape(ns, nmodes)
    psi = jnp.arange(ns, dtype=jnp.float32)
    prob = jnp.arange(ns, dtype=jnp.float32) + 10.0
    state_internal = {"a": jnp.arange(ns, dtype=jnp.float32) + 100.0}
    reweight = jnp.ones(ns, dtype=jnp.float32)
    return Samples(spins, psi=psi, prob=prob, state_internal=state_internal, r=reweight)


# --- construction ---


def test_fields_after_spins_are_keyword_only():
    # Only `spins` is positional. The other fields must be passed by keyword, so
    # that a call site can't silently shift a value into a neighbouring field
    # (e.g. r into state_internal) when fields are added or reordered.
    with pytest.raises(TypeError):
        Samples(jnp.ones((4, 2)), jnp.arange(4.0))  # type: ignore[call-arg]


# --- __getitem__ ---


def test_getitem_preserves_none_fields():
    s = Samples(jnp.arange(8.0).reshape(4, 2))
    sub = s[:2]
    assert sub.psi is None
    assert sub.prob is None
    assert sub.state_internal is None
    assert sub.r is None
    assert sub.nsamples == 2


def test_getitem_indexes_every_field_along_the_sample_axis():
    # including the array leaves of the nested state_internal pytree
    s = _samples(ns=8, nmodes=4)
    sub = s[jnp.array([0, 2, 4])]
    assert sub.nsamples == 3
    np.testing.assert_array_equal(np.asarray(sub.spins), np.asarray(s.spins)[::2][:3])
    np.testing.assert_array_equal(np.asarray(sub.psi), [0.0, 2.0, 4.0])
    np.testing.assert_array_equal(np.asarray(sub.prob), [10.0, 12.0, 14.0])
    np.testing.assert_array_equal(np.asarray(sub.r), [1.0, 1.0, 1.0])
    internal = np.asarray(sub.state_internal["a"])
    np.testing.assert_array_equal(internal, [100.0, 102.0, 104.0])


def test_getitem_recurses_into_psiarray():
    # `psi` and `prob` may be PyTrees (LogArray); indexing must reach their inner
    # leaves rather than treating them as opaque leaves.
    psi = LogArray.from_value(jnp.array([1.0, -2.0, 3.0, -4.0]))
    s = Samples(jnp.ones((4, 2)), psi=psi, prob=abs(psi))
    sub = s[:2]
    assert isinstance(sub.psi, LogArray) and isinstance(sub.prob, LogArray)
    assert np.array_equal(np.asarray(sub.psi.value()), [1.0, -2.0])
    assert np.array_equal(np.asarray(sub.prob.value()), [1.0, 2.0])


def test_getitem_renormalizes_r():
    # r of a subset is normalized to mean 1 again, while the original r is kept.
    r = jnp.array([0.5, 1.0, 1.5, 1.0])
    s = Samples(jnp.ones((4, 2)), r=r)
    np.testing.assert_allclose(np.asarray(s[:2].r), np.array([0.5, 1.0]) / 0.75)
    np.testing.assert_allclose(np.asarray(s[jnp.array([0, 2])].r), [0.5, 1.5])
    np.testing.assert_array_equal(np.asarray(s.r), np.asarray(r))


def test_getitem_returns_independent_samples():
    # Filling a field of a sliced copy leaves the original samples untouched.
    s = Samples(jnp.arange(8.0).reshape(4, 2))
    sub = s[:2]
    sub.psi = jnp.ones(2)
    assert s.psi is None


# --- not a PyTree: Samples can't be passed through jitted functions ---


def test_rejected_by_jit():
    # An assignment inside a jitted function would be silently lost on a copy, so
    # the samples are rejected instead.
    s = _samples()
    with pytest.raises(TypeError):
        jax.jit(lambda samples: samples.spins)(s)
    # unhashable, so equinox doesn't silently treat it as a static argument
    with pytest.raises(ValueError):
        eqx.filter_jit(lambda samples: samples.spins)(s)


def test_block_until_ready():
    # `jax.block_until_ready` calls the method of the samples, which blocks on all
    # of its arrays instead of silently skipping a non-PyTree object.
    s = _samples()
    assert jax.block_until_ready(s) is s
    assert s.block_until_ready() is s


# --- SamplesTree: the PyTree used inside the jitted Metropolis sweep ---


def _tree(ns: int = 8, nmodes: int = 4) -> SamplesTree:
    spins = jnp.arange(ns * nmodes, dtype=jnp.float32).reshape(ns, nmodes)
    psi = jnp.arange(ns, dtype=jnp.float32)
    state_internal = {"a": jnp.arange(ns, dtype=jnp.float32) + 100.0}
    return SamplesTree(spins, psi, state_internal=state_internal)


def test_tree_jittable_as_argument_and_return():
    # The fields, including a state_internal pytree, go through jit as arguments
    # and return values, while a None state_internal flattens to no leaf and
    # comes back as None.
    @jax.jit
    def f(tree: SamplesTree) -> SamplesTree:
        return jax.tree.map(lambda x: x[:4], tree)

    out = f(_tree(ns=8))
    assert isinstance(out, SamplesTree)
    assert out.spins.shape == (4, 4)
    np.testing.assert_array_equal(np.asarray(out.psi), [0.0, 1.0, 2.0, 3.0])
    internal = np.asarray(out.state_internal["a"])
    np.testing.assert_array_equal(internal, [100.0, 101.0, 102.0, 103.0])

    tree = SamplesTree(jnp.ones((8, 2)), jnp.ones(8))
    assert len(jax.tree_util.tree_leaves(tree)) == 2
    out = f(tree)
    assert out.spins.shape == (4, 2) and out.state_internal is None


# --- update_r ---


def _reweighted() -> tuple[Samples, np.ndarray]:
    """Samples drawn from q = |psi|^1 together with the expected r = |psi|^2 / q."""
    rng = np.random.default_rng(0)
    psi = rng.normal(size=8) + 1j * rng.normal(size=8)
    prob = np.abs(psi)
    r = np.abs(psi) ** 2 / prob
    spins = jnp.ones((8, 2))
    s = Samples(
        spins, psi=jnp.asarray(psi, jnp.complex64), prob=jnp.asarray(prob, jnp.float32)
    )
    return s, r / r.mean()


def test_update_r_is_normalized_born_over_prob():
    # r_s = (p_s / q_s) / <p/q>_q with p_s = |psi(s)|^2 and q_s = prob_s.
    s, expected = _reweighted()
    assert s.update_r() is None  # r is updated in place
    r = s.r
    assert isinstance(r, jax.Array) and not jnp.iscomplexobj(r)
    np.testing.assert_allclose(np.asarray(r), expected, rtol=1e-5)
    assert np.isclose(float(r.mean()), 1.0)


def test_update_r_follows_updated_psi():
    # r is recomputed from the current psi, e.g. after psi is filled in place.
    s, _ = _reweighted()
    s.update_r()
    s.psi = jnp.ones(8, jnp.complex64)
    s.update_r()
    expected = 1 / np.asarray(s.prob, np.float64)
    np.testing.assert_allclose(np.asarray(s.r), expected / expected.mean(), rtol=1e-5)


def test_update_r_keeps_r_without_prob():
    # Without prob there is nothing to update, so a given r (or None) is kept.
    s, _ = _reweighted()
    stored = jnp.full(8, 0.5)
    s = dataclasses.replace(s, prob=None, r=stored)
    s.update_r()
    assert s.r is stored
    s = Samples(jnp.ones((4, 2)), psi=jnp.ones(4))
    s.update_r()
    assert s.r is None


def test_update_r_requires_psi_with_prob():
    # A missing psi must not silently stand for the trivial factor r = 1.
    s, _ = _reweighted()
    s = dataclasses.replace(s, psi=None)
    with pytest.raises(ValueError, match="psi"):
        s.update_r()
    assert s.r is None


def test_update_r_zeroes_nonfinite_weights():
    # A single NaN amplitude must not contaminate the normalization of the rest.
    s, _ = _reweighted()
    psi = np.asarray(s.psi).astype(np.complex128)
    bad = dataclasses.replace(s, psi=jnp.asarray(s.psi).at[3].set(jnp.nan))
    bad.update_r()
    r = np.asarray(bad.r)
    expected = np.abs(psi) ** 2 / np.asarray(s.prob, np.float64)
    expected[3] = 0.0
    expected /= expected.mean()
    assert r[3] == 0.0 and np.all(np.isfinite(r))
    np.testing.assert_allclose(r, expected, rtol=1e-5)


def test_update_r_accepts_logarray_beyond_float32_range():
    # |psi| ~ e^200 overflows float32 when materialized, but the ratio stays finite
    # when the computation is carried out in the LogArray representation.
    logabs = jnp.asarray(200.0 + np.linspace(-1.0, 1.0, 8), jnp.float32)
    psi = LogArray(jnp.ones(8), logabs)
    s = Samples(jnp.ones((8, 2)), psi=psi, prob=psi**1.0)
    s.update_r()
    r = np.asarray(s.r)
    expected = np.exp(np.asarray(logabs, np.float64))
    expected /= expected.mean()
    assert np.all(np.isfinite(r))
    # float32 resolution of logabs ~ 200 is ~1.5e-5, which propagates into r.
    np.testing.assert_allclose(r, expected, rtol=1e-4)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
def test_update_r_promotes_low_precision_psi(dtype):
    # log|psi| of a half-precision psi is computed in float32, so r = 1 for prob from
    # the same psi. The rounding of log|psi| in half precision would distort r by up
    # to ~1%.
    psi = jnp.asarray(np.linspace(100.0, 300.0, 8), dtype)
    prob = abs(LogArray.from_value(psi.astype(jnp.float32))) ** 2.0
    s = Samples(jnp.ones((8, 2)), psi=psi, prob=prob)
    s.update_r()
    np.testing.assert_allclose(np.asarray(s.r), 1.0, rtol=1e-6)


@pytest.mark.parametrize("scale", [1e25, 1e-25])
def test_update_r_plain_psi_beyond_float32_range(scale):
    # |psi|^2 of a plain float32 psi ~ 1e25 or 1e-25 overflows or underflows
    # float32, but r is computed in the LogArray representation and stays accurate,
    # for prob given as a LogArray or as a plain array within the float32 range.
    x = np.linspace(1.0, 2.0, 8)
    psi = jnp.asarray(scale * x, jnp.float32)
    abs_psi = abs(LogArray.from_value(psi))
    for n, prob in [(2.0, abs_psi**2.0), (1.0, abs_psi**1.0), (1.0, jnp.abs(psi))]:
        s = Samples(jnp.ones((8, 2)), psi=psi, prob=prob)
        s.update_r()
        expected = x ** (2 - n)
        # float32 resolution of log|psi| ~ 57 is ~4e-6, which propagates into r
        np.testing.assert_allclose(
            np.asarray(s.r), expected / expected.mean(), rtol=1e-4
        )
