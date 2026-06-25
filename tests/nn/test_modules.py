import numpy as np
import jax.numpy as jnp
import equinox as eqx
import pytest
from quantax.nn import Sequential, RawInputLayer, RefModel

# ---------- helper layers ----------


class AddConst(eqx.Module):
    c: float

    def __call__(self, x):
        return x + self.c


class MulConst(eqx.Module):
    c: float

    def __call__(self, x):
        return x * self.c


class ReturnRaw(RawInputLayer):
    # Ignores the running activation and echoes back the raw input ``s``,
    # so a test can tell whether ``s`` is the original input or the running ``x``.
    def __call__(self, x, s):
        return s


class AddRaw(RawInputLayer):
    def __call__(self, x, s):
        return x + s


X = jnp.asarray(np.array([1.0, 2.0, 3.0], dtype=np.float32))


# ---------- container protocol ----------


def test_len():
    seq = Sequential([AddConst(1.0), MulConst(2.0), AddConst(3.0)])
    assert len(seq) == 3


def test_iter_yields_layers_in_order():
    layers = [AddConst(1.0), MulConst(2.0)]
    seq = Sequential(layers)
    assert list(seq) == layers


def test_getitem_int_returns_layer():
    l0, l1 = AddConst(1.0), MulConst(2.0)
    seq = Sequential([l0, l1])
    assert seq[0] is l0
    assert seq[1] is l1


def test_getitem_slice_returns_sequential_preserving_holomorphic():
    seq = Sequential([AddConst(1.0), MulConst(2.0), AddConst(3.0)], holomorphic=True)
    sub = seq[1:]
    assert isinstance(sub, Sequential)
    assert len(sub) == 2
    assert sub.holomorphic is True
    assert list(sub) == list(seq)[1:]


def test_getitem_invalid_type_raises():
    seq = Sequential([AddConst(1.0)])
    with pytest.raises(TypeError):
        seq["a"]  # type: ignore[index]


def test_holomorphic_defaults_false():
    seq = Sequential([AddConst(1.0)])
    assert seq.holomorphic is False


# ---------- forward pass ----------


def test_applies_layers_in_order():
    # (x + 1) * 2 = 2x + 2, order matters: multiply-then-add would differ.
    seq = Sequential([AddConst(1.0), MulConst(2.0)])
    np.testing.assert_allclose(np.asarray(seq(X)), np.asarray((X + 1.0) * 2.0))


def test_empty_sequential_is_identity():
    seq = Sequential([])
    np.testing.assert_array_equal(np.asarray(seq(X)), np.asarray(X))


# ---------- RawInputLayer raw-input forwarding ----------


def test_rawinputlayer_receives_raw_input_by_default():
    # s defaults to the original input, NOT the running activation. AddConst(10)
    # changes x, but ReturnRaw echoes the untouched raw input.
    seq = Sequential([AddConst(10.0), ReturnRaw()])
    np.testing.assert_array_equal(np.asarray(seq(X)), np.asarray(X))


def test_rawinputlayer_uses_explicit_s():
    s = jnp.asarray(np.array([-1.0, -2.0, -3.0], dtype=np.float32))
    seq = Sequential([AddConst(10.0), ReturnRaw()])
    np.testing.assert_array_equal(np.asarray(seq(X, s=s)), np.asarray(s))


def test_rawinputlayer_combines_running_and_raw():
    # AddRaw gets the running x (= X + 5) and the raw s (= X by default).
    seq = Sequential([AddConst(5.0), AddRaw()])
    np.testing.assert_allclose(np.asarray(seq(X)), np.asarray((X + 5.0) + X))


def test_rawinputlayer_base_call_not_implemented():
    with pytest.raises(NotImplementedError):
        RawInputLayer()(X, X)


# ---------- RefModel defaults / abstract surface ----------


def test_refmodel_defaults():
    model = RefModel()
    assert model.use_ref is True
    assert model.required_update_modes == ()


def test_refmodel_abstract_methods_raise():
    model = RefModel()
    with pytest.raises(NotImplementedError):
        model(X)
    with pytest.raises(NotImplementedError):
        model.init_internal(X)
    with pytest.raises(NotImplementedError):
        model.ref_forward(X, X, {"nflips": 2}, None)
