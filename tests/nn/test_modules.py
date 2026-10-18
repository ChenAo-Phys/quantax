import numpy as np
import equinox as eqx
import pytest
from quantax.nn import Sequential, RawInputLayer, RefModel

# Sequential only dispatches between layers, so NumPy inputs exercise it fully
# without any JAX compilation.

# ---------- helper layers ----------


class AddConst(eqx.Module):
    c: float

    def __call__(self, x):
        return x + self.c


class MulConst(eqx.Module):
    c: float

    def __call__(self, x):
        return x * self.c


class AddRaw(RawInputLayer):
    def __call__(self, x, s):
        return x + s


X = np.array([1.0, 2.0, 3.0], dtype=np.float32)
S = np.array([-1.0, -2.0, -4.0], dtype=np.float32)


# ---------- container protocol ----------


def test_container_protocol():
    layers = [AddConst(1.0), MulConst(2.0), AddConst(3.0)]
    # Variational reads this flag to choose the gradient path, so a holomorphic
    # default would give wrong gradients for non-holomorphic complex networks
    assert Sequential(layers).holomorphic is False
    seq = Sequential(layers, holomorphic=True)
    assert len(seq) == 3
    assert list(seq) == layers
    assert seq[1] is layers[1]
    # slicing returns a Sequential that keeps the holomorphic flag
    sub = seq[1:]
    assert isinstance(sub, Sequential)
    assert sub.holomorphic is True
    assert list(sub) == layers[1:]
    with pytest.raises(TypeError):
        seq["a"]  # type: ignore[index]


# ---------- forward pass ----------


def test_applies_layers_in_order():
    # (x + 1) * 2 = 2x + 2, order matters: multiply-then-add would differ.
    seq = Sequential([AddConst(1.0), MulConst(2.0)])
    np.testing.assert_allclose(seq(X), (X + 1.0) * 2.0)


def test_rawinputlayer_gets_running_x_and_raw_s():
    # A RawInputLayer gets the running activation x and the raw input s, which
    # defaults to the network input (not the running x) and can be overridden.
    seq = Sequential([AddConst(5.0), AddRaw(), MulConst(2.0), AddRaw()])
    np.testing.assert_allclose(seq(X), ((X + 5.0) + X) * 2.0 + X)
    np.testing.assert_allclose(seq(X, s=S), ((X + 5.0) + S) * 2.0 + S)


# ---------- RefModel defaults ----------


def test_refmodel_defaults():
    # Metropolis samplers read these defaults to decide whether to use local updates
    model = RefModel()
    assert model.use_ref is True
    assert model.required_update_modes == ()
