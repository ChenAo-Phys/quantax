import numpy as np
import jax.numpy as jnp
import pytest
from equinox.nn import Linear, Conv
from quantax.sites import Square, Grid
from quantax.nn import ReshapeConv, prod_by_log
from quantax.model import SingleDense, RBM_Dense, SingleConv, RBM_Conv
from quantax.model.shallow_nets import _get_scale
from quantax.global_defs import PARTICLE_TYPE, get_sites, get_lattice

# ---------- _get_scale ----------


def test_get_scale_returns_scalar_in_unit_interval():
    # The calibration searches the grid arange(0, 1, 0.01), so the chosen scale
    # is a finite scalar in [0, 1).
    Square(2)
    scale = np.asarray(_get_scale(jnp.cosh, features=8))
    assert scale.shape == ()
    assert np.isfinite(scale)
    assert 0.0 <= float(scale) < 1.0


def test_get_scale_dtype_matches_request():
    Square(2)
    assert jnp.asarray(_get_scale(jnp.cosh, 8, jnp.complex64)).dtype == jnp.complex64


# ---------- SingleDense ----------


def test_single_dense_layer_structure():
    # psi(s) = prod f(W s + b): a Linear, the activation, then prod_by_log.
    Square(2)
    net = SingleDense(4, jnp.cosh)
    assert len(net) == 3
    assert isinstance(net.layers[0], Linear)
    assert net.layers[1] is jnp.cosh
    assert net.layers[2] is prod_by_log


def test_single_dense_in_features_is_nmodes():
    # The dense layer reads the whole configuration, so fan-in equals Nmodes.
    Square(2)  # spin: Nmodes = Nsites = 4
    net = SingleDense(5, jnp.cosh)
    assert net.layers[0].in_features == get_sites().Nmodes
    assert net.layers[0].out_features == 5


def test_single_dense_forward_matches_product_formula():
    # The forward pass equals the literal product over hidden units.
    Square(2)
    net = SingleDense(4, jnp.cosh)
    s = jnp.array([1, -1, -1, 1])
    pre = net.layers[0](s)  # W s + b
    expected = jnp.prod(jnp.cosh(pre))
    np.testing.assert_allclose(np.asarray(net(s)), np.asarray(expected), rtol=1e-5)


def test_single_dense_use_bias_false_drops_bias():
    Square(2)
    net = SingleDense(4, jnp.cosh, use_bias=False)
    assert net.layers[0].bias is None


def test_single_dense_dtype_propagates_to_weight():
    Square(2)
    net = SingleDense(4, jnp.cosh, dtype=jnp.complex64)
    assert net.layers[0].weight.dtype == jnp.complex64


def test_single_dense_in_features_doubles_for_spinful_fermion():
    # Spinful fermions store 2 * Nsites modes; the dense fan-in follows Nmodes.
    Square(2, particle_type=PARTICLE_TYPE.spinful_fermion)
    net = SingleDense(4, jnp.cosh)
    assert net.layers[0].in_features == 2 * get_lattice().Nsites


# ---------- RBM_Dense ----------


def test_rbm_dense_uses_cosh_and_matches_product():
    Square(2)
    net = RBM_Dense(4)
    assert net.layers[1] is jnp.cosh
    s = jnp.array([1, -1, 1, -1])
    pre = net.layers[0](s)
    expected = jnp.prod(jnp.cosh(pre))
    np.testing.assert_allclose(np.asarray(net(s)), np.asarray(expected), rtol=1e-5)


def test_rbm_dense_holomorphic_follows_dtype():
    # cosh is holomorphic, so the network is holomorphic iff the dtype is complex.
    Square(2)
    assert RBM_Dense(4, dtype=jnp.float32).holomorphic is False
    assert RBM_Dense(4, dtype=jnp.complex64).holomorphic is True


# ---------- SingleConv ----------


def test_single_conv_layer_structure():
    # psi(s) = prod f(Conv(s)): reshape to the grid, convolve, activate, prod_by_log.
    Square(2)
    net = SingleConv(3, jnp.cosh)
    assert len(net) == 4
    assert isinstance(net.layers[0], ReshapeConv)
    assert isinstance(net.layers[1], Conv)
    assert net.layers[2] is jnp.cosh
    assert net.layers[3] is prod_by_log


def test_single_conv_channels_match_lattice():
    # weight is (out_channels, in_channels, *kernel); in_channels = shape[0] for spin.
    Square(2)  # spin: shape = (1, 2, 2)
    conv = SingleConv(3, jnp.cosh).layers[1]
    assert conv.weight.shape[0] == 3
    assert conv.weight.shape[1] == get_lattice().shape[0]


def test_single_conv_forward_matches_product_formula():
    Square(2)
    net = SingleConv(3, jnp.cosh)
    s = jnp.array([1, -1, -1, 1])
    x = net.layers[0](s)  # ReshapeConv
    pre = net.layers[1](x)  # Conv
    expected = jnp.prod(jnp.cosh(pre))
    np.testing.assert_allclose(np.asarray(net(s)), np.asarray(expected), rtol=1e-4)


def test_single_conv_use_bias_false_drops_bias():
    Square(2)
    assert SingleConv(3, jnp.cosh, use_bias=False).layers[1].bias is None


def test_single_conv_pbc_uses_circular_padding():
    Square(2, boundary=1)
    assert SingleConv(3, jnp.cosh).layers[1].padding_mode == "CIRCULAR"


def test_single_conv_obc_uses_zeros_padding():
    Square(2, boundary=0)
    assert SingleConv(3, jnp.cosh).layers[1].padding_mode == "ZEROS"


def test_single_conv_mixed_boundary_raises():
    # Mixing periodic and open boundaries has no well-defined padding mode.
    Grid([2, 2], boundary=[1, 0])
    with pytest.raises(ValueError):
        SingleConv(3, jnp.cosh)


def test_single_conv_spinful_fermion_doubles_in_channels():
    # Regression: ReshapeConv doubles the channel axis for spinful fermions, so the
    # conv must accept 2 * shape[0] input channels (otherwise the shapes mismatch).
    Square(2, particle_type=PARTICLE_TYPE.spinful_fermion)
    conv = SingleConv(3, jnp.cosh).layers[1]
    assert conv.weight.shape[1] == 2 * get_lattice().shape[0]


def test_single_conv_spinful_fermion_forward_runs():
    Square(2, particle_type=PARTICLE_TYPE.spinful_fermion)
    net = SingleConv(3, jnp.cosh)
    s = jnp.array([1, -1, 1, -1, 1, 1, -1, -1])  # length Nmodes = 2 * Nsites = 8
    out = np.asarray(net(s))
    assert out.shape == ()
    assert np.isfinite(out)


# ---------- RBM_Conv ----------


def test_rbm_conv_uses_cosh_and_matches_product():
    Square(2)
    net = RBM_Conv(3)
    assert net.layers[2] is jnp.cosh
    s = jnp.array([1, -1, 1, -1])
    x = net.layers[0](s)
    pre = net.layers[1](x)
    expected = jnp.prod(jnp.cosh(pre))
    np.testing.assert_allclose(np.asarray(net(s)), np.asarray(expected), rtol=1e-4)


def test_rbm_conv_holomorphic_follows_dtype():
    Square(2)
    assert RBM_Conv(3, dtype=jnp.float32).holomorphic is False
    assert RBM_Conv(3, dtype=jnp.complex64).holomorphic is True
