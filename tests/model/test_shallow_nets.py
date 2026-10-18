from itertools import product

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
from quantax.sites import Square, Grid
from quantax.model import SingleDense, RBM_Dense, SingleConv, RBM_Conv
from quantax.model.shallow_nets import _get_scale
from quantax.global_defs import PARTICLE_TYPE, get_sites

# Forward passes are jitted and vmapped over configurations: eager evaluation compiles
# every primitive separately, which costs several times more. Every check jits a fresh
# closure, as the networks read the global lattice while tracing.


def _values(net, configs):
    return np.asarray(eqx.filter_jit(lambda m, x: jax.vmap(m)(x).value())(net, configs))


def _all_configs():
    return jnp.array(list(product([1, -1], repeat=get_sites().Nmodes)), jnp.int8)


# ---------- _get_scale ----------


def test_get_scale_calibrates_log_amplitude_spread():
    # The init scale targets std(log|psi|) = std0 * sqrt(Nsites) with std0 = 0.1 for
    # unit Gaussian pre-activations. For fn = exp, log|fn(scale x)| = scale x, so the
    # sum over 4 features has std 2 * scale and the target 0.2 on 4 sites is hit at
    # scale = 0.1, up to the sampling error of the 1000 draws and the 0.01 grid.
    Square(2)
    assert abs(float(_get_scale(jnp.exp, features=4)) - 0.1) <= 0.01


# ---------- SingleDense / RBM_Dense ----------


@pytest.mark.parametrize(
    "actfn, use_bias, dtype, particle_type",
    [
        (None, True, jnp.float32, PARTICLE_TYPE.spin),  # RBM_Dense
        (None, True, jnp.complex64, PARTICLE_TYPE.spin),  # complex RBM_Dense
        # signed activation, no bias; spinful fermions have Nmodes = 2 * Nsites inputs
        (jnp.tanh, False, jnp.float32, PARTICLE_TYPE.spinful_fermion),
    ],
    ids=["rbm", "rbm-complex", "tanh-nobias-spinful"],
)
def test_single_dense_forward_matches_formula(actfn, use_bias, dtype, particle_type):
    # psi(s) = prod_j f(W s + b)_j on every configuration, signs included. RBM_Dense
    # uses f = cosh, which is holomorphic, so it is holomorphic iff dtype is complex.
    # features = 3 differs from Nmodes, so the layer orientation is pinned too.
    Square(2, particle_type=particle_type)
    if actfn is None:
        net = RBM_Dense(3, use_bias, dtype)
        assert net.holomorphic == jnp.issubdtype(dtype, jnp.complexfloating)
        fn = np.cosh
    else:
        net = SingleDense(3, actfn, use_bias, dtype=dtype)
        fn = np.tanh
    linear = net.layers[0]
    assert linear.weight.shape == (3, get_sites().Nmodes)
    assert linear.weight.dtype == dtype
    assert (linear.bias is not None) == use_bias
    W = np.asarray(linear.weight)
    b = np.asarray(linear.bias) if use_bias else 0.0
    configs = _all_configs()
    expected = np.prod(fn(np.asarray(configs) @ W.T + b), axis=1)
    np.testing.assert_allclose(_values(net, configs), expected, rtol=1e-5)


# ---------- SingleConv / RBM_Conv ----------


def _conv_reference(fn, w, b, s):
    """numpy prod_{c,x,y} f(sum_{c',i,j} w[c,c',i,j] grid[c',x+i,y+j] + b[c]) with
    periodic indices, on the (channels, Lx, Ly) grid that ReshapeConv feeds to the
    conv, the spin-up and spin-down blocks being the input channels. The product runs
    over all positions, so it doesn't depend on the padding offset of the kernel."""
    Lx, Ly = get_sites().shape[1:]
    grid = s.reshape(s.shape[0], w.shape[1], Lx, Ly)
    pre = np.zeros((s.shape[0], w.shape[0], Lx, Ly), w.dtype) + b[:, None, None]
    for i in range(w.shape[2]):
        for j in range(w.shape[3]):
            shifted = np.roll(grid, (-i, -j), axis=(2, 3))  # grid[c', x+i, y+j]
            pre += np.einsum("dc,bcxy->bdxy", w[:, :, i, j], shifted)
    return np.prod(fn(pre), axis=(1, 2, 3))


@pytest.mark.parametrize(
    "actfn, use_bias, dtype, particle_type",
    [
        (None, True, jnp.float32, PARTICLE_TYPE.spin),  # RBM_Conv
        (None, True, jnp.complex64, PARTICLE_TYPE.spin),  # complex RBM_Conv
        (None, True, jnp.float32, PARTICLE_TYPE.spinful_fermion),
        (jnp.tanh, False, jnp.float32, PARTICLE_TYPE.spin),  # signed, no bias
    ],
    ids=["rbm", "rbm-complex", "rbm-spinful", "tanh-nobias"],
)
def test_single_conv_forward_matches_formula(actfn, use_bias, dtype, particle_type):
    # psi(s) = prod f(Conv(s)) on every configuration, with a full-lattice kernel and
    # circular padding under periodic boundaries; the formula is then invariant under
    # all lattice translations. For spinful fermions ReshapeConv stacks the up and
    # down occupations as 2 * shape[0] input channels (regression: the conv must
    # accept them). channels = 3 rather than 1 pins the number of output channels.
    Square(2, particle_type=particle_type)
    if actfn is None:
        net = RBM_Conv(3, use_bias, dtype)
        assert net.holomorphic == jnp.issubdtype(dtype, jnp.complexfloating)
        fn = np.cosh
    else:
        net = SingleConv(3, actfn, use_bias, dtype=dtype)
        fn = np.tanh
    conv = net.layers[1]
    lattice_shape = get_sites().shape
    spinful = particle_type == PARTICLE_TYPE.spinful_fermion
    in_channels = lattice_shape[0] * (2 if spinful else 1)
    assert conv.weight.shape == (3, in_channels, *lattice_shape[1:])
    assert conv.weight.dtype == dtype
    assert (conv.bias is not None) == use_bias
    w = np.asarray(conv.weight)
    b = np.asarray(conv.bias).reshape(-1) if use_bias else np.zeros(3, w.dtype)
    configs = _all_configs()
    expected = _conv_reference(fn, w, b, np.asarray(configs))
    psi = _values(net, configs)
    assert np.iscomplexobj(psi) == jnp.issubdtype(dtype, jnp.complexfloating)
    np.testing.assert_allclose(psi, expected, rtol=1e-5)


def test_single_conv_padding_follows_boundary():
    Square(2, boundary=0)
    assert SingleConv(1, jnp.cosh).layers[1].padding_mode == "ZEROS"


def test_single_conv_mixed_boundary_raises():
    # Mixing periodic and open boundaries has no well-defined padding mode.
    Grid([2, 2], boundary=[1, 0])
    with pytest.raises(ValueError):
        SingleConv(1, jnp.cosh)
