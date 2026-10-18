import itertools
import numpy as np
import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import quantax as qtx
from quantax.sites import Square, Grid, Lattice, TriangularB
from quantax.nn import ReshapeConv, ConvSymmetrize, Conv, circular_pad
from quantax.symmetry import TransND, Translation, Identity, Z2Inversion
from quantax.global_defs import PARTICLE_TYPE

# Layers are evaluated under jit; eager evaluation compiles every primitive
# separately and dominates the test time.

# ---------- ReshapeConv ----------


def test_reshape_multisite_cell_casts_to_float32():
    # 2 sites per cell, 3 cells -> lattice shape (2, 3); layout is [sub0, sub1].
    # Integer inputs are cast to float32 by default.
    Lattice(extent=(3,), basis_vectors=[[1.0]], site_offsets=[[0.0], [0.3]])
    sub0 = [1, -1, 1]
    sub1 = [-1, -1, 1]
    s = np.array(sub0 + sub1, dtype=np.int32)
    out = jax.jit(lambda s: ReshapeConv()(s))(s)
    assert out.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(out), [sub0, sub1])


def test_reshape_spinful_fermion_stacks_spin_blocks():
    # Spinful fermions store (up sites, dn sites); the first axis size doubles
    # and the up / down blocks fill the two halves.
    Square(2, particle_type=PARTICLE_TYPE.spinful_fermion)
    up = np.array([1, -1, 1, -1], dtype=np.float32)
    dn = np.array([1, 1, -1, -1], dtype=np.float32)
    out = jax.jit(lambda s: ReshapeConv(jnp.float16)(s))(np.concatenate([up, dn]))
    assert out.dtype == jnp.float16
    np.testing.assert_array_equal(np.asarray(out), [up.reshape(2, 2), dn.reshape(2, 2)])


# ---------- ConvSymmetrize ----------


@pytest.mark.parametrize(
    "boundary, vectors",
    [
        (1, [[1, 0], [0, 1]]),
        # anti-periodic axes are translated as well, not averaged like open ones
        # (ResConv uses this default, e.g. for Hubbard models with APBC)
        ((-1, 1), [[1, 0], [0, 1]]),
        ((1, 0), [[1, 0]]),
        (0, None),
    ],
    ids=["pbc", "apbc", "mixed", "obc"],
)
def test_conv_symmetrize_default_symmetry(boundary, vectors):
    # The default imposes unit translations along the periodic axes only, and full
    # OBC reduces to the Identity. In the trivial sector the output is the mean of
    # the feature map, whatever the ordering of the group elements.
    # Spins can't have anti-periodic boundaries, so that case uses spinless fermions.
    apbc = np.any(np.asarray(boundary) < 0)
    particle_type = PARTICLE_TYPE.spinless_fermion if apbc else PARTICLE_TYPE.spin
    Grid([4, 2], boundary=boundary, particle_type=particle_type)
    layer = ConvSymmetrize()
    if vectors is None:
        assert layer.symm is Identity()
    else:
        assert isinstance(layer.symm, Translation)
        np.testing.assert_array_equal(layer.symm.vectors, vectors)
    x = np.random.default_rng(0).normal(size=(2, 4, 2)).astype(np.float32)
    s = -np.ones(8, np.float32)  # no fermions, so no permutation signs
    out = jax.jit(lambda x, s: layer(x, s))(x, s)
    np.testing.assert_allclose(float(out), x.mean(), rtol=1e-5)


def _feature_map(s, a, e):
    """A 2-channel feature map f(s)[c, x, y] on the 4x2 grid, covariant under the
    translations along x, and along y as well if ``e == 0``: f(T_v s)[r] = f(s)[r + v]
    for (T_v s)(r) = s(r + v), the convention of `Symmetry.get_symm_spins`."""
    g = s.reshape(4, 2)
    terms = jnp.stack([g, g * jnp.roll(g, -1, axis=0), g * jnp.roll(g, -1, axis=1)])
    f = jnp.tanh(jnp.einsum("ck,kxy->cxy", a, terms) + e * g)
    return f.astype(qtx.get_default_dtype())


@pytest.mark.parametrize(
    "boundary, make_symm, cpl, region",
    [
        (1, lambda: TransND(sector=(1, 1)), True, np.s_[:1, :1]),
        # each group element covers 2 positions along x, averaged into one value
        (1, lambda: Translation([[2, 0], [0, 1]], sector=(1, 1)), False, np.s_[:2, :1]),
        # the open y axis is averaged over
        ((1, 0), lambda: Translation([1, 0], sector=1), True, np.s_[:1, :]),
    ],
    ids=["pbc_momentum", "pbc_step2", "open_y"],
)
def test_conv_symmetrize_matches_projection(boundary, make_symm, cpl, region):
    # A covariant feature map f symmetrized by ConvSymmetrize must equal the
    # projection sum_g chi_g psi0(T_g s) of psi0(s) = mean of f(s) over the
    # channels and the positions of the identity element. In non-trivial sectors
    # this pins the order of the values against the group elements.
    if cpl:
        qtx.set_default_dtype(jnp.complex64)
    Grid([4, 2], boundary=boundary)
    symm = make_symm()
    layer = ConvSymmetrize(symm)
    rng = np.random.default_rng(1)
    a = rng.normal(size=(2, 3)).astype(np.float32)
    # position-dependent along the open axis, which breaks the y translations
    e = 0.0 if boundary == 1 else rng.normal(size=(2, 1, 2)).astype(np.float32)
    f = lambda s: _feature_map(s, a, e)

    @jax.jit
    @jax.vmap
    def run(s):
        psi0 = jax.vmap(lambda t: f(t)[(slice(None), *region)].mean())
        return layer(f(s), s), symm.symmetrize(psi0(symm.get_symm_spins(s)), s)

    s = np.array(list(itertools.product([1.0, -1.0], repeat=8)), np.float32)
    out, expected = run(s)
    # the projection vanishes only on configurations with a matching symmetry
    assert np.mean(np.abs(np.asarray(expected)) > 1e-3) > 0.5
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), atol=1e-6)


def test_conv_symmetrize_identity_is_passthrough():
    Grid([4, 2])
    x = np.arange(16, dtype=np.float32).reshape(2, 4, 2)
    out = ConvSymmetrize(Identity())(x, np.ones(8, np.float32))
    np.testing.assert_array_equal(np.asarray(out), x)


def test_conv_symmetrize_rejects_unsupported_symmetry():
    Grid([4, 2])
    with pytest.raises(NotImplementedError, match="tilted"):
        ConvSymmetrize(Translation([1, 1]))
    with pytest.raises(ValueError, match="only supports"):
        ConvSymmetrize(Z2Inversion())


# ---------- circular_pad ----------


@pytest.mark.parametrize(
    "padding, np_pad",
    [
        (1, ((1, 1), (1, 1))),
        ((2, 0), ((2, 2), (0, 0))),
        # asymmetric, and wrapping more than once around the y axis of size 4
        (((0, 2), (5, 1)), ((0, 2), (5, 1))),
    ],
    ids=["int", "per_dim", "pairs"],
)
def test_circular_pad_matches_numpy_wrap(padding, np_pad):
    # On an unskewed torus the twist-aware wrap reduces to a plain periodic wrap.
    Grid([3, 4])
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    out = jax.jit(lambda x: circular_pad(x, padding))(x)
    expected = np.pad(x, ((0, 0), *np_pad), mode="wrap")
    np.testing.assert_array_equal(np.asarray(out), expected)


def test_circular_pad_rejects_mismatched_input():
    Grid([3, 4])
    with pytest.raises(ValueError, match="spatial shape"):
        circular_pad(np.zeros((2, 4, 3)), 1)
    with pytest.raises(ValueError, match="length"):
        circular_pad(np.zeros((2, 3, 4)), (1, 1, 1))


# ---------- Conv ----------

# All Conv layers share the weight shape (40, 30, 3, 2) of test_initializers.py, so
# the lecun_normal initialization compiles once. The (3, 2) kernel needs asymmetric
# "SAME" padding along y.
C_IN, C_OUT, KERNEL = 30, 40, (3, 2)


def _conv_reference(x, w, b, boundary, stride, dilation):
    """Direct sum of the "SAME" cross-correlation: periodic and anti-periodic axes
    wrap around without a sign, open axes are zero-padded."""
    spatial = x.shape[1:]
    sizes, lows = [], []
    for L, k, st, d in zip(spatial, w.shape[2:], stride, dilation):
        n = -(-L // st)
        sizes.append(n)
        lows.append(max((n - 1) * st + (k - 1) * d + 1 - L, 0) // 2)

    out = np.zeros((w.shape[0], *sizes))
    for pos in np.ndindex(*sizes):
        for koff in np.ndindex(*w.shape[2:]):
            src = []
            for ax, L in enumerate(spatial):
                i = pos[ax] * stride[ax] + koff[ax] * dilation[ax] - lows[ax]
                if boundary[ax] != 0:
                    i %= L
                elif not 0 <= i < L:
                    break
                src.append(i)
            else:
                out[(slice(None), *pos)] += w[(..., *koff)] @ x[(slice(None), *src)]
    if b is not None:
        out += b
    return out


@pytest.mark.parametrize(
    "boundary, stride, dilation, use_bias",
    [
        ((1, 1), (1, 1), (1, 1), True),
        # anti-periodic x wraps without a sign; strided x, dilated open y
        ((-1, 0), (2, 1), (1, 2), True),
        ((0, 0), (1, 1), (1, 1), False),
    ],
    ids=["pbc", "apbc_obc_strided_dilated", "obc_no_bias"],
)
def test_conv_matches_direct_sum(boundary, stride, dilation, use_bias):
    # spinless fermions, as spins can't have anti-periodic boundaries
    Square(4, boundary=boundary, particle_type=PARTICLE_TYPE.spinless_fermion)
    conv = Conv(C_IN, C_OUT, KERNEL, stride, dilation, use_bias, key=qtx.get_subkeys())
    assert conv.weight.shape == (C_OUT, C_IN, *KERNEL)
    rng = np.random.default_rng(0)
    if use_bias:
        assert np.all(np.asarray(conv.bias) == 0)  # zero-initialized
        bias = rng.normal(size=(C_OUT, 1, 1)).astype(np.float32)
        conv = eqx.tree_at(lambda m: m.bias, conv, bias)
    else:
        assert conv.bias is None
    x = rng.normal(size=(C_IN, 4, 4)).astype(np.float32)
    out = jax.jit(lambda x: conv(x))(x)
    w = np.asarray(conv.weight, np.float64)
    b = None if conv.bias is None else np.asarray(conv.bias, np.float64)
    expected = _conv_reference(x, w, b, boundary, stride, dilation)
    np.testing.assert_allclose(np.asarray(out), expected, atol=1e-5)


def test_conv_parameters_keep_dtype():
    # shapes and dtypes only, so the abstract layer avoids an initialization compile
    Square(4)
    make = lambda key: Conv(C_IN, C_OUT, KERNEL, dtype=jnp.float16, key=key)
    conv = eqx.filter_eval_shape(make, qtx.get_subkeys())
    assert conv.weight.dtype == jnp.float16 and conv.bias.dtype == jnp.float16


def test_conv_rejects_wrong_number_of_kernel_sizes():
    Square(4)
    with pytest.raises(ValueError, match="Expected 2 values"):
        Conv(C_IN, C_OUT, (3, 2, 1), key=qtx.get_subkeys())


def test_conv_translation_covariance_triangularb():
    # On the skewed TriangularB lattice, the pipeline
    # to_neighbor_repr -> Conv (twist-aware circular padding) -> to_original_repr
    # must commute with every lattice translation, which rolls the grid in the
    # original representation: (T_v s)(r) = s(r + v) gives f(T_v s)[r] = f(s)[r + v].
    lattice = TriangularB(2)
    Lx, Ly = lattice.shape[1:]
    # The twisted wrap is along y. Dilating the 2-tap y kernel gives the "SAME" pads
    # (1, 1) instead of (0, 1), so both sides of that wrap are used.
    conv = Conv(C_IN, C_OUT, KERNEL, dilation=(1, 2), key=qtx.get_subkeys())
    rng = np.random.default_rng(0)
    # a random pointwise embedding of the spins into C_IN channels is covariant
    alpha, beta = rng.normal(size=(2, C_IN, 1, 1)).astype(np.float32)

    @jax.jit
    @jax.vmap
    def f(s):
        x = lattice.to_neighbor_repr(alpha + beta * s)
        return lattice.to_original_repr(conv(x))

    s = rng.choice([-1.0, 1.0], size=(Lx, Ly)).astype(np.float32)
    shifts = list(np.ndindex(Lx, Ly))
    out = np.asarray(f(np.stack([np.roll(s, (-vx, -vy), (0, 1)) for vx, vy in shifts])))
    expected = [np.roll(out[0], (-vx, -vy), (1, 2)) for vx, vy in shifts]
    np.testing.assert_allclose(out, np.stack(expected), atol=1e-5)
