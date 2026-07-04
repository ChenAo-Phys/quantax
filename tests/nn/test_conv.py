import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import quantax as qtx
from quantax.sites import Square, Chain, Lattice, TriangularB
from quantax.nn import ReshapeConv, ConvSymmetrize, Conv
from quantax.symmetry import TransND, Identity
from quantax.global_defs import PARTICLE_TYPE, get_lattice

# ---------- ReshapeConv ----------


def test_reshape_spin_matches_lattice_shape():
    # A flat +-1 configuration is reshaped to the lattice shape (1, 2, 2).
    Square(2)
    lattice = get_lattice()
    s = jnp.array([1, -1, -1, 1], dtype=jnp.float32)
    out = ReshapeConv()(s)
    assert out.shape == lattice.shape
    np.testing.assert_array_equal(np.asarray(out), np.asarray(s.reshape(lattice.shape)))


def test_reshape_default_dtype_is_float32():
    # The layer casts the input to float32 by default.
    Square(2)
    s = jnp.array([1, -1, -1, 1], dtype=jnp.int32)
    assert ReshapeConv()(s).dtype == jnp.float32


def test_reshape_custom_dtype_propagates():
    Square(2)
    s = jnp.array([1, -1, -1, 1], dtype=jnp.float32)
    assert ReshapeConv(jnp.float16)(s).dtype == jnp.float16


def test_reshape_multisite_cell():
    # 2 sites per cell, 3 cells -> lattice shape (2, 3); layout is [sub0, sub1].
    Lattice(extent=(3,), basis_vectors=[[1.0]], site_offsets=[[0.0], [0.3]])
    sub0 = [1, -1, 1]
    sub1 = [-1, -1, 1]
    out = ReshapeConv()(jnp.array(sub0 + sub1, dtype=jnp.float32))
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(np.asarray(out[0]), sub0)
    np.testing.assert_array_equal(np.asarray(out[1]), sub1)


def test_reshape_spinful_fermion_doubles_first_axis():
    # Spinful fermions store (up sites, dn sites); the first axis size doubles
    # and the up / down blocks fill the two halves.
    Square(2, particle_type=PARTICLE_TYPE.spinful_fermion)
    up = jnp.array([1, -1, 1, -1], dtype=jnp.float32)
    dn = jnp.array([1, 1, -1, -1], dtype=jnp.float32)
    out = ReshapeConv()(jnp.concatenate([up, dn]))
    assert out.shape == (2, 2, 2)  # (s_per_cell * 2, *spatial)
    np.testing.assert_array_equal(np.asarray(out[0]), np.asarray(up.reshape(2, 2)))
    np.testing.assert_array_equal(np.asarray(out[1]), np.asarray(dn.reshape(2, 2)))


# ---------- ConvSymmetrize ----------


def test_default_symmetry_is_full_translation():
    # Without arguments the layer uses the full translation group.
    Square(2)
    layer = ConvSymmetrize()
    assert layer.symm is not Identity()
    assert layer.symm.nsymm == get_lattice().ncells


def test_identity_returns_input_unchanged():
    # With the Identity symmetry the layer is a passthrough.
    Square(2)
    x = jnp.arange(12, dtype=jnp.float32)
    s = jnp.array([1, -1, -1, 1], dtype=jnp.float32)
    out = ConvSymmetrize(Identity())(x, s)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(x))


def test_trivial_translation_averages_all_copies():
    # For the trivial sector every character is 1, so symmetrization reduces to
    # the mean over all group copies (independent of the spin configuration s).
    Square(2)
    layer = ConvSymmetrize(TransND())
    s = jnp.array([1, -1, -1, 1], dtype=jnp.float32)
    x = jnp.arange(12, dtype=jnp.float32)
    out = layer(x, s)
    np.testing.assert_allclose(np.asarray(out), float(x.mean()), rtol=1e-6)


def test_momentum_sector_applies_character_phase():
    # In a non-trivial momentum sector, cyclically shifting the group copies
    # multiplies the symmetrized output by the generator's character, confirming
    # the copies are weighted by the characters in order.
    qtx.set_default_dtype(jnp.complex64)
    Chain(4, boundary=1)
    symm = TransND(sector=1)
    layer = ConvSymmetrize(symm)
    s = jnp.array([1, -1, 1, -1], dtype=jnp.float32)
    v = jnp.array([1 + 0j, 0.3 + 0.2j, -0.5 + 1j, 0.1 - 0.4j], dtype=jnp.complex64)
    out = layer(v, s)
    out_rolled = layer(jnp.roll(v, 1), s)
    np.testing.assert_allclose(
        np.asarray(out_rolled), np.asarray(symm.character[1] * out), atol=1e-5
    )


# ---------- Conv ----------


def test_conv_matches_eqx_circular_on_square():
    # On a periodic square lattice, Conv reduces to eqx.nn.Conv with "SAME"
    # circular padding: same weights give the same output.
    Square(4)
    conv = Conv(2, 3, 3, key=qtx.get_subkeys())
    econv = eqx.nn.Conv(
        num_spatial_dims=2,
        in_channels=2,
        out_channels=3,
        kernel_size=3,
        padding="SAME",
        padding_mode="CIRCULAR",
        key=qtx.get_subkeys(),
    )
    econv = eqx.tree_at(lambda m: (m.weight, m.bias), econv, (conv.weight, conv.bias))

    x = jax.random.normal(qtx.get_subkeys(), (2, 4, 4))
    out = conv(x)
    assert out.shape == (3, 4, 4)
    np.testing.assert_allclose(np.asarray(out), np.asarray(econv(x)), atol=1e-5)


def test_conv_translation_covariance_triangularb():
    # On the skewed TriangularB lattice, the pipeline
    # to_neighbor_repr -> Conv (twist-aware circular padding) -> to_original_repr
    # must commute with every lattice translation: translating the input
    # configuration permutes the output feature map by the same translation.
    lattice = TriangularB(2, boundary=1)
    trans = TransND()
    channels = 3
    conv = Conv(lattice.shape[0], channels, 3, key=qtx.get_subkeys())

    s = qtx.utils.rand_states()
    x = trans.get_symm_spins(s)  # all translated configurations
    x = x.reshape(trans.nsymm, *lattice.shape).astype(jnp.float32)
    x = jax.vmap(lattice.to_neighbor_repr)(x)
    x = jax.vmap(conv)(x)
    x = jax.vmap(lattice.to_original_repr)(x)

    out = np.asarray(x.reshape(trans.nsymm, channels, -1))
    perm = np.asarray(trans._perm)
    expected = np.transpose(out[0][:, perm], (1, 0, 2))
    np.testing.assert_allclose(out, expected, atol=1e-5)
