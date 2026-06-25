import numpy as np
import jax
import jax.numpy as jnp
import pytest
from quantax.sites import Triangular, TriangularB, Square
from quantax.nn import Sequential, Embedding, ReshapeTo_TriangularB, Reshape_TriangularB
from quantax.symmetry import TransND
from quantax.model import Triangular_ResConv, Triangular_Neighbor_Conv
from quantax.model.triangular_nets import _ResBlock
from quantax.utils import rand_states
from quantax.global_defs import get_lattice


def _conv_layers(net):
    """All Triangular_Neighbor_Conv layers inside the residual blocks of ``net``."""
    convs = []
    for layer in net.layers:
        if isinstance(layer, _ResBlock):
            convs += [layer.conv1, layer.conv2]
    return convs


# ---------- class structure ----------


def test_is_sequential_subclass():
    # The model is now a class (a Sequential subclass), not a factory function.
    Triangular(3)
    net = Triangular_ResConv(2, 4)
    assert isinstance(net, Sequential)
    assert type(net).__name__ == "Triangular_ResConv"


def test_config_stored_as_fields():
    Triangular(3)
    net = Triangular_ResConv(nblocks=2, channels=5, sublattice=(1, 1))
    assert net.nblocks == 2
    assert net.channels == 5
    assert net.sublattice == (1, 1)
    assert net.dtype == jnp.float32
    assert net.holomorphic is False


def test_non_triangular_lattice_raises():
    Square(2)
    with pytest.raises(ValueError):
        Triangular_ResConv(2, 4)


def test_complex_dtype_raises():
    Triangular(3)
    with pytest.raises(ValueError):
        Triangular_ResConv(2, 4, dtype=jnp.complex64)


# ---------- first layer is Embedding (the requested change) ----------


def test_triangular_first_layer_is_embedding():
    Triangular(3)
    net = Triangular_ResConv(2, 4)
    assert isinstance(net.layers[0], Embedding)


def test_triangularb_first_layer_is_embedding():
    TriangularB(2)
    net = Triangular_ResConv(2, 4)
    assert isinstance(net.layers[0], Embedding)


def test_embedding_table_sized_for_channels():
    # The embedding maps each per-cell state to a ``channels``-vector: Et is
    # (channels, 2 ** s_per_cell), with s_per_cell = shape[0] = 1 for spin.
    Triangular(3)
    net = Triangular_ResConv(2, 6)
    emb = net.layers[0]
    assert isinstance(emb, Embedding)
    assert emb.Et.shape == (6, 1 << get_lattice().shape[0])


def test_sublattice_creates_periodic_bias():
    Triangular(4)
    net = Triangular_ResConv(2, 4, sublattice=(2, 2))
    emb = net.layers[0]
    assert isinstance(emb, Embedding)
    assert emb.Ep is not None and emb.Ep.shape == (4, 2, 2)


# ---------- revised new_layer: every conv is channels -> channels ----------


def test_all_convs_are_channels_to_channels():
    # With the embedding in front, no conv reads the raw configuration anymore;
    # every convolution maps channels -> channels (previously the first conv had
    # in_channels = lattice.shape[0]).
    Triangular(3)
    channels = 7
    net = Triangular_ResConv(3, channels)
    convs = _conv_layers(net)
    assert len(convs) == 2 * 3
    for conv in convs:
        assert isinstance(conv, Triangular_Neighbor_Conv)
        assert conv.in_channels == channels
        assert conv.out_channels == channels


def test_first_block_conv1_has_bias_last_block_conv2_has_none():
    # conv1 always keeps its bias; only the very last conv (last block, conv2) drops it.
    Triangular(3)
    net = Triangular_ResConv(2, 4)
    blocks = [l for l in net.layers if isinstance(l, _ResBlock)]
    assert blocks[0].conv1.bias is not None
    assert blocks[-1].conv2.bias is None


# ---------- TriangularB geometry: the permutation matches Reshape_TriangularB ----------


def test_triangularb_inserts_permutation_layer():
    # For TriangularB an extra layer rearranges the embedding before the convs,
    # and the inverse rearrangement is applied at the end.
    TriangularB(2)
    net = Triangular_ResConv(2, 4)
    assert isinstance(net.layers[0], Embedding)
    assert callable(net.layers[1]) and not isinstance(net.layers[1], _ResBlock)
    assert any(isinstance(l, ReshapeTo_TriangularB) for l in net.layers)


def test_triangular_has_no_permutation_or_reshape_back():
    # The regular Triangular lattice needs neither the permute glue (the second
    # layer is already a residual block) nor the reshape-back at the end.
    Triangular(3)
    net = Triangular_ResConv(2, 4)
    assert isinstance(net.layers[1], _ResBlock)
    assert not any(isinstance(l, ReshapeTo_TriangularB) for l in net.layers)


def test_triangularb_permutation_matches_reshape_arrangement():
    # The glue layer after the embedding must reproduce exactly the Triangular
    # arrangement that Reshape_TriangularB produced from the raw configuration,
    # otherwise the convolutions would see the wrong neighbors.
    TriangularB(2)
    lattice = get_lattice()
    net = Triangular_ResConv(2, 4)
    permute = net.layers[1]

    probe = jnp.arange(lattice.Nsites, dtype=jnp.float32)
    # how the glue rearranges a single-channel natural-grid feature map
    got = np.asarray(permute(probe.reshape(1, *lattice.shape[1:])))
    # how Reshape_TriangularB rearranges the same raw indices
    ref = np.asarray(Reshape_TriangularB()(probe))
    np.testing.assert_array_equal(got, ref)


# ---------- forward pass ----------


@pytest.mark.parametrize(
    "make_lattice", [lambda: Triangular(3), lambda: TriangularB(2)]
)
def test_forward_runs_and_is_finite(make_lattice):
    make_lattice()
    net = Triangular_ResConv(2, 4)
    s = rand_states()
    out = np.asarray(net(s))
    assert out.shape == ()
    assert np.isfinite(out)


@pytest.mark.parametrize(
    "make_lattice", [lambda: Triangular(3), lambda: TriangularB(2)]
)
def test_forward_batches_under_vmap(make_lattice):
    make_lattice()
    net = Triangular_ResConv(2, 4)
    s = rand_states(5)
    out = np.asarray(jax.vmap(net)(s))
    assert out.shape == (5,)
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize(
    "make_lattice", [lambda: Triangular(3), lambda: TriangularB(2)]
)
def test_translation_invariant_in_trivial_sector(make_lattice):
    # The default ConvSymmetrize projects onto the trivial translation sector, so
    # the amplitude is invariant under every lattice translation.
    make_lattice()
    net = Triangular_ResConv(2, 4)
    s = rand_states()
    psi0 = np.asarray(net(s))
    perms = np.asarray(TransND()._perm)
    for perm in perms:
        np.testing.assert_allclose(np.asarray(net(s[perm])), psi0, rtol=1e-5, atol=1e-6)


# ---------- dtype threading ----------


def test_default_dtype_is_float32():
    Triangular(3)
    net = Triangular_ResConv(2, 4)
    emb = net.layers[0]
    assert isinstance(emb, Embedding)
    assert emb.Et.dtype == jnp.float32
    for conv in _conv_layers(net):
        assert conv.weight.dtype == jnp.float32


def test_dtype_threads_into_embedding_and_convs(x64):
    Triangular(3)
    net = Triangular_ResConv(2, 4, dtype=jnp.float64)
    emb = net.layers[0]
    assert isinstance(emb, Embedding)
    assert emb.Et.dtype == jnp.float64
    for conv in _conv_layers(net):
        assert conv.weight.dtype == jnp.float64
