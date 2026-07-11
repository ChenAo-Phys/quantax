import numpy as np
import jax.numpy as jnp
import pytest

import quantax as qtx
from quantax.nn import Sequential, Embedding
from quantax.model import ResConv

# ---------- ResConv ----------


def test_resconv_is_sequential_with_embedding():
    Square = qtx.sites.Square
    Square(2)
    net = ResConv(nblocks=1, channels=4, kernel_size=2)
    assert isinstance(net, Sequential)
    # the embedding comes first, followed by the neighbor-representation conversion
    assert isinstance(net.layers[0], Embedding)


def test_resconv_complex_dtype_raises_with_correct_name():
    # Regression for the "ResSum" copy-paste typo: the message must name ResConv.
    qtx.sites.Square(2)
    with pytest.raises(ValueError, match="ResConv"):
        ResConv(nblocks=1, channels=4, kernel_size=2, dtype=jnp.complex64)


def test_resconv_forward_is_finite_scalar():
    qtx.sites.Square(2)
    net = ResConv(nblocks=1, channels=4, kernel_size=2)
    s = qtx.utils.rand_states(1)[0]
    out = np.asarray(net(s))
    assert out.shape == ()
    assert np.isfinite(out)


def test_resconv_complex_out_dtype_gives_complex_output():
    # out_dtype complex routes the features through pair_cpl, so psi is complex.
    qtx.sites.Square(2)
    net = ResConv(nblocks=1, channels=4, kernel_size=2, out_dtype=jnp.complex64)
    s = qtx.utils.rand_states(1)[0]
    assert jnp.iscomplexobj(jnp.asarray(net(s)))


def test_resconv_use_final_bias():
    # Only the last conv of the last block is affected by use_final_bias;
    # all other convs always have a bias.
    qtx.sites.Square(2)
    net = ResConv(nblocks=2, channels=4, kernel_size=2)
    net_bias = ResConv(nblocks=2, channels=4, kernel_size=2, use_final_bias=True)
    assert net.layers[3].conv2.bias is None
    assert net_bias.layers[3].conv2.bias is not None
    for block in (net.layers[2], net.layers[3], net_bias.layers[2]):
        assert block.conv1.bias is not None
    assert net.layers[2].conv2.bias is not None


def test_resconv_sublattice_invariance_triangularb():
    # Regression for the Embedding / to_neighbor_repr ordering: the sublattice
    # positional encoding must be applied in the original site ordering, where
    # translations are plain grid rolls. On TriangularB, translations act on the
    # neighbor grid as twisted diagonal rolls, so a PE tiled there breaks
    # covariance. The (3, 1) sublattice is the discriminating case; a (2, 2) PE
    # would be accidentally covariant under either ordering.
    qtx.sites.TriangularB(2)
    symm = qtx.symmetry.Translation([[3, 0], [0, 1]])
    net = ResConv(
        nblocks=1, channels=4, kernel_size=3, sublattice=(3, 1), trans_symm=symm
    )
    s = qtx.utils.rand_states(1)[0]
    psi = np.asarray([np.asarray(net(x)) for x in symm.get_symm_spins(s)])
    np.testing.assert_allclose(psi, psi[0], rtol=1e-5)
