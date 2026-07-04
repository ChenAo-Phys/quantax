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
    # layers[0] converts to the neighbor representation; the embedding follows
    assert isinstance(net.layers[1], Embedding)


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
