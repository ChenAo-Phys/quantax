import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

import quantax as qtx
from quantax.model import ResConv
from quantax.utils import LogArray

# ---------- ResConv ----------


# All tests use the TriangularB(2) lattice and 4 channels with kernel size 3, so that
# the eager compilations of the layer initializers are shared.


def test_resconv_complex_dtype_raises_with_correct_name():
    # Regression for the "ResSum" copy-paste typo: the message must name ResConv.
    qtx.sites.TriangularB(2)
    with pytest.raises(ValueError, match="ResConv"):
        ResConv(nblocks=1, channels=4, kernel_size=3, dtype=jnp.complex64)


@pytest.mark.parametrize(
    "nblocks, use_rmsnorm, use_final_bias, out_dtype",
    [(2, False, False, None), (1, True, True, jnp.complex64)],
)
def test_resconv_triangularb_covariance_bias_layout_and_rmsnorm(
    nblocks, use_rmsnorm, use_final_bias, out_dtype
):
    # Regression for the Embedding / to_neighbor_repr ordering: the sublattice
    # positional encoding must be applied in the original site ordering, where
    # translations are plain grid rolls. On TriangularB, translations act on the
    # neighbor grid as twisted diagonal rolls, so a PE tiled there breaks
    # covariance. The (3, 1) sublattice is the discriminating case; a (2, 2) PE
    # would be accidentally covariant under either ordering.
    # RMSNorm normalizes over all positions, so it must keep the covariance. The
    # default final activation keeps psi a LogArray, and a complex out_dtype routes
    # the features through pair_cpl so that psi is complex.
    lattice = qtx.sites.TriangularB(2)
    symm = qtx.symmetry.Translation([[3, 0], [0, 1]])
    net = ResConv(
        nblocks=nblocks,
        channels=4,
        kernel_size=3,
        sublattice=(3, 1),
        use_rmsnorm=use_rmsnorm,
        use_final_bias=use_final_bias,
        trans_symm=symm,
        out_dtype=out_dtype,
    )
    # Only the last conv of the last block follows use_final_bias, all other convs
    # always have a bias.
    blocks = net.layers[2 : 2 + nblocks]
    for i, block in enumerate(blocks):
        assert block.conv1.bias is not None
        assert (block.conv2.bias is not None) == (use_final_bias or i < nblocks - 1)
    rng = np.random.default_rng(0)
    s = jnp.asarray(rng.choice([-1, 1], size=12), jnp.int8)
    x = jnp.asarray(rng.normal(size=(4, *lattice.shape[1:])), jnp.float32)

    @eqx.filter_jit
    def check(net, s_symm, x):
        block = net.layers[2]
        # the residual branch of the first block, f(x) = block(x) - x
        branch = jnp.stack([block(x) - x, block(2 * x) - 2 * x])
        return jax.vmap(net)(s_symm), branch

    psi, branch = check(net, symm.get_symm_spins(s), x)
    # RMSNorm makes the residual branch invariant under rescaling the block input (up
    # to its eps), unlike the default normalization x / sqrt(i + 1). A block without
    # the residual connection would break this too.
    branch = np.asarray(branch)
    invariant = np.allclose(branch[1], branch[0], rtol=0, atol=1e-4 * abs(branch).max())
    assert invariant == use_rmsnorm
    assert isinstance(psi, LogArray)
    psi = np.asarray(psi.value())
    assert np.all(np.isfinite(psi))
    if out_dtype is None:
        assert np.isrealobj(psi)
    else:  # a genuine phase, not just a complex dtype
        assert abs(psi[0].imag) > 1e-3 * abs(psi[0])
    np.testing.assert_allclose(psi, psi[0], rtol=1e-5)
