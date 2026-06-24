import numpy as np
import jax.numpy as jnp
import pytest

import quantax as qtx
from quantax.nn import Sequential, Embedding, GConv, ReshapeConv, exp_by_scale
from quantax.model import ResConv, ResGConv
from quantax.symmetry import C4v

# ---------- ResConv ----------


def test_resconv_is_sequential_with_embedding_first():
    Square = qtx.sites.Square
    Square(2)
    net = ResConv(nblocks=1, channels=4, kernel_size=2)
    assert isinstance(net, Sequential)
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


# ---------- ResGConv ----------


def _spinful_square():
    return qtx.sites.Square(
        2, particle_type="spinful_fermion", Nparticles=(2, 2), double_occ=True
    )


def test_resgconv_is_sequential_subclass():
    # ResGConv is now a class (a Sequential subclass), not a factory function.
    _spinful_square()
    net = ResGConv(nblocks=1, channels=3, pg_symm=C4v())
    assert isinstance(net, ResGConv)
    assert isinstance(net, Sequential)
    # layers[0] reshapes the config to the grid; layers[1] is the GConv embedding.
    assert isinstance(net.layers[0], ReshapeConv)
    assert isinstance(net.layers[1], GConv)


def test_resgconv_stores_config_fields():
    _spinful_square()
    pg = C4v()
    net = ResGConv(nblocks=2, channels=5, pg_symm=pg, project=False)
    assert net.nblocks == 2
    assert net.channels == 5
    assert net.project is False
    assert net.dtype == jnp.float32
    assert net.pg_symm is pg
    assert net.final_activation is exp_by_scale


def test_resgconv_forward_is_finite_scalar_when_projecting():
    _spinful_square()
    net = ResGConv(nblocks=1, channels=3, pg_symm=C4v(), project=True)
    s = qtx.utils.rand_states(1)[0]
    out = np.asarray(net(s))
    assert out.shape == ()
    assert np.isfinite(out)


def test_resgconv_project_false_output_shape():
    # Without projection the equivariant group axis is kept: (channels, npoint, ntrans).
    lattice = _spinful_square()
    pg = C4v()
    net = ResGConv(nblocks=1, channels=3, pg_symm=pg, project=False)
    s = qtx.utils.rand_states(1)[0]
    out = jnp.asarray(net(s))
    ntrans = int(np.prod(lattice.shape[1:]))
    assert out.shape == (3, pg.nsymm, ntrans)


def test_resgconv_complex_dtype_raises():
    _spinful_square()
    with pytest.raises(ValueError, match="ResGConv"):
        ResGConv(nblocks=1, channels=3, pg_symm=C4v(), dtype=jnp.complex64)


def test_resgconv_runs_on_spin_lattice():
    # The layer-0 lift now handles any input-channel count, including a spin
    # system (1 channel) which previously failed (the lift hardcoded 2 channels).
    qtx.sites.Square(2)  # spin: 1 input channel
    net = ResGConv(nblocks=1, channels=3, pg_symm=C4v())
    s = qtx.utils.rand_states(1)[0]
    out = np.asarray(net(s))
    assert out.shape == ()
    assert np.isfinite(out)


def test_resgconv_lift_is_equivariant_on_spin_lattice():
    # The projected output must be invariant under the full symmetry group in the
    # trivial sector -- this holds only if the generalized lift stays exactly
    # equivariant, so the npoint*ntrans feature entries are the symmetry orbit.
    from quantax.symmetry import TransND

    qtx.sites.Square(2)
    pg = C4v()
    net = ResGConv(nblocks=1, channels=3, pg_symm=pg, project=True)
    s = qtx.utils.rand_states(1)[0]
    psi = np.asarray(jnp.asarray(net(s)))
    for perm in (TransND() @ pg)._perm:
        psi_perm = np.asarray(jnp.asarray(net(s[perm])))
        np.testing.assert_allclose(psi_perm, psi, rtol=1e-4)
