import numpy as np
import jax.numpy as jnp
from quantax.sites import Square, Lattice
from quantax.nn import Embedding, input_to_index
from quantax.global_defs import PARTICLE_TYPE, get_lattice

# ---------- input_to_index ----------


def test_index_spin_single_site():
    # 2x2 square, one site per cell: index is just the per-site occupation bit.
    Square(2)
    s = jnp.array([1, -1, -1, 1], dtype=jnp.float32)
    np.testing.assert_array_equal(np.asarray(input_to_index(s)), [1, 0, 0, 1])


def test_index_spinless_fermion_matches_spin():
    # Spinless fermions carry one occupation bit per site, exactly like spins.
    Square(2, particle_type=PARTICLE_TYPE.spinless_fermion, Nparticles=2)
    s = jnp.array([1, -1, -1, 1], dtype=jnp.float32)
    np.testing.assert_array_equal(np.asarray(input_to_index(s)), [1, 0, 0, 1])


def test_index_spinful_fermion_packs_up_and_down():
    # Config is (up sites, dn sites); each site packs to up + 2 * dn.
    Square(2, particle_type=PARTICLE_TYPE.spinful_fermion)
    up = [1, -1, 1, -1]  # bits 1,0,1,0
    dn = [1, 1, -1, -1]  # bits 1,1,0,0
    s = jnp.array(up + dn, dtype=jnp.float32)
    # site 0: 1 + 2*1 = 3, site 1: 0 + 2*1 = 2, site 2: 1, site 3: 0
    np.testing.assert_array_equal(np.asarray(input_to_index(s)), [3, 2, 1, 0])


def test_index_multisite_cell_packs_sublattice():
    # 2 sites per unit cell, 3 cells: index[c] = bit(sub0, c) + 2 * bit(sub1, c).
    Lattice(extent=(3,), basis_vectors=[[1.0]], site_offsets=[[0.0], [0.3]])
    # layout (s_per_cell=2, spatial=3): [sub0 over cells, sub1 over cells]
    s = jnp.array([1, -1, 1, -1, 1, 1], dtype=jnp.float32)
    # sub0 bits [1,0,1], sub1 bits [0,1,1] -> [1, 2, 3]
    np.testing.assert_array_equal(np.asarray(input_to_index(s)), [1, 2, 3])


# ---------- Embedding shapes / table sizing ----------


def test_table_size_and_output_shape_spin():
    Square(2)
    emb = Embedding(3)
    assert emb.PE.shape == (3, 1 << 1, 1, 1)  # one bit per cell, trivial sublattice
    s = jnp.array([1, -1, -1, 1], dtype=jnp.float32)
    assert emb(s).shape == (3, 2, 2)  # (d, *spatial)


def test_table_size_spinful_fermion():
    Square(2, particle_type=PARTICLE_TYPE.spinful_fermion)
    emb = Embedding(3)
    assert emb.PE.shape == (3, 1 << 2, 1, 1)  # two bits per cell
    s = jnp.ones(8, dtype=jnp.float32)
    assert emb(s).shape == (3, 2, 2)


def test_table_size_multisite_cell():
    Lattice(extent=(4,), basis_vectors=[[1.0]], site_offsets=[[0.0], [0.3]])
    emb = Embedding(3)
    assert emb.PE.shape == (3, 1 << 2, 1)
    s = jnp.array([1, 1, -1, -1, 1, -1, 1, -1], dtype=jnp.float32)
    assert emb(s).shape == (3, 4)


def test_table_size_with_sublattice():
    Square(4)
    emb = Embedding(3, sublattice=(2, 2))
    assert emb.PE.shape == (3, 1 << 1, 2, 2)


# ---------- forward pass: gather from PE ----------


def test_forward_is_table_gather():
    Square(2)
    emb = Embedding(4)
    s = jnp.array([1, -1, -1, 1], dtype=jnp.float32)
    idx = input_to_index(s)
    expected = emb.PE[:, idx, 0, 0].reshape(4, 2, 2)
    np.testing.assert_allclose(np.asarray(emb(s)), np.asarray(expected))


def test_forward_gathers_per_sublattice_position():
    # Each grid position uses the table of its sublattice position.
    Square(4)
    emb = Embedding(3, sublattice=(2, 2))
    s = jnp.sign(jnp.cos(jnp.arange(16, dtype=jnp.float32))).astype(jnp.float32)
    idx = np.asarray(input_to_index(s)).reshape(4, 4)
    PE = np.asarray(emb.PE)
    expected = np.empty((3, 4, 4), dtype=PE.dtype)
    for x in range(4):
        for y in range(4):
            expected[:, x, y] = PE[:, idx[x, y], x % 2, y % 2]
    np.testing.assert_allclose(np.asarray(emb(s)), expected)


# ---------- dtype ----------


def test_dtype_propagates():
    Square(2)
    emb = Embedding(2, dtype=jnp.float16)
    assert emb.PE.dtype == jnp.float16
    assert emb(jnp.ones(4, dtype=jnp.float32)).dtype == jnp.float16


# ---------- (sublattice) translation invariance ----------


def _roll_spatial(x, shift):
    # Roll the spatial axes of an array shaped (*, *spatial) by ``shift``.
    axes = tuple(range(x.ndim - len(shift), x.ndim))
    return jnp.roll(x, shift=shift, axis=axes)


def _roll_config(s, shift):
    # Translate a flat configuration by ``shift`` on the spatial grid.
    s = s.reshape(get_lattice().shape)
    return _roll_spatial(s, shift).reshape(-1)


def test_translation_equivariant_without_sublattice():
    # With a trivial sublattice the table gather is equivariant under *any* spatial
    # translation: shifting the input shifts the embedding by the same amount.
    Square(4)
    emb = Embedding(3)
    s = jnp.sign(jnp.cos(jnp.arange(16, dtype=jnp.float32))).astype(jnp.float32)
    shift = (1, 2)
    np.testing.assert_allclose(
        np.asarray(emb(_roll_config(s, shift))),
        np.asarray(_roll_spatial(emb(s), shift)),
        rtol=1e-6,
    )


def test_sublattice_translation_equivariant_with_pe():
    # With a positional encoding of period (2, 2), equivariance survives only for
    # translations by a multiple of that period.
    Square(4)
    emb = Embedding(3, sublattice=(2, 2))
    s = jnp.sign(jnp.cos(jnp.arange(16, dtype=jnp.float32))).astype(jnp.float32)
    shift = (2, 2)
    np.testing.assert_allclose(
        np.asarray(emb(_roll_config(s, shift))),
        np.asarray(_roll_spatial(emb(s), shift)),
        rtol=1e-6,
    )


def test_nonsublattice_translation_breaks_equivariance_with_pe():
    # A shift that is not a multiple of the PE period must break equivariance,
    # otherwise the positional encoding would be doing nothing.
    Square(4)
    emb = Embedding(3, sublattice=(2, 2))
    s = jnp.sign(jnp.cos(jnp.arange(16, dtype=jnp.float32))).astype(jnp.float32)
    shift = (1, 0)
    assert not np.allclose(
        np.asarray(emb(_roll_config(s, shift))),
        np.asarray(_roll_spatial(emb(s), shift)),
    )
