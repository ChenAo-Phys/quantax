import numpy as np
import jax
import jax.numpy as jnp
from quantax.sites import Square, Grid, Lattice
from quantax.nn import Embedding, input_to_index
from quantax.global_defs import PARTICLE_TYPE

# Functions are evaluated under jit; eager evaluation compiles every primitive
# separately and dominates the test time.


def _two_site_chain(**kwargs):
    # 2 sites per unit cell, 3 cells -> lattice shape (2, 3), sites ordered as
    # [sub0 over cells, sub1 over cells]
    return Lattice((3,), [[1.0]], [[0.0], [0.3]], **kwargs)


def _random_spins(n, N, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=(n, N)).astype(np.float32)


# ---------- input_to_index ----------


def test_index_spinful_fermion_packs_up_and_down():
    # Config is (up sites, dn sites); each site packs to up + 2 * dn.
    Square(2, particle_type=PARTICLE_TYPE.spinful_fermion)
    up = [1, -1, 1, -1]  # bits 1,0,1,0
    dn = [1, 1, -1, -1]  # bits 1,1,0,0
    s = np.array(up + dn, dtype=np.float32)
    # site 0: 1 + 2*1 = 3, site 1: 0 + 2*1 = 2, site 2: 1, site 3: 0
    np.testing.assert_array_equal(np.asarray(jax.jit(input_to_index)(s)), [3, 2, 1, 0])


def test_index_multisite_cell_packs_sublattice():
    # index[c] = bit(sub0, c) + 2 * bit(sub1, c)
    _two_site_chain()
    s = np.array([1, -1, 1, -1, 1, 1], dtype=np.float32)
    # sub0 bits [1,0,1], sub1 bits [0,1,1] -> [1, 2, 3]
    np.testing.assert_array_equal(np.asarray(jax.jit(input_to_index)(s)), [1, 2, 3])


# ---------- Embedding: gather from the positional encoding ----------


def test_forward_spinful_multisite_cell_gathers_cell_state():
    # Spinful fermions on 2 sites per cell: 4 bits per cell (up sub0, up sub1,
    # dn sub0, dn sub1) select one of 16 entries of a table shared by all cells.
    # The table keeps the given dtype, and so does the output.
    _two_site_chain(particle_type=PARTICLE_TYPE.spinful_fermion)
    emb = Embedding(3, dtype=jnp.float16)
    assert emb.PE.shape == (3, 16, 1) and emb.PE.dtype == jnp.float16
    s = _random_spins(16, 12)
    out = np.asarray(jax.jit(jax.vmap(emb))(s))
    assert out.dtype == np.float16

    bits = (s > 0).reshape(-1, 4, 3)  # (batch, species x sublattice, cell)
    index = np.einsum("bkc,k->bc", bits, 1 << np.arange(4))
    PE = np.asarray(emb.PE)
    np.testing.assert_array_equal(out, np.moveaxis(PE[:, index, 0], 0, 1))


def test_forward_gathers_per_sublattice_position():
    # Each grid position uses the table of its sublattice position, here period 2
    # along both axes, tiled twice along x and once along y. The rectangular grid
    # catches mixed-up axes, and the period along y a PE that only follows x.
    Grid([4, 2])
    emb = Embedding(3, sublattice=(2, 2))
    assert emb.PE.shape == (3, 2, 2, 2)  # one bit per cell
    s = _random_spins(8, 8)
    out = np.asarray(jax.jit(jax.vmap(emb))(s))

    index = (s > 0).reshape(-1, 4, 2).astype(int)
    PE = np.asarray(emb.PE)
    x, y = np.arange(4)[:, None], np.arange(2)[None, :]
    expected = PE[:, index, x % 2, y % 2]  # (d, batch, x, y)
    np.testing.assert_array_equal(out, np.moveaxis(expected, 0, 1))


def test_translation_equivariance_follows_sublattice_period():
    # With a positional encoding of period (2, 2), translating the input translates
    # the embedding only for shifts that are multiples of the period on both axes.
    Grid([4, 2])
    emb = Embedding(3, sublattice=(2, 2))
    s = _random_spins(1, 8)[0].reshape(4, 2)
    shifts = [(2, 0), (0, 1), (2, 1), (1, 0), (3, 1)]
    rolled = np.stack([np.roll(s, v, (0, 1)) for v in shifts]).reshape(-1, 8)
    out = np.asarray(jax.jit(jax.vmap(emb))(np.concatenate([s.reshape(1, 8), rolled])))
    for v, out_v in zip(shifts, out[1:]):
        equivariant = np.allclose(out_v, np.roll(out[0], v, (1, 2)))
        assert equivariant == (v[0] % 2 == 0 and v[1] % 2 == 0), v
