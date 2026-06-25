import numpy as np
import jax.numpy as jnp
import quantax as qtx
from quantax.sites import Square, Chain
from quantax.symmetry import Translation, TransND
from _dtype import use_dtype

# --- basic construction ---


def test_transnd_square():
    Square(4)
    t = TransND()
    # full 2D translation group on a 4x4 lattice has 16 elements
    assert t.nsymm == 16
    assert np.all(np.asarray(t._perm_sign) == 1)


def test_translation_incommensurate_raises():
    Square(4)
    import pytest

    with pytest.raises(ValueError):
        Translation((3, 0))  # 3 does not divide 4


def test_translation_open_boundary_raises():
    Square(4, boundary=0)
    import pytest

    with pytest.raises(ValueError):
        Translation((1, 0))


# --- negative translation vectors (regression) ---


def test_negative_translation_is_inverse():
    # A negative translation vector must not crash and should generate the same
    # cyclic group as its positive counterpart (it is the group inverse).
    Square(4)
    tp = Translation((1, 0))
    tn = Translation((-1, 0))
    assert tp.nsymm == tn.nsymm

    gp = np.asarray(tp._generator[0])
    gn = np.asarray(tn._generator[0])
    # the two generators are inverse permutations of each other
    assert np.array_equal(gp[gn], np.arange(gp.size))
    # and they span the same set of group permutations
    sp = set(map(tuple, np.asarray(tp._perm).tolist()))
    sn = set(map(tuple, np.asarray(tn._perm).tolist()))
    assert sp == sn


def test_negative_translation_apbc_sign():
    # Anti-periodic boundary: the site that wraps across the boundary picks up a
    # -1, regardless of the translation direction.
    Chain(4, boundary=-1, particle_type="spinless_fermion", Nparticles=2)
    tp = Translation(1)
    tn = Translation(-1)
    # +1 moves site 3 across the boundary, -1 moves site 0 across it
    assert np.array_equal(np.asarray(tp._generator_sign[0]), [1, 1, 1, -1])
    assert np.array_equal(np.asarray(tn._generator_sign[0]), [-1, 1, 1, 1])


def test_negative_translation_momentum_mirror(x64):
    # T(1) in sector q projects onto the same momentum subspace as
    # T(-1) in sector (N - q), since T(-1) = T(1)^{-1}.
    use_dtype(jnp.complex128)
    N = 6
    Chain(N)

    def projector(symm):
        perm = np.asarray(symm._perm)
        chi = np.asarray(symm._character)
        P = np.zeros((N, N), complex)
        for g, c in zip(perm, chi):
            Pg = np.zeros((N, N), complex)
            Pg[np.arange(N), g] = 1
            P += np.conj(c) * Pg
        return P / len(chi)

    for q in range(N):
        Pp = projector(Translation(1, sector=q))
        Pn = projector(Translation(-1, sector=(N - q) % N))
        assert np.allclose(Pp, Pn)
