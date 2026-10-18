import numpy as np
import jax.numpy as jnp
import pytest
from quantax.sites import Chain, Square, Pyrochlore, Sites
from quantax.operator import (
    sigma_x,
    sigma_y,
    sigma_z,
    sigma_p,
    sigma_m,
    S_x,
    S_y,
    S_z,
    S_p,
    S_m,
    create,
    create_u,
    create_d,
    annihilate,
    annihilate_u,
    annihilate_d,
    number,
    number_u,
    number_d,
)
from quantax.symmetry import Identity
from quantax.utils import ints_to_array
from _dtype import use_dtype
from _op_helpers import dense

_SPINLESS_OPS = [create, annihilate, number]
_SPINFUL_OPS = [create_u, create_d, annihilate_u, annihilate_d, number_u, number_d]


def _index(op) -> int:
    """The single flat site index an operator acts on."""
    (term,) = op.op_list
    (inds,) = term.indices
    (idx,) = inds
    return idx


def _strength(op) -> complex:
    """The strength of a single-site operator."""
    return op.op_list[0].strength[0]


# --- single-site spin matrices (QuSpin orders basis as |up>, |down>) ---


def _on_site0(m) -> np.ndarray:
    """A 2x2 matrix on site 0 of a 4-site chain, the leading factor in QuSpin."""
    return np.kron(m, np.eye(8))


def test_spin_single_site_matrices():
    use_dtype(jnp.complex64)
    Chain(4, boundary=1)
    sx, sy, sz = [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]
    sp, sm = [[0, 1], [0, 0]], [[0, 0], [1, 0]]
    assert np.allclose(dense(sigma_x(0)), _on_site0(sx))
    assert np.allclose(dense(sigma_y(0)), _on_site0(sy))
    assert np.allclose(dense(sigma_z(0)), _on_site0(sz))
    assert np.allclose(dense(sigma_p(0)), _on_site0(sp))
    assert np.allclose(dense(sigma_m(0)), _on_site0(sm))
    # spin-1/2 operators, with S^± = sigma^±
    assert np.allclose(dense(S_x(0)), _on_site0(np.divide(sx, 2)))
    assert np.allclose(dense(S_y(0)), _on_site0(np.divide(sy, 2)))
    assert np.allclose(dense(S_z(0)), _on_site0(np.divide(sz, 2)))
    assert np.allclose(dense(S_p(0)), _on_site0(sp))
    assert np.allclose(dense(S_m(0)), _on_site0(sm))


def test_pauli_algebra():
    use_dtype(jnp.complex64)
    Chain(4, boundary=1)
    x, y, z = sigma_x(1), sigma_y(1), sigma_z(1)
    # sigma_x sigma_y = i sigma_z
    assert np.allclose(dense(x @ y), 1j * dense(z))
    # {sigma_x, sigma_y} = 0
    assert np.allclose(dense(x @ y) + dense(y @ x), 0)
    # sigma_x^2 = I
    assert np.allclose(dense(x @ x), np.eye(16))


@pytest.mark.parametrize("op", [sigma_y, S_y])
def test_y_requires_complex_dtype(op):
    Chain(1, boundary=1)  # default dtype is float32 from the conftest fixture
    with pytest.raises(RuntimeError, match="real default data types"):
        op(0)


# --- fermionic site operators (Jordan-Wigner, QuSpin convention) ---


def _anticommutator(a, b):
    a, b = dense(a), dense(b)
    return a @ b + b @ a


def test_spinless_fermion_anticommutation():
    Chain(3, boundary=1, particle_type="spinless_fermion")
    eye = np.eye(2**3)
    assert np.allclose(_anticommutator(annihilate(0), create(0)), eye)
    # the Jordan-Wigner string makes distant modes anticommute
    assert np.allclose(_anticommutator(annihilate(0), create(2)), 0)
    assert np.allclose(_anticommutator(annihilate(0), annihilate(1)), 0)


def test_spinful_fermion_anticommutation():
    Chain(3, boundary=1, particle_type="spinful_fermion")
    eye = np.eye(4**3)
    assert np.allclose(_anticommutator(annihilate_u(1), create_u(1)), eye)
    assert np.allclose(_anticommutator(annihilate_d(1), create_d(1)), eye)
    # spin-up and spin-down modes anticommute, on the same and different sites
    assert np.allclose(_anticommutator(annihilate_u(0), create_d(0)), 0)
    assert np.allclose(_anticommutator(annihilate_d(0), create_u(2)), 0)
    assert np.allclose(_anticommutator(annihilate_u(0), create_u(2)), 0)


@pytest.mark.parametrize(
    "particle_type, ops",
    [
        ("spinless_fermion", [(number, create, annihilate, 0)]),
        (
            "spinful_fermion",
            [
                (number_u, create_u, annihilate_u, 0),
                (number_d, create_d, annihilate_d, 3),
            ],
        ),
    ],
)
def test_number_operators_count_configuration_occupation(particle_type, ops):
    # n = c^† c is diagonal and reads the occupation of the mode in the ±1
    # configuration convention of quantax (spin-down modes at idx + Nsites)
    Chain(3, boundary=1, particle_type=particle_type)
    symm = Identity()
    symm.basis_make()
    occupied = ints_to_array(symm.basis.states) > 0
    for number_op, create_op, annihilate_op, offset in ops:
        for i in range(3):
            n = dense(number_op(i))
            assert np.allclose(n, dense(create_op(i)) @ dense(annihilate_op(i)))
            assert np.allclose(n, np.diag(occupied[:, i + offset]))


def test_spin_flips_and_spinful_ops_reject_spinless_fermion_system():
    Chain(2, boundary=1, particle_type="spinless_fermion")
    for op in [sigma_p, sigma_m, S_p, S_m] + _SPINFUL_OPS:
        with pytest.raises(RuntimeError):
            op(0)


def test_fermion_ops_reject_spin_system():
    Chain(2)  # spin system
    for op in _SPINLESS_OPS + _SPINFUL_OPS:
        with pytest.raises(RuntimeError):
            op(0)


# --- coordinate indexing with boundary sign ---


def test_square_coordinate_matches_index_from_xyz():
    # On a 2D lattice with one site per cell, (x, y) and (sublattice, x, y) must
    # both resolve to the flat index recorded in index_from_xyz.
    lat = Square(3, boundary=1)
    for x in range(3):
        for y in range(3):
            expected = int(lat.index_from_xyz[0, x, y])
            assert _index(sigma_z(x, y)) == expected
            assert _index(sigma_z(0, x, y)) == expected


def test_sublattice_index_with_multiple_sites_per_cell():
    # Pyrochlore has 4 sites per unit cell, so the sublattice index is mandatory.
    lat = Pyrochlore(1, boundary=1)
    assert lat.shape[0] == 4
    for c in range(4):
        assert _index(sigma_z(c, 0, 0, 0)) == int(lat.index_from_xyz[c, 0, 0, 0])
    # Omitting the sublattice index is ambiguous and must be rejected.
    with pytest.raises(ValueError, match="doesn't match the lattice shape"):
        sigma_z(0, 0, 0)


# --- boundary-condition signs on out-of-range coordinates ---


def test_pbc_wrap_has_unit_sign():
    Square(3, boundary=1)
    # (3, 0) wraps to (0, 0) with a +1 periodic sign.
    assert _index(sigma_z(3, 0)) == _index(sigma_z(0, 0))
    assert np.isclose(_strength(sigma_z(3, 0)), _strength(sigma_z(0, 0)))


def test_apbc_wrap_counts_boundary_crossings():
    Chain(4, boundary=-1, particle_type="spinless_fermion")
    base = _strength(sigma_z(0))
    # Each crossing of the anti-periodic boundary flips the sign, e.g. site -1
    # addresses the last site with a flipped sign.
    assert _index(sigma_z(-1)) == 3 and np.isclose(_strength(sigma_z(-1)), -base)
    assert _index(sigma_z(4)) == 0 and np.isclose(_strength(sigma_z(4)), -base)
    assert _index(sigma_z(8)) == 0 and np.isclose(_strength(sigma_z(8)), base)
    assert _index(sigma_z(-4)) == 0 and np.isclose(_strength(sigma_z(-4)), -base)


def test_obc_wrap_gives_zero_strength():
    # On an open chain a coordinate past the edge has no image, so the boundary
    # sign (0 ** crossings) zeroes the strength; in-range sites are untouched.
    Chain(4, boundary=0)
    assert np.isclose(_strength(sigma_z(4)), 0.0)
    assert not np.isclose(_strength(sigma_z(3)), 0.0)


# --- error messages for invalid indices ---


def test_non_lattice_requires_single_in_range_index():
    Sites(4)  # generic sites, not a lattice
    with pytest.raises(ValueError, match="non-lattice"):
        sigma_z(4)  # flat index out of range
    with pytest.raises(ValueError, match="non-lattice"):
        sigma_z(0, 0)  # coordinate form unsupported without a lattice


def test_lattice_rejects_mismatched_index():
    Square(3, boundary=1)  # shape (1, 3, 3), Nsites = 9
    with pytest.raises(ValueError, match="doesn't match the lattice shape"):
        sigma_z(0, 1, 2, 3)  # too many coordinates
    with pytest.raises(ValueError, match="doesn't match the lattice shape"):
        sigma_z(100)  # out-of-range single integer is treated as coordinates


# --- spinful fermion modes (spin-down lives at idx + Nsites) ---


def test_spinful_down_spin_offset():
    Chain(3, particle_type="spinful_fermion")  # Nsites = 3, modes 0..5
    assert _index(create_u(1)) == 1
    assert _index(create_d(1)) == 1 + 3
    assert _index(annihilate_d(0)) == 0 + 3
    assert _index(number_d(2)) == 2 + 3
    # the down-spin offset is added after wrapping the coordinate
    assert _index(create_d(4)) == 1 + 3
