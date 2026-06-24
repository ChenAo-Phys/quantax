import numpy as np
import jax.numpy as jnp
import pytest
import quantax as qtx
from quantax.sites import Chain, Square, Pyrochlore, Sites
from quantax.operator import (
    sigma_x,
    sigma_y,
    sigma_z,
    sigma_p,
    sigma_m,
    S_x,
    S_z,
    create,
    create_u,
    create_d,
    annihilate,
    annihilate_d,
    number,
    number_u,
    number_d,
)
from _dtype import use_dtype


def _index(op) -> int:
    """The single flat site index an operator acts on."""
    (term,) = op.op_list
    (inds,) = term.indices
    (idx,) = inds
    return idx


def _strength(op) -> complex:
    """The strength of a single-site operator."""
    return op.op_list[0].strength[0]


def _dense(op) -> np.ndarray:
    return np.asarray(op.todense())


# --- single-site spin matrices (QuSpin orders basis as |up>, |down>) ---


def test_pauli_single_site_matrices(x64):
    use_dtype(jnp.complex128)
    Chain(1, boundary=1)
    assert np.allclose(_dense(sigma_x(0)), [[0, 1], [1, 0]])
    assert np.allclose(_dense(sigma_y(0)), [[0, -1j], [1j, 0]])
    assert np.allclose(_dense(sigma_z(0)), [[1, 0], [0, -1]])
    assert np.allclose(_dense(sigma_p(0)), [[0, 1], [0, 0]])
    assert np.allclose(_dense(sigma_m(0)), [[0, 0], [1, 0]])


def test_spin_half_is_pauli_over_two():
    Chain(1, boundary=1)
    assert np.allclose(_dense(S_x(0)), _dense(sigma_x(0)) / 2)
    assert np.allclose(_dense(S_z(0)), _dense(sigma_z(0)) / 2)


def test_pauli_algebra_two_site(x64):
    use_dtype(jnp.complex128)
    Chain(2, boundary=1)
    x, y, z = sigma_x(0), sigma_y(0), sigma_z(0)
    # sigma_x sigma_y = i sigma_z
    assert np.allclose(_dense(x @ y), 1j * _dense(z))
    # {sigma_x, sigma_y} = 0
    assert np.allclose(_dense(x @ y) + _dense(y @ x), 0)
    # sigma_x^2 = I
    assert np.allclose(_dense(x @ x), np.eye(4))


def test_sigma_y_requires_complex_dtype():
    Chain(1, boundary=1)  # default dtype is float32 from the conftest fixture
    with pytest.raises(RuntimeError):
        sigma_y(0)


# --- fermionic site operators (Jordan-Wigner, QuSpin convention) ---


def test_fermion_anticommutation():
    Chain(3, boundary=1, particle_type="spinless_fermion")
    eye = np.eye(2**3)
    c0, c0d = _dense(annihilate(0)), _dense(create(0))
    c1d = _dense(create(1))
    # {c_i, c_i^†} = I
    assert np.allclose(c0 @ c0d + c0d @ c0, eye)
    # {c_0, c_1^†} = 0
    assert np.allclose(c0 @ c1d + c1d @ c0, 0)


def test_fermion_number_operator():
    Chain(3, boundary=1, particle_type="spinless_fermion")
    n0 = _dense(number(0))
    c0, c0d = _dense(annihilate(0)), _dense(create(0))
    # n = c^† c, and it is a projector (diagonal 0/1)
    assert np.allclose(n0, c0d @ c0)
    assert np.allclose(n0 @ n0, n0)


def test_spin_operators_reject_fermion_system():
    Chain(2, boundary=1, particle_type="spinless_fermion")
    with pytest.raises(RuntimeError):
        sigma_p(0)


# --- coordinate indexing with boundary sign ---


def test_coordinate_index_wraps_with_boundary_sign():
    # On an anti-periodic chain, addressing site -1 should map to the last site
    # and carry the boundary sign into the strength.
    Chain(4, boundary=-1, particle_type="spinless_fermion")
    op_wrap = sigma_z(-1)
    op_last = sigma_z(3)
    assert op_wrap.op_list[0].indices == op_last.op_list[0].indices
    assert np.isclose(op_wrap.op_list[0].strength[0], -op_last.op_list[0].strength[0])


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
    # Each crossing of the anti-periodic boundary flips the sign.
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


def test_spinful_number_operators_are_distinct_projectors(x64):
    use_dtype(jnp.complex128)
    Chain(1, particle_type="spinful_fermion")
    nu, nd = _dense(number_u(0)), _dense(number_d(0))
    # Each number operator is a diagonal projector, and the two spins differ.
    assert np.allclose(nu @ nu, nu)
    assert np.allclose(nd @ nd, nd)
    assert np.allclose(nu @ nd, nd @ nu)
    assert not np.allclose(nu, nd)


def test_spinful_ops_reject_spinless_system():
    Chain(2, particle_type="spinless_fermion")
    with pytest.raises(RuntimeError):
        create_u(0)
    with pytest.raises(RuntimeError):
        number_d(0)


def test_spinless_ops_reject_spin_system():
    Chain(2)  # spin system
    with pytest.raises(RuntimeError):
        create(0)
    with pytest.raises(RuntimeError):
        number(0)
