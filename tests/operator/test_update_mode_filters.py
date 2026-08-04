import pytest
from quantax.sites import Sites
from quantax.operator.update_mode_filters import (
    none_filter,
    nflips_filter,
    nflips_up_dn_filter,
    DIAGONAL_OPS,
)


def test_none_filter_is_always_empty():
    assert none_filter("xx", [0, 1]) == {}
    assert none_filter("+-", [0, 1]) == {}
    assert none_filter("Inz", [0, 1, 2]) == {}


@pytest.mark.parametrize(
    "opstr, expected",
    [
        ("I", 0),
        ("z", 0),
        ("n", 0),
        ("nz", 0),
        ("x", 1),
        ("y", 1),
        ("+", 1),
        ("-", 1),
        ("+-", 2),  # hopping / spin exchange
        ("+-z", 2),  # diagonal factor does not add a flip
        ("xyxy", 4),
    ],
)
def test_nflips_counts_offdiagonal_operators(opstr, expected):
    assert nflips_filter(opstr, list(range(len(opstr)))) == {"nflips": expected}


@pytest.mark.parametrize(
    "opstr, indices, expected",
    [
        # up hopping: both modes below Nsites
        ("+-", [0, 1], {"nflips_up": 2, "nflips_dn": 0}),
        # down hopping: both modes above Nsites
        ("+-", [6, 7], {"nflips_up": 0, "nflips_dn": 2}),
        # one flip per sector
        ("+-", [0, 6], {"nflips_up": 1, "nflips_dn": 1}),
        # diagonal density-density term
        ("nn", [0, 6], {"nflips_up": 0, "nflips_dn": 0}),
        # diagonal characters don't count regardless of their sector
        ("+-nn", [2, 3, 0, 6], {"nflips_up": 2, "nflips_dn": 0}),
        # two hops in each sector
        ("+-+-", [0, 1, 6, 7], {"nflips_up": 2, "nflips_dn": 2}),
    ],
)
def test_nflips_up_dn_counts_sectors(opstr, indices, expected):
    Sites(6, particle_type="spinful_fermion")
    assert nflips_up_dn_filter(opstr, indices) == expected


@pytest.mark.parametrize("particle_type", ["spin", "spinless_fermion"])
def test_nflips_up_dn_requires_spinful_fermions(particle_type):
    Sites(6, particle_type=particle_type)
    with pytest.raises(ValueError, match="spinful fermion"):
        nflips_up_dn_filter("+-", [0, 1])


def test_diagonal_ops_are_exactly_the_zero_flip_chars():
    # every diagonal character contributes zero flips, every other known
    # operator character contributes one
    for op in DIAGONAL_OPS:
        assert nflips_filter(op, [0])["nflips"] == 0
    for op in ("x", "y", "+", "-"):
        assert op not in DIAGONAL_OPS
        assert nflips_filter(op, [0])["nflips"] == 1
