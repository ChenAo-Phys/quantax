import pytest
from quantax.operator.update_mode_filters import (
    none_filter,
    nflips_filter,
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


def test_diagonal_ops_are_exactly_the_zero_flip_chars():
    # every diagonal character contributes zero flips, every other known
    # operator character contributes one
    for op in DIAGONAL_OPS:
        assert nflips_filter(op, [0])["nflips"] == 0
    for op in ("x", "y", "+", "-"):
        assert op not in DIAGONAL_OPS
        assert nflips_filter(op, [0])["nflips"] == 1
