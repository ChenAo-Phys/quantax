import pytest
from quantax.symmetry.symmetry import _permutation_sign


@pytest.fixture(autouse=True)
def _fresh_permutation_sign():
    # `_permutation_sign` bakes get_sites().Ntotal in at trace time, so a trace
    # cached by an earlier test with equal shapes but another Ntotal would be reused
    _permutation_sign.clear_cache()
