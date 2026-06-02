import pytest
import jax
import quantax as qtx
from quantax.sites import Sites


@pytest.fixture(autouse=True)
def reset_quantax_globals():
    """
    Reset global state in quantax before/after each test to ensure isolation.
    Quantax stores global state in:
    - qtx.global_defs.KEY (random seed)
    - qtx.sites.Sites._SITES (global lattice definition)
    """
    # Setup: Reset to default state before each test
    qtx.set_random_seed(42)
    qtx.set_default_dtype(jax.numpy.float64)
    Sites._SITES = None

    yield

    # Teardown: Clean up again
    Sites._SITES = None
