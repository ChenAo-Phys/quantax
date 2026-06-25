import pytest
import jax
import quantax as qtx
import quantax.symmetry.common_symmetries as common_symmetries
from quantax.sites import Sites


def _reset_globals():
    qtx.set_random_seed(42)
    qtx.set_default_dtype(jax.numpy.float32)
    Sites._SITES = None
    # The Identity / Z2Inversion symmetries are memoized module-level singletons
    # whose cached QuSpin basis would otherwise outlive the lattice and lock its
    # dimension across tests (see common_symmetries._Identity / _Z2Inverse).
    common_symmetries._Identity = None
    common_symmetries._Z2Inverse = dict()


@pytest.fixture(autouse=True)
def reset_quantax_globals():
    """
    Reset global state in quantax before/after each test to ensure isolation.
    Quantax stores global state in:
    - qtx.global_defs.KEY (random seed)
    - qtx.sites.Sites._SITES (global lattice definition)
    - the memoized symmetry singletons in quantax.symmetry.common_symmetries
    """
    # Setup: Reset to default state before each test
    _reset_globals()

    yield

    # Teardown: Clean up again
    _reset_globals()


@pytest.fixture
def x64():
    """Enable 64-bit precision for the scope of a test.

    Pair with ``use_dtype`` (see ``tests/_dtype.py``) to set the matching
    Quantax default dtype. Using this fixture instead of
    ``qtx.set_default_dtype(jnp.float64)`` keeps ``jax_enable_x64`` scoped to
    the test so it never leaks into single-precision tests.
    """
    with jax.enable_x64():
        yield
