import pytest
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import quantax as qtx
from quantax import global_defs
from quantax.sites import Sites, Chain
from quantax.utils import get_replicated_sharding


def _key_data(keys):
    return np.asarray(jr.key_data(keys))


def test_random_seed_reproducible():
    qtx.set_random_seed(3)
    first = _key_data(qtx.get_subkeys(3))
    second = _key_data(qtx.get_subkeys())
    third = _key_data(qtx.get_subkeys(3))
    # every draw advances the global key, and the keys of one draw are distinct
    assert first.shape[0] == 3 and len(np.unique(first, axis=0)) == 3
    handed_out = np.concatenate([first, second[None], third])
    assert len(np.unique(handed_out, axis=0)) == 7
    # the global key is never one of the keys handed out, which would correlate
    # later draws with a key the caller already consumed
    key = _key_data(global_defs.KEY)
    assert not any(np.array_equal(key, k) for k in handed_out)
    # resetting the seed replays the same sequence; another seed gives other keys
    qtx.set_random_seed(3)
    np.testing.assert_array_equal(_key_data(qtx.get_subkeys(3)), first)
    np.testing.assert_array_equal(_key_data(qtx.get_subkeys()), second)
    np.testing.assert_array_equal(_key_data(qtx.get_subkeys(3)), third)
    qtx.set_random_seed(4)
    assert not np.array_equal(_key_data(qtx.get_subkeys(3)), first)
    # the global key is replicated on every device
    assert global_defs.KEY.sharding.is_equivalent_to(get_replicated_sharding(), 0)


def test_default_seed_is_42():
    global_defs.KEY = None
    lazy = _key_data(qtx.get_subkeys())
    qtx.set_random_seed(42)
    np.testing.assert_array_equal(lazy, _key_data(qtx.get_subkeys()))


def test_default_dtype():
    qtx.set_default_dtype(jnp.complex64)
    assert qtx.get_default_dtype() == jnp.complex64
    assert qtx.get_real_dtype() == jnp.float32
    assert qtx.is_default_cpl()
    with pytest.raises(ValueError):
        qtx.set_default_dtype(jnp.int32)
    assert qtx.get_default_dtype() == jnp.complex64  # unchanged by the failed call
    qtx.set_default_dtype(jnp.float32)
    assert not qtx.is_default_cpl() and qtx.get_real_dtype() == jnp.float32


def test_get_sites_guards():
    with pytest.raises(RuntimeError, match="hasn't been defined"):
        qtx.get_sites()
    sites = Sites(4)  # a general Sites is not a Lattice
    assert qtx.get_sites() is sites
    with pytest.raises(RuntimeError, match="Lattice"):
        qtx.get_lattice()


def test_get_lattice():
    lattice = Chain(4)
    assert qtx.get_lattice() is lattice and qtx.get_sites() is lattice
