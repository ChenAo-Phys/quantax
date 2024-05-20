from typing import Optional
from jaxtyping import DTypeLike
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.sharding import PositionalSharding


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "float32")
jax.config.update('jax_threefry_partitionable', True)


DTYPE = jnp.float64
PARAMS_DTYPE = jnp.float32


def set_default_dtype(dtype: DTypeLike) -> None:
    if not (
        jnp.issubdtype(dtype, jnp.floating)
        or jnp.issubdtype(dtype, jnp.complexfloating)
    ):
        raise ValueError("'dtype' should be float or complex types")
    global DTYPE
    DTYPE = dtype


def set_params_dtype(dtype: DTypeLike) -> None:
    if not (
        jnp.issubdtype(dtype, jnp.floating)
        or jnp.issubdtype(dtype, jnp.complexfloating)
    ):
        raise ValueError("'dtype' should be float or complex types")
    global PARAMS_DTYPE
    PARAMS_DTYPE = dtype


def get_default_dtype() -> jnp.dtype:
    return DTYPE


def get_params_dtype() -> jnp.dtype:
    return PARAMS_DTYPE


def is_default_cpl() -> bool:
    return jnp.issubdtype(DTYPE, jnp.complexfloating)


def is_params_cpl() -> bool:
    return jnp.issubdtype(PARAMS_DTYPE, jnp.complexfloating)


def set_random_seed(seed: int) -> None:
    global KEY
    sharding = PositionalSharding(jax.local_devices()).replicate()
    KEY = jax.device_put(jr.key(seed), sharding)

set_random_seed(np.random.randint(0, 4294967296))


def get_subkeys(num: Optional[int] = None) -> jax.Array:
    global KEY
    KEY, new_keys = _gen_keys(KEY, num)
    return new_keys


@partial(jax.jit, static_argnums=1)
def _gen_keys(key, num: Optional[int] = None) -> jax.Array:
    nkeys = 2 if num is None else num + 1
    new_keys = jr.split(key, nkeys)
    key = new_keys[0]
    new_keys = new_keys[1] if num is None else new_keys[1:]
    return key, new_keys


from .sites import Sites, Lattice


def get_sites() -> Sites:
    sites = Sites._SITES
    if sites is None:
        raise RuntimeError("The `Sites` hasn't been defined.")
    return sites


def get_lattice() -> Lattice:
    sites = get_sites()
    if not isinstance(sites, Lattice):
        raise RuntimeError("Require a `Lattice`, but got a general `Sites`")
    return sites


"""
Todo:
- Lattice move axis
- Multiple hosts compatibility
"""
