from typing import Optional, Tuple
from functools import partial
from enum import Enum
import jax
import jax.numpy as jnp
import jax.random as jr


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "float32")


DTYPE = jnp.float64


def set_default_dtype(dtype: jnp.dtype) -> None:
    """
    Set the default data type in Quantax.
    Recommended to be ``jnp.float64`` or ``jnp.complex128``. Default to ``jnp.float64``.

    .. note::
        This doesn't alter the computation inside ``quantax.model``.
    """
    if not (
        jnp.issubdtype(dtype, jnp.floating)
        or jnp.issubdtype(dtype, jnp.complexfloating)
    ):
        raise ValueError("'dtype' should be float or complex types")
    global DTYPE
    DTYPE = dtype


def get_default_dtype() -> jnp.dtype:
    """Return the default data type in Quantax."""
    return DTYPE


def get_real_dtype() -> jnp.dtype:
    """
    Return the default real data type in Quantax.
    If the default data type is complex, then return the corresponding real data type.
    """
    return jnp.finfo(DTYPE).dtype


def is_default_cpl() -> bool:
    """Return whether the default data type is complex."""
    return jnp.issubdtype(DTYPE, jnp.complexfloating)


KEY = None


def set_random_seed(seed: int) -> None:
    """
    Set the initial random seed in Quantax. Default to be 42.
    """
    global KEY
    KEY = jr.key(seed)

    from .utils import to_replicate_array

    KEY = to_replicate_array(KEY)


@partial(jax.jit, static_argnums=1)
def _gen_keys(key, num: Optional[int] = None) -> Tuple[jax.Array, jax.Array]:
    nkeys = 2 if num is None else num + 1
    new_keys = jr.split(key, nkeys)
    key = new_keys[0]
    new_keys = new_keys[1] if num is None else new_keys[1:]
    return key, new_keys


def get_subkeys(num: Optional[int] = None) -> jax.Array:
    """
    Get jax keys stored in Quantax. The keys are replicated across all devices.

    :param num:
        The number of returned keys.
        If ``num`` is not given, then return only 1 key instead of an array of keys.

    .. warning::
        This function is not jittable, because it reads and writes the global key
        stored in quantax.
    """
    global KEY
    if KEY is None:
        set_random_seed(42)
    KEY, new_keys = _gen_keys(KEY, num)
    return new_keys


class PARTICLE_TYPE(Enum):
    r"""
    The enums to distinguish different particle types.

    - 0: spin

    - 1: spinful_fermion

    - 2: spinless_fermion

    - (Not implemented) 3: boson
    """

    spin = 0
    spinful_fermion = 1
    spinless_fermion = 2


from .sites import Sites, Lattice


def get_sites() -> Sites:
    """
    Get the `~quantax.sites.Sites` used in the current program.

    .. warning::
        Unlike other NQS packages, in Quantax the geometry graph and the hilbert space
        is defined as a global constant which shouldn't be changed within a single program.
    """
    sites = Sites._SITES
    if sites is None:
        raise RuntimeError("The `Sites` hasn't been defined.")
    return sites


def get_lattice() -> Lattice:
    """
    Get the `~quantax.sites.Lattice` used in the current program. This is similar to `get_sites`,
    but will raise an error if the defined `~quantax.sites.Sites` is not a `~quantax.sites.Lattice`.
    """
    sites = get_sites()
    if not isinstance(sites, Lattice):
        raise RuntimeError("Require a `Lattice`, but got a general `Sites`")
    return sites
