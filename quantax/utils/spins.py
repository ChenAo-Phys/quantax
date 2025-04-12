from typing import Optional, Callable, Union
from jaxtyping import Key
from jaxlib.xla_extension import Sharding
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from quspin.tools import misc
from .sharding import get_replicate_sharding, get_global_sharding
from ..global_defs import get_sites, get_lattice, get_subkeys


_Array = Union[np.ndarray, jax.Array]


def ints_to_array(basis_ints: _Array, N: Optional[int] = None) -> np.ndarray:
    """Converts QuSpin basis integers to int8 state array"""
    if N is None:
        N = get_sites().nstates
    state_array = misc.ints_to_array(basis_ints, N)
    state_array = state_array.astype(np.int8) * 2 - 1
    return state_array


def array_to_ints(state_array: _Array) -> np.ndarray:
    """Converts int8 state array to QuSpin basis integers"""
    state_array = np.asarray(state_array)
    state_array = np.where(state_array > 0, state_array, 0)
    basis_ints = misc.array_to_ints(state_array)
    return basis_ints.flatten()


def neel(bipartiteA: bool = True) -> jax.Array:
    lattice = get_lattice()
    xyz = lattice.xyz_from_index
    spin_down = np.sum(xyz, axis=1) % 2 == 1
    spins = np.ones((lattice.N,), dtype=np.int8)
    spins[spin_down] = -1
    if not bipartiteA:
        spins = -spins
    spins = jnp.asarray(spins)
    return spins


def stripe(alternate_dim: int = 1) -> jax.Array:
    lattice = get_lattice()
    xyz = lattice.xyz_from_index
    spin_down = xyz[:, alternate_dim + 1] % 2 == 1
    spins = np.ones((lattice.N,), dtype=np.int8)
    spins[spin_down] = -1
    spins = jnp.asarray(spins)
    return spins


def Sqz_factor(*q: float) -> Callable:
    sites = get_sites()
    qr = np.einsum("i,ni->n", q, sites.coord)
    e_iqr = np.exp(-1j * qr)
    if np.allclose(e_iqr.imag, 0.0):
        e_iqr = e_iqr.real

    factor = 1 / (2 * np.sqrt(sites.N)) * e_iqr
    factor = jnp.asarray(factor)

    @jax.jit
    def evaluate(spin: jax.Array) -> jax.Array:
        return jnp.dot(factor, spin)

    return evaluate


@partial(jax.jit, static_argnums=(1, 2))
def _rand_states(key: Key, shape: tuple, sharding: Sharding) -> jax.Array:
    s = jr.randint(key, shape, 0, 2, jnp.int8)
    s = s * 2 - 1
    return jax.lax.with_sharding_constraint(s, sharding)


@partial(jax.jit, static_argnums=(1, 2, 3))
def _rand_Nconserved(key: Key, shape: tuple, Np: int, sharding: Sharding) -> jax.Array:
    s = -jnp.ones(shape, jnp.int8)
    s = s.at[:, :Np].set(1)
    s = jr.permutation(key, s, axis=1, independent=True)
    return jax.lax.with_sharding_constraint(s, sharding)


@partial(jax.jit, static_argnums=(1, 2))
def _rand_single_occ(key: Key, shape: tuple, sharding: Sharding) -> jax.Array:
    rand_int = jr.randint(key, shape, 0, 3)
    s_up = jnp.where(rand_int == 2, 1, -1)
    s_down = jnp.where(rand_int == 1, 1, -1)
    s = jnp.concatenate([s_up, s_down], axis=1, dtype=jnp.int8)
    return jax.lax.with_sharding_constraint(s, sharding)


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def _rand_Nconserved_single_occ(
    key: Key, shape: tuple, Nup: int, Ndown: int, sharding: Sharding
) -> jax.Array:
    s = jnp.zeros(shape, jnp.int8)
    s = s.at[:, :Nup].set(2)
    s = s.at[:, Nup : Nup + Ndown].set(1)

    s = jr.permutation(key, s, axis=1, independent=True)
    s_up = jnp.where(s == 2, 1, -1)
    s_down = jnp.where(s == 1, 1, -1)
    s = jnp.concatenate([s_up, s_down], axis=1, dtype=jnp.int8)
    return jax.lax.with_sharding_constraint(s, sharding)


def rand_states(ns: Optional[int] = None) -> jax.Array:
    nsamples = 1 if ns is None else ns
    if nsamples % jax.device_count() == 0:
        sharding = get_global_sharding()
    else:
        sharding = get_replicate_sharding()

    sites = get_sites()
    Nparticle = sites.Nparticle
    key = get_subkeys()

    if sites.is_fermion:
        # fermion system
        shape = (nsamples, sites.nstates)
        if Nparticle is None:
            if sites.double_occ:
                s = _rand_states(key, shape, sharding)
            else:
                s = _rand_single_occ(key, shape, sharding)
        elif isinstance(Nparticle, int):
            if sites.double_occ:
                s = _rand_Nconserved(key, shape, sharding)
            else:
                raise NotImplementedError(
                    "Single occupancy with conserved particle number not implemented."
                )
        else:
            Nup, Ndown = Nparticle
            shape = (nsamples, sites.N)
            if sites.double_occ:
                key_up, key_down = jr.split(key, 2)
                s_up = _rand_Nconserved(key_up, shape, Nup, sharding)
                s_down = _rand_Nconserved(key_down, shape, Ndown, sharding)
                s = jnp.concatenate([s_up, s_down], axis=1)
            else:
                s = _rand_Nconserved_single_occ(key, shape, Nup, Ndown, sharding)
    else:
        # spin system
        shape = (nsamples, sites.N)
        if isinstance(Nparticle, int):
            s = _rand_states(key, shape, sharding)
        else:
            Nup = Nparticle[0]
            s = _rand_Nconserved(key, shape, Nup, sharding)

    if ns is None:
        s = s[0]
    return s
