from typing import Optional, Callable, Union, Sequence
from jaxlib.xla_extension import Sharding
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from quspin.tools import misc
from .sharding import get_local_sharding, get_global_sharding
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
    spins = np.ones((lattice.nsites,), dtype=np.int8)
    spins[spin_down] = -1
    if not bipartiteA:
        spins = -spins
    spins = jnp.asarray(spins)
    return spins


def stripe(alternate_dim: int = 1) -> jax.Array:
    lattice = get_lattice()
    xyz = lattice.xyz_from_index
    spin_down = xyz[:, alternate_dim + 1] % 2 == 1
    spins = np.ones((lattice.nsites,), dtype=np.int8)
    spins[spin_down] = -1
    spins = jnp.asarray(spins)
    return spins


def Sqz_factor(*q: float) -> Callable:
    sites = get_sites()
    qr = np.einsum("i,ni->n", q, sites.coord)
    e_iqr = np.exp(-1j * qr)
    if np.allclose(e_iqr.imag, 0.0):
        e_iqr = e_iqr.real

    factor = 1 / (2 * np.sqrt(sites.nsites)) * e_iqr
    factor = jnp.asarray(factor)

    @jax.jit
    def evaluate(spin: jax.Array) -> jax.Array:
        return jnp.dot(factor, spin)

    return evaluate


@partial(jax.jit, static_argnums=(1,2))
def _rand_states(key: jax.Array, shape: tuple, sharding: Sharding):
    fock_states = jr.randint(key, shape, 0, 2, jnp.int8)
    fock_states = fock_states * 2 - 1
    return jax.lax.with_sharding_constraint(fock_states, sharding)


@partial(jax.jit, static_argnums=(1,2,3))
def _rand_Nconserved_states(key: jax.Array, shape: tuple, Np: int, sharding: Sharding):
    fock_states = -jnp.ones(shape, jnp.int8)
    fock_states = fock_states.at[:, :Np].set(1)
    fock_states = jr.permutation(key, fock_states, axis=1, independent=True)
    return jax.lax.with_sharding_constraint(fock_states, sharding)


def rand_states(
    ns: Optional[int] = None,
    Nparticle: Optional[Union[int, Sequence]] = None,
    distributed: bool = False,
) -> jax.Array:
    if distributed:
        if ns is None or ns % jax.device_count() != 0:
            raise ValueError(f"{ns} samples can't be distributed.")
        sharding = get_global_sharding()
    else:
        sharding = get_local_sharding()

    sites = get_sites()
    if Nparticle is None:
        shape = (1, sites.nstates) if ns is None else (ns, sites.nstates)
        fock_states = _rand_states(get_subkeys(), shape, sharding)
    else:
        shape = (1, sites.nsites) if ns is None else (ns, sites.nsites)
        if sites.is_fermion:
            if not isinstance(Nparticle[0], int):
                if len(Nparticle) == 1:
                    Nparticle = Nparticle[0]
                else:
                    raise NotImplementedError(
                        "`rand_states` with multiple Nparticle sectors is not implemented"
                    )
            Nup, Ndown = Nparticle
            s_up = _rand_Nconserved_states(get_subkeys(), shape, Nup, sharding)
            s_down = _rand_Nconserved_states(get_subkeys(), shape, Ndown, sharding)
            fock_states = jnp.concatenate([s_up, s_down], axis=1)
        else:
            if not isinstance(Nparticle, int):
                if len(Nparticle) == 1:
                    Nparticle = Nparticle[0]
                else:
                    raise NotImplementedError(
                        "`rand_states` with multiple Nparticle sectors is not implemented"
                    )
            fock_states = _rand_Nconserved_states(
                get_subkeys(), shape, Nparticle, sharding
            )

    if ns is None:
        fock_states = fock_states[0]
    return fock_states
