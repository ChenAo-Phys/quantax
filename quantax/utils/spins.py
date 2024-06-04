from typing import Optional, Callable, Union, Sequence
from numpy.typing import ArrayLike
import numpy as np
import jax
import jax.numpy as jnp
from quspin.tools import misc
from ..global_defs import get_sites, get_lattice, get_subkeys


def ints_to_array(basis_ints: ArrayLike, N: Optional[int] = None) -> np.ndarray:
    """Converts quspin basis integers to int8 state array"""
    if N is None:
        N = get_sites().nstates
    state_array = misc.ints_to_array(basis_ints, N)
    state_array = state_array.astype(np.int8) * 2 - 1
    return state_array


def array_to_ints(state_array: Union[np.ndarray, jax.Array]) -> np.ndarray:
    """Converts int8 state array to quspin basis integers"""
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


def _rand_Nconserved_states(shape: tuple, Np: int) -> jax.Array:
    fock_states = -jnp.ones(shape, jnp.int8)
    fock_states = fock_states.at[:, :Np].set(1)
    key = get_subkeys()
    fock_states = jax.random.permutation(key, fock_states, axis=1, independent=True)
    return fock_states


def rand_states(
    ns: Optional[int] = None, Nparticle: Optional[Union[int, Sequence]] = None
) -> jax.Array:
    sites = get_sites()
    if Nparticle is None:
        shape = (1, sites.nstates) if ns is None else (ns, sites.nstates)
        fock_states = jax.random.randint(get_subkeys(), shape, 0, 2, jnp.int8)
        fock_states = fock_states * 2 - 1
    else:
        shape = (1, sites.nsites) if ns is None else (ns, sites.nsites)
        if sites.is_fermion:
            if not isinstance(Nparticle[0], int):
                if len(Nparticle) == 1:
                    Nparticle = Nparticle[0]
                else:
                    raise NotImplementedError(
                        "`rand_states` with multiple Nparticle sectors not implemented"
                    )
            Nup, Ndown = Nparticle
            s_up = _rand_Nconserved_states(shape, Nup)
            s_down = _rand_Nconserved_states(shape, Ndown)
            fock_states = jnp.concatenate([s_up, s_down], axis=1)
        else:
            if not isinstance(Nparticle, int):
                if len(Nparticle) == 1:
                    Nparticle = Nparticle[0]
                else:
                    raise NotImplementedError(
                        "`rand_states` with multiple Nparticle sectors not implemented"
                    )
            fock_states = _rand_Nconserved_states(shape, Nparticle)

    if ns is None:
        fock_states = fock_states[0]
    return fock_states
