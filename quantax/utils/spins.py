from typing import Optional, Callable, Union
import numpy as np
from numpy.typing import ArrayLike
import jax
import jax.numpy as jnp
from quspin.tools import misc
from ..global_defs import get_sites, get_lattice, get_subkeys


def ints_to_array(basis_ints: ArrayLike, N: Optional[int] = None) -> np.ndarray:
    """Converts quspin basis integers to int8 state array"""
    if N is None:
        N = get_sites().nsites
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


def rand_spins(ns: Optional[int] = None, total_sz: Optional[int] = None) -> jax.Array:
    nsites = get_sites().nsites
    shape = (1, nsites) if ns is None else (ns, nsites)
    key = get_subkeys()
    if total_sz is None:
        spins = jax.random.randint(key, shape, 0, 2, jnp.int8)
        spins = spins * 2 - 1
    else:
        spins_down = (nsites - total_sz) // 2
        spins = jnp.ones(shape, jnp.int8)
        spins = spins.at[:, :spins_down].set(-1)
        spins = jax.random.permutation(key, spins, axis=1, independent=True)
    if ns is None:
        spins = spins[0]
    return spins
