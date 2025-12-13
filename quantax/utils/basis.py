from typing import Optional, Callable, Union
from jaxtyping import Key
from jax.sharding import Sharding
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from .sharding import get_replicate_sharding, get_distribute_sharding
from ..global_defs import PARTICLE_TYPE, get_sites, get_lattice, get_subkeys


_Array = Union[np.ndarray, jax.Array]


def ints_to_array(basis_ints: _Array, Nmodes: Optional[int] = None) -> np.ndarray:
    """
    Converts QuSpin basis integers to int8 state array.
    The similar function in QuSpin is
    `int_to_state <https://quspin.github.io/QuSpin/generated/quspin.basis.spin_basis_general.html#quspin.basis.spin_basis_general.int_to_state>`_.

    :param basis_ints:
        The basis integers in QuSpin.

    :param Nmodes:
        The number of modes. If not specified, use the value from
        `~quantax.global_defs.get_sites`.

    :return:
        The int8 state array with values being -1 and 1.
    """
    from quspin.tools import misc

    if Nmodes is None:
        Nmodes = get_sites().Nmodes
    state_array = misc.ints_to_array(basis_ints, Nmodes)
    state_array = state_array.astype(np.int8) * 2 - 1
    return state_array


def array_to_ints(state_array: _Array) -> np.ndarray:
    """
    Converts state array to QuSpin basis integers.
    The similar function in QuSpin is
    `index <https://quspin.github.io/QuSpin/generated/quspin.basis.spin_basis_general.html#quspin.basis.spin_basis_general.index>`_.

    :param state_array:
        The state array with values being -1 and 1.

    :return:
        The basis integers in QuSpin.
    """
    from quspin.tools import misc

    state_array = np.asarray(state_array)
    state_array = np.where(state_array > 0, state_array, 0)
    basis_ints = misc.array_to_ints(state_array)
    return basis_ints.flatten()


def neel(bipartiteA: bool = True) -> jax.Array:
    """
    Return a single neel state with alternating spins.

    :param bipartiteA:
        Whether the spin at (0, 0) is up (+1).
    """
    lattice = get_lattice()
    xyz = lattice.xyz_from_index
    spin_down = np.sum(xyz, axis=1) % 2 == 1
    spins = np.ones((lattice.Nsites,), dtype=np.int8)
    spins[spin_down] = -1
    if not bipartiteA:
        spins = -spins
    spins = jnp.asarray(spins)
    return spins


def stripe(alternate_dim: int = 1) -> jax.Array:
    """
    Return a single stripe state.

    :param alternate_dim:
        The dimension along which the spins alternate.
    """
    lattice = get_lattice()
    xyz = lattice.xyz_from_index
    spin_down = xyz[:, alternate_dim + 1] % 2 == 1
    spins = np.ones((lattice.Nsites,), dtype=np.int8)
    spins[spin_down] = -1
    spins = jnp.asarray(spins)
    return spins


def Sqz_factor(*q: float) -> Callable:
    r"""
    Spin structure factor :math:`\left< \frac{1}{2 \sqrt{N}} S^z_r S^z_0 e^{-iqr} \right>`

    :param q:
        Momentum of structure factor

    :return:
        A function f that gives the structure factor of spin configuration s by calling f(s)
    """
    sites = get_sites()
    qr = np.einsum("i,ni->n", q, sites.coord)
    e_iqr = np.exp(-1j * qr)
    if np.allclose(e_iqr.imag, 0.0):
        e_iqr = e_iqr.real

    factor = 1 / (2 * np.sqrt(sites.Nsites)) * e_iqr
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
    """
    Random basis states. The method for generating random states is automatically adjusted
    for different particle types.

    :param ns:
        The number of basis states.
        If not specified, only 1 basis state without batch dimension will be returned.
    """
    nsamples = 1 if ns is None else ns
    if nsamples % jax.device_count() == 0:
        sharding = get_distribute_sharding()
    else:
        sharding = get_replicate_sharding()

    sites = get_sites()
    Nparticles = sites.Nparticles
    shape = (nsamples, sites.Nmodes)
    key = get_subkeys()

    if sites.particle_type == PARTICLE_TYPE.spin:
        if isinstance(Nparticles, int):
            s = _rand_states(key, shape, sharding)
        else:
            Nup = Nparticles[0]
            s = _rand_Nconserved(key, shape, Nup, sharding)
    elif sites.particle_type == PARTICLE_TYPE.spinful_fermion:
        if Nparticles is None:
            if sites.double_occ:
                s = _rand_states(key, shape, sharding)
            else:
                s = _rand_single_occ(key, shape, sharding)
        elif isinstance(Nparticles, int):
            if sites.double_occ:
                s = _rand_Nconserved(key, shape, Nparticles, sharding)
            else:
                raise NotImplementedError(
                    "Single occupancy with conserved particle number not implemented."
                )
        else:
            Nup, Ndown = Nparticles
            shape = (nsamples, sites.Nsites)
            if sites.double_occ:
                key_up, key_down = jr.split(key, 2)
                s_up = _rand_Nconserved(key_up, shape, Nup, sharding)
                s_down = _rand_Nconserved(key_down, shape, Ndown, sharding)
                s = jnp.concatenate([s_up, s_down], axis=1)
            else:
                s = _rand_Nconserved_single_occ(key, shape, Nup, Ndown, sharding)
    elif sites.particle_type == PARTICLE_TYPE.spinless_fermion:
        if Nparticles is None:
            s = _rand_states(key, shape, sharding)
        else:
            s = _rand_Nconserved(key, shape, Nparticles, sharding)
    else:
        raise NotImplementedError

    if ns is None:
        s = s[0]
    return s
