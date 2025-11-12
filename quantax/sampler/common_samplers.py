from typing import Tuple, Optional, Union, Sequence
from jaxtyping import Key
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from .metropolis import Metropolis
from ..state import State
from ..utils import get_replicate_sharding
from ..global_defs import PARTICLE_TYPE, get_sites


class LocalFlip(Metropolis):
    """
    Generate Monte Carlo samples by locally flipping spins. This sampler is suitable for
    spin systems with unconserved spin-up and spin-down numbers.
    """

    @property
    def particle_type(self) -> Tuple[PARTICLE_TYPE, ...]:
        return (PARTICLE_TYPE.spin,)

    @property
    def nflips(self) -> int:
        return 1

    @partial(jax.jit, static_argnums=0)
    def propose(self, key: Key, old_spins: jax.Array) -> jax.Array:
        nsamples, N = old_spins.shape
        pos = jr.choice(key, N, (nsamples,))
        new_spins = old_spins.at[jnp.arange(nsamples), pos].multiply(-1)
        return new_spins


def _get_site_neighbors(n_neighbor: Union[int, Sequence[int]]) -> jax.Array:
    """
    Get the neighboring sites for each site.
    """
    sites = get_sites()
    n_neighbor = [n_neighbor] if isinstance(n_neighbor, int) else n_neighbor
    neighbors = sites.get_neighbor(n_neighbor)
    neighbors = np.concatenate(neighbors, axis=0)
    neighbor_matrix = np.zeros((sites.Nsites, sites.Nsites), dtype=np.bool_)
    neighbor_matrix[neighbors[:, 0], neighbors[:, 1]] = True
    neighbor_matrix = neighbor_matrix | neighbor_matrix.T
    max_neighbors = np.max(np.sum(neighbor_matrix, axis=1)).item()
    neighbor_matrix = jnp.asarray(neighbor_matrix, dtype=jnp.bool_)
    fn = jax.vmap(lambda x: jnp.flatnonzero(x, size=max_neighbors, fill_value=-1))
    neighbors = fn(neighbor_matrix)
    neighbors = jnp.asarray(neighbors, dtype=jnp.int32, device=get_replicate_sharding())
    if sites.particle_type == PARTICLE_TYPE.spinful_fermion:
        neighbors_dn = jnp.where(neighbors == -1, -1, neighbors + sites.Nsites)
        neighbors = jnp.concatenate([neighbors, neighbors_dn], axis=0)
    return neighbors


def _propose_exchange(
    key: Key,
    old_spins: jax.Array,
    hopping_particle: jax.Array,
    neighbors: jax.Array,
) -> jax.Array:
    nsamples, Nmodes = old_spins.shape
    keys = jr.split(key, 2 * nsamples)

    p_site = old_spins == hopping_particle
    choice_vmap = jax.vmap(lambda key, p: jr.choice(key, Nmodes, p=p))
    particle_idx = choice_vmap(keys[:nsamples], p_site)

    neighbors = neighbors[particle_idx]
    choice_vmap = jax.vmap(lambda key, neighbor: jr.choice(key, neighbor))
    neighbor_idx = choice_vmap(keys[nsamples:], neighbors)
    neighbor_idx = jnp.where(neighbor_idx == -1, particle_idx, neighbor_idx)

    arange = jnp.arange(nsamples)
    particle = old_spins[arange, particle_idx]
    neighbor = old_spins[arange, neighbor_idx]
    new_spins = old_spins
    new_spins = new_spins.at[arange, particle_idx].set(neighbor)
    new_spins = new_spins.at[arange, neighbor_idx].set(particle)
    return new_spins


class SpinExchange(Metropolis):
    """
    Generate Monte Carlo samples by exchanging neighbor spins in spin systems.
    This sampler only works when the system has fixed number of spin-up and spin-down particles.
    """

    def __init__(
        self,
        state: State,
        nsamples: int,
        reweight: float = 2.0,
        thermal_steps: Optional[int] = None,
        sweep_steps: Optional[int] = None,
        initial_spins: Optional[jax.Array] = None,
        n_neighbor: Union[int, Sequence[int]] = 1,
    ):
        r"""
        :param state:
            The state used for computing the wave function and probability.
            Since exchanging neighbor spins doesn't change the total Sz,
            the state must have `quantax.symmetry.ParticleConserve` symmetry to specify
            the symmetry sector.

        :param nsamples:
            Number of samples generated per iteration.
            It should be a multiple of the total number of machines to allow samples
            to be equally distributed on different machines.

        :param reweight:
            The reweight factor n defining the sample probability :math:`|\psi|^n`,
            default to 2.0.

        :param thermal_steps:
            The number of thermalization steps in the beginning of each Markov chain,
            default to be 20 * fock state length.

        :param sweep_steps:
            The number of steps for generating new samples, default to be 2 * fock state length.

        :param initial_spins:
            The initial spins for every Markov chain before the thermalization steps,
            default to be random spins.

        :param n_neighbor:
            The neighbors to be considered in exchanges, default to nearest neighbors.
        """
        sites = get_sites()
        if isinstance(sites.Nparticles, int):
            raise ValueError(
                "The number spin-up and spin-down particles should be specified in "
                "sites for `SpinExchange` sampler."
            )

        Nup = sites.Nparticles[0]
        if 2 * Nup <= state.Nmodes:
            self._hopping_particle = 1
        else:
            self._hopping_particle = -1

        self._neighbors = _get_site_neighbors(n_neighbor)

        super().__init__(
            state, nsamples, reweight, thermal_steps, sweep_steps, initial_spins
        )

    @property
    def particle_type(self) -> Tuple[PARTICLE_TYPE, ...]:
        return (PARTICLE_TYPE.spin,)

    @property
    def nflips(self) -> int:
        return 2

    @partial(jax.jit, static_argnums=0)
    def propose(self, key: Key, old_spins: jax.Array) -> jax.Array:
        return _propose_exchange(
            key, old_spins, self._hopping_particle, self._neighbors
        )


class ParticleHop(Metropolis):
    """
    Generate Monte Carlo samples by hopping random fermions to neighbor sites.
    This sampler only works when the system has fixed number of fermions.
    """

    def __init__(
        self,
        state: State,
        nsamples: int,
        reweight: float = 2.0,
        thermal_steps: Optional[int] = None,
        sweep_steps: Optional[int] = None,
        initial_spins: Optional[jax.Array] = None,
        n_neighbor: Union[int, Sequence[int]] = 1,
    ):
        r"""
        :param state:
            The state used for computing the wave function and probability.
            Since exchanging neighbor spins doesn't change the total Sz,
            the state must have `quantax.symmetry.ParticleConserve` symmetry to specify
            the symmetry sector.

        :param nsamples:
            Number of samples generated per iteration.
            It should be a multiple of the total number of machines to allow samples
            to be equally distributed on different machines.

        :param reweight:
            The reweight factor n defining the sample probability :math:`|\psi|^n`,
            default to 2.0.

        :param thermal_steps:
            The number of thermalization steps in the beginning of each Markov chain,
            default to be 20 * fock state length.

        :param sweep_steps:
            The number of steps for generating new samples,
            default to be 2 * fock state length.

        :param initial_spins:
            The initial spins for every Markov chain before the thermalization steps,
            default to be random spins.

        :param n_neighbor:
            The neighbors to be considered by particle hoppings, default to nearest neighbors.
        """
        sites = get_sites()
        if sites.Nparticles is None:
            raise ValueError(
                "The number of fermions should be specified in sites for `ParticleHop` sampler."
            )

        if 2 * sites.Ntotal <= state.Nmodes:
            self._hopping_particle = 1
        else:
            self._hopping_particle = -1

        self._neighbors = _get_site_neighbors(n_neighbor)

        super().__init__(
            state, nsamples, reweight, thermal_steps, sweep_steps, initial_spins
        )

    @property
    def particle_type(self) -> Tuple[PARTICLE_TYPE, ...]:
        return (PARTICLE_TYPE.spinful_fermion, PARTICLE_TYPE.spinless_fermion)

    @property
    def nflips(self) -> int:
        return 2

    @partial(jax.jit, static_argnums=0)
    def propose(self, key: Key, old_spins: jax.Array) -> jax.Array:
        return _propose_exchange(
            key, old_spins, self._hopping_particle, self._neighbors
        )


class SiteExchange(Metropolis):
    """
    Generate Monte Carlo samples by exchanging the spinful fermions on neighbor sites.

    .. warning::

        This sampler conserves the number of doublons and holons.
    """

    def __init__(
        self,
        state: State,
        nsamples: int,
        reweight: float = 2.0,
        thermal_steps: Optional[int] = None,
        sweep_steps: Optional[int] = None,
        initial_spins: Optional[jax.Array] = None,
        n_neighbor: Union[int, Sequence[int]] = 1,
    ):
        r"""
        :param state:
            The state used for computing the wave function and probability.
            Since exchanging neighbor spins doesn't change the total Sz,
            the state must have `quantax.symmetry.ParticleConserve` symmetry to specify
            the symmetry sector.

        :param nsamples:
            Number of samples generated per iteration.
            It should be a multiple of the total number of machines to allow samples
            to be equally distributed on different machines.

        :param reweight:
            The reweight factor n defining the sample probability :math:`|\psi|^n`,
            default to 2.0.

        :param thermal_steps:
            The number of thermalization steps in the beginning of each Markov chain,
            default to be 20 * fock state length.

        :param sweep_steps:
            The number of steps for generating new samples, default to be 2 * fock state length.

        :param initial_spins:
            The initial spins for every Markov chain before the thermalization steps,
            default to be random spins.

        :param n_neighbor:
            The neighbors to be considered by exchanges, default to nearest neighbors.
        """
        sites = get_sites()
        if sites.Nparticles is None:
            raise ValueError("`Nparticle` should be specified for `SiteExchange`.")

        n_neighbor = [n_neighbor] if isinstance(n_neighbor, int) else n_neighbor
        neighbors = sites.get_neighbor(n_neighbor)
        neighbors = np.concatenate(neighbors, axis=0)
        self._neighbors = jnp.asarray(neighbors, dtype=jnp.uint16)

        super().__init__(
            state, nsamples, reweight, thermal_steps, sweep_steps, initial_spins
        )

    @property
    def particle_type(self) -> Tuple[PARTICLE_TYPE, ...]:
        return (PARTICLE_TYPE.spinful_fermion,)

    @property
    def nflips(self) -> int:
        return 4

    @partial(jax.jit, static_argnums=0)
    def propose(self, key: Key, old_spins: jax.Array) -> jax.Array:
        nsamples = old_spins.shape[0]
        n_neighbors = self._neighbors.shape[0]
        pos = jr.choice(key, n_neighbors, (nsamples,))
        pairs = self._neighbors[pos]

        N = get_sites().Nsites
        arange = jnp.arange(nsamples)
        arange = jnp.tile(arange, (2, 1)).T
        s_exchange_up = old_spins[arange, pairs[:, ::-1]]
        new_spins = old_spins.at[arange, pairs].set(s_exchange_up)
        s_exchange_dn = old_spins[arange, pairs[:, ::-1] + N]
        new_spins = new_spins.at[arange, pairs + N].set(s_exchange_dn)
        return new_spins


class SiteFlip(Metropolis):
    """
    Generate Monte Carlo samples by flipping spins of spinful fermions locally.

    .. warning::

        This sampler conserves the number of fermions on each site.
    """

    @property
    def particle_type(self) -> Tuple[PARTICLE_TYPE, ...]:
        return (PARTICLE_TYPE.spinful_fermion,)

    @property
    def nflips(self) -> int:
        return 2

    @partial(jax.jit, static_argnums=0)
    def propose(self, key: Key, old_spins: jax.Array) -> jax.Array:
        nsamples, Nmodes = old_spins.shape
        N = Nmodes // 2
        pos = jr.choice(key, N, (nsamples,))
        arange = jnp.arange(nsamples)
        s_up = old_spins[arange, pos]
        s_dn = old_spins[arange, pos + N]
        new_spins = old_spins.at[arange, pos].set(s_dn).at[arange, pos + N].set(s_up)
        return new_spins
