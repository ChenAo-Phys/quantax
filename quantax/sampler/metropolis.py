from typing import Optional, Tuple, Union, Sequence
from jaxtyping import Key, PyTree
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from .sampler import Sampler
from .status import SamplerStatus, Samples
from ..state import State, Variational
from ..global_defs import get_subkeys, get_sites, get_default_dtype
from ..utils import (
    to_global_array,
    to_replicate_array,
    get_global_sharding,
    rand_states,
    filter_tree_map,
    chunk_map,
)


class Metropolis(Sampler):
    """
    Abstract class for metropolis samplers.
    The samples are equally distributed on different machines.
    """

    def __init__(
        self,
        state: State,
        nsamples: int,
        reweight: float = 2.0,
        thermal_steps: Optional[int] = None,
        sweep_steps: Optional[int] = None,
        initial_spins: Optional[jax.Array] = None,
    ):
        """
        :param state:
            The state used for computing the wave function and probability.

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
        """
        super().__init__(state, nsamples, reweight)
        self._reweight = to_replicate_array(reweight)

        if thermal_steps is None:
            self._thermal_steps = 20 * self.nstates
        else:
            self._thermal_steps = thermal_steps
        if sweep_steps is None:
            self._sweep_steps = 2 * self.nstates
        else:
            self._sweep_steps = sweep_steps

        if initial_spins is not None:
            if initial_spins.ndim == 1:
                initial_spins = jnp.tile(initial_spins, (self.nsamples, 1))
            else:
                initial_spins = initial_spins.reshape(self.nsamples, self.nstates)
            initial_spins = to_global_array(initial_spins.astype(jnp.int8))
        self._initial_spins = initial_spins

        self._is_fermion = get_sites().is_fermion
        self._double_occ = get_sites().double_occ

        self.reset()

    @property
    def nflips(self) -> Optional[int]:
        """
        The number of flips in new proposal.
        """
        return None

    def reset(self) -> None:
        """
        Reset all Markov chains to ``initial_spins`` and thermalize them
        """
        if self._initial_spins is None:
            spins = rand_states(self.nsamples)
        else:
            spins = self._initial_spins

        self._spins, self._propose_prob = self._propose(get_subkeys(), spins)
        self.sweep(self._thermal_steps)

    def sweep(self, nsweeps: Optional[int] = None) -> Samples:
        """
        Generate new samples

        :param nsweeps:
            Number of sweeps for generating the new samples, default to be
            ``self._sweep_steps``
        """
        if nsweeps is None:
            nsweeps = self._sweep_steps

        if hasattr(self._state, "ref_chunk"):
            chunk_size = self._state.ref_chunk
            fn_sweep = chunk_map(
                self._partial_sweep, in_axes=(None, 0, 0), chunk_size=chunk_size
            )
        else:
            fn_sweep = self._partial_sweep

        spins, wf, propose_prob = fn_sweep(nsweeps, self._spins, self._propose_prob)

        self._spins = spins
        self._propose_prob = propose_prob
        if isinstance(self._state, Variational):
            new_max = jnp.max(jnp.abs(wf))
            old_max = self._state._maximum
            self._state._maximum = jnp.where(new_max > old_max, new_max, old_max)
        return Samples(spins, wf, self._reweight)

    def _partial_sweep(
        self, nsweeps: int, spins: jax.Array, propose_prob: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        wf = self._state(spins)
        if self.nflips is None:
            state_internal = None
        else:
            state_internal = self._state.init_internal(spins)
        status = SamplerStatus(spins, wf, propose_prob, state_internal)

        keys_propose = to_replicate_array(get_subkeys(nsweeps))
        keys_update = to_replicate_array(get_subkeys(nsweeps))
        for keyp, keyu in zip(keys_propose, keys_update):
            status = self._single_sweep(keyp, keyu, status)

        spins = status.spins
        wf = status.wave_function
        propose_prob = status.propose_prob
        if status.state_internal is not None:
            del status
            wf = self._state(spins)

        return spins, wf, propose_prob

    def _single_sweep(
        self, keyp: Key, keyu: Key, status: SamplerStatus
    ) -> SamplerStatus:
        new_spins, new_propose_prob = self._propose(keyp, status.spins)
        if self.nflips is None:
            new_wf = self._state(new_spins)
            state_internal = None
        else:
            new_wf, state_internal = self._state.ref_forward_with_updates(
                new_spins, status.spins, self.nflips, status.state_internal
            )
        new_status = SamplerStatus(new_spins, new_wf, new_propose_prob, state_internal)
        status = self._update(keyu, status, new_status)
        return status

    def _propose(
        self, key: jax.Array, old_spins: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Propose new spin configurations, return spin and proposal weight

        :return:
            spins:
                The proposed spin configurations

            propose_prob:
                The probability of the proposal
        """

    @staticmethod
    @eqx.filter_jit
    @eqx.filter_vmap
    def _update_selected(
        is_selected: jax.Array, new_tree: PyTree, old_tree: PyTree
    ) -> PyTree:
        fn = lambda new, old: jnp.where(is_selected, new, old)
        return filter_tree_map(fn, new_tree, old_tree)

    @partial(jax.jit, static_argnums=0, donate_argnums=3)
    def _update(
        self, key: jax.Array, old_status: SamplerStatus, new_status: SamplerStatus
    ) -> SamplerStatus:
        nsamples, nstates = old_status.spins.shape
        old_prob = jnp.abs(old_status.wave_function) ** self._reweight
        new_prob = jnp.abs(new_status.wave_function) ** self._reweight
        rand = 1.0 - jr.uniform(key, (nsamples,), old_prob.dtype)
        rate_accept = new_prob * old_status.propose_prob
        rate_reject = old_prob * new_status.propose_prob * rand

        if self._is_fermion and not self._double_occ:
            s = new_status.spins.reshape(nsamples, 2, nstates // 2)
            occ_allowed = jnp.all(jnp.any(s <= 0, axis=1), axis=1)
        else:
            occ_allowed = True

        accepted = (rate_accept > rate_reject) | (old_prob == 0.0)
        updated = jnp.any(old_status.spins != new_status.spins, axis=1)

        cond = accepted & updated & occ_allowed
        return self._update_selected(cond, new_status, old_status)


class LocalFlip(Metropolis):
    """
    Generate Monte Carlo samples by locally flipping spins.
    """

    @property
    def nflips(self) -> int:
        return 1

    @partial(jax.jit, static_argnums=0)
    def _propose(
        self, key: jax.Array, old_spins: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        nsamples, N = old_spins.shape
        pos = jr.choice(key, N, (nsamples,))
        new_spins = old_spins.at[jnp.arange(nsamples), pos].multiply(-1)
        propose_prob = jnp.ones(
            nsamples, dtype=get_default_dtype(), device=get_global_sharding()
        )
        return new_spins, propose_prob


class NeighborExchange(Metropolis):
    """
    Generate Monte Carlo samples by exchanging neighbor spins or fermions.
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
        """
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
        if sites.Nparticle is None:
            raise ValueError("`Nparticle` should be specified for NeighborExchange.")

        n_neighbor = [n_neighbor] if isinstance(n_neighbor, int) else n_neighbor
        neighbors = sites.get_neighbor(n_neighbor)
        neighbors = np.concatenate(neighbors, axis=0)
        if sites.is_fermion:
            neighbors = np.concatenate([neighbors, neighbors + sites.N], axis=0)
        self._neighbors = jnp.asarray(neighbors, dtype=jnp.uint16)

        super().__init__(
            state, nsamples, reweight, thermal_steps, sweep_steps, initial_spins
        )

    @property
    def nflips(self) -> int:
        return 2

    @partial(jax.jit, static_argnums=0)
    def _propose(
        self, key: jax.Array, old_spins: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        nsamples = old_spins.shape[0]
        pos = jr.choice(key, self._neighbors.shape[0], (nsamples,))
        pairs = self._neighbors[pos]

        arange = jnp.arange(nsamples)
        arange = jnp.tile(arange, (2, 1)).T
        s_exchange = old_spins[arange, pairs[:, ::-1]]
        new_spins = old_spins.at[arange, pairs].set(s_exchange)
        propose_prob = jnp.ones(
            nsamples, dtype=get_default_dtype(), device=get_global_sharding()
        )
        return new_spins, propose_prob


class ParticleHop(Metropolis):
    """
    Generate Monte Carlo samples by hopping random particles to neighbor sites.
    The sampler will automatically choose to hop particles or holes, in the spin case
    spin up or down.

    .. warning::

        This sampler only works if all sites have the same amount of neighbors.
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
        """
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
        if sites.Nparticle is None:
            raise ValueError("`Nparticle` should be specified for ParticleHop.")

        Nup = sites.Ntotal if sites.is_fermion else sites.Nparticle[0]
        if 2 * Nup <= state.nstates:
            self._hopping_particle = 1
            self._nhopping = Nup
        else:
            self._hopping_particle = -1
            self._nhopping = state.nstates - Nup

        n_neighbor = [n_neighbor] if isinstance(n_neighbor, int) else n_neighbor
        neighbors = sites.get_neighbor(n_neighbor)
        neighbors = np.concatenate(neighbors, axis=0)

        neighbor_matrix = np.zeros((state.N, state.N), dtype=np.bool_)
        neighbor_matrix[neighbors[:, 0], neighbors[:, 1]] = True
        neighbor_matrix = neighbor_matrix | neighbor_matrix.T
        neighbor_count = np.sum(neighbor_matrix, axis=1)
        if not np.all(neighbor_count == neighbor_count[0]):
            raise RuntimeError("Different sites have different amount of neighbors.")

        neighbor_idx = np.nonzero(neighbor_matrix)[1].reshape(sites.N, -1)
        if sites.is_fermion:
            neighbor_idx = np.concatenate(
                [neighbor_idx, neighbor_idx + sites.N], axis=0
            )
        self._neighbor_idx = jnp.asarray(neighbor_idx, dtype=jnp.uint16)

        super().__init__(
            state, nsamples, reweight, thermal_steps, sweep_steps, initial_spins
        )

    @property
    def nflips(self) -> int:
        return 2

    @partial(jax.jit, static_argnums=0)
    def _propose(
        self, key: jax.Array, old_spins: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        nsamples, nstates = old_spins.shape
        key = jr.split(key, nsamples + 1)

        p_hop = old_spins == self._hopping_particle
        choice_vmap = jax.vmap(lambda key, p: jr.choice(key, nstates, p=p))
        hopping_particles = choice_vmap(key[:-1], p_hop)

        neighbors = self._neighbor_idx[hopping_particles]
        neighbor_idx = jr.choice(key[-1], neighbors.shape[1], (nsamples,))
        arange = jnp.arange(nsamples)
        neighbors = neighbors[arange, neighbor_idx]

        pairs = jnp.stack([hopping_particles, neighbors], axis=1)
        arange = jnp.tile(arange, (2, 1)).T
        s_exchange = old_spins[arange, pairs[:, ::-1]]
        new_spins = old_spins.at[arange, pairs].set(s_exchange)
        propose_prob = jnp.ones(
            nsamples, dtype=get_default_dtype(), device=get_global_sharding()
        )
        return new_spins, propose_prob

class HopExchangeMix(Metropolis):
    """
    Generate Monte Carlo samples by exchanging neighbor spins or fermions.
    """

    def __init__(
        self,
        state: State,
        nsamples: int,
        ratio: float,
        reweight: float = 2.0,
        thermal_steps: Optional[int] = None,
        sweep_steps: Optional[int] = None,
        initial_spins: Optional[jax.Array] = None,
        n_neighbor: Union[int, Sequence[int]] = 1,
    ):
        """
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
        if sites.Nparticle is None:
            raise ValueError("`Nparticle` should be specified for NeighborExchange.")

        n_neighbor = [n_neighbor] if isinstance(n_neighbor, int) else n_neighbor
        neighbors = sites.get_neighbor(n_neighbor)
        neighbors = np.concatenate(neighbors, axis=0)
        if sites.is_fermion:
            neighbors = np.concatenate([neighbors, neighbors + sites.N], axis=0)
        self._neighbors = jnp.asarray(neighbors, dtype=jnp.uint16)
        self.ratio = ratio 

        super().__init__(
            state, nsamples, reweight, thermal_steps, sweep_steps, initial_spins
        )

    @property
    def nflips(self) -> int:
        return 4

    @partial(jax.jit, static_argnums=0)
    def _propose(
        self, key: jax.Array, old_spins: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        nsamples = old_spins.shape[0]
        n_total_neighbors = self._neighbors.shape[0]
        n_up_neighbors = n_total_neighbors//2

        key, key2, key3 = jr.split(key, 3)

        nexchange = int(self.ratio*nsamples)

        exchange, hop = jnp.split(jr.permutation(key,nsamples),[nexchange,])
        exchange = jnp.tile(exchange, (4, 1)).T
        hop = jnp.tile(hop, (2, 1)).T

        #exchange rule
        pos = jr.choice(key2, n_up_neighbors, (nexchange,))
        pairs = jnp.concatenate((self._neighbors[pos],self._neighbors[pos]+get_sites().N),-1)              
        rev_pairs = pairs[:,(1,0,3,2)]
        s_exchange = old_spins[exchange, rev_pairs]       
        new_spins = old_spins.at[exchange, pairs].set(s_exchange)

        #hop rule
        pos = jr.choice(key3, n_total_neighbors, (nsamples-nexchange,))
        pairs = self._neighbors[pos]       
        s_exchange = new_spins[hop, pairs[:, ::-1]]
        new_spins = new_spins.at[hop, pairs].set(s_exchange)

        propose_prob = jnp.ones(
            nsamples, dtype=get_default_dtype(), device=get_global_sharding()
        )

        return new_spins, propose_prob
