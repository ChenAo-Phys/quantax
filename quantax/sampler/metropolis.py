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
from ..state import State
from ..global_defs import get_subkeys, get_sites, get_default_dtype
from ..utils import to_global_array, to_replicate_array, rand_states, filter_tree_map


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
            initial_spins = to_global_array(initial_spins)
        self._initial_spins = initial_spins
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
            spins = rand_states(self.nsamples, self.state.Nparticle)
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
        wf = self._state(self._spins)
        state_internal = self._state.init_internal(self._spins)
        status = SamplerStatus(self._spins, wf, self._propose_prob, state_internal)

        if nsweeps is None:
            nsweeps = self._sweep_steps
        keys_propose = to_replicate_array(get_subkeys(nsweeps))
        keys_update = to_replicate_array(get_subkeys(nsweeps))
        for keyp, keyu in zip(keys_propose, keys_update):
            status = self._single_sweep(keyp, keyu, status)

        self._spins = status.spins
        self._propose_prob = status.propose_prob
        return Samples(
            status.spins, status.wave_function, self._reweight, status.state_internal
        )

    def _single_sweep(
        self, keyp: Key, keyu: Key, status: SamplerStatus
    ) -> SamplerStatus:
        new_spins, new_propose_prob = self._propose(keyp, status.spins)
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
        is_selected: jax.Array, old_tree: PyTree, new_tree: PyTree
    ) -> PyTree:
        fn = lambda old, new: jnp.where(is_selected, new, old)
        return filter_tree_map(fn, old_tree, new_tree)

    @partial(jax.jit, static_argnums=0, donate_argnums=2)
    def _update(
        self, key: jax.Array, old_status: SamplerStatus, new_status: SamplerStatus
    ) -> SamplerStatus:
        nsamples, nstates = old_status.spins.shape
        old_prob = jnp.abs(old_status.wave_function) ** self._reweight
        new_prob = jnp.abs(new_status.wave_function) ** self._reweight
        rand = 1.0 - jr.uniform(key, (nsamples,), old_prob.dtype)
        rate_accept = new_prob * old_status.propose_prob
        rate_reject = old_prob * new_status.propose_prob * rand

        accepted = (rate_accept > rate_reject) | (old_prob == 0.)
        updated = jnp.any(old_status.spins != new_status.spins, axis=1)
        is_selected = accepted & updated
        return self._update_selected(is_selected, old_status, new_status)


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
        nsamples, nsites = old_spins.shape
        pos = jr.choice(key, nsites, (nsamples,))
        new_spins = old_spins.at[jnp.arange(nsamples), pos].multiply(-1)
        propose_prob = to_global_array(jnp.ones(nsamples, dtype=get_default_dtype()))
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
        if state.Nparticle is None:
            raise ValueError("`Nparticle` of 'state' should be specified.")

        sites = get_sites()
        n_neighbor = [n_neighbor] if isinstance(n_neighbor, int) else n_neighbor
        neighbors = sites.get_neighbor(n_neighbor)
        neighbors = np.concatenate(neighbors, axis=0)
        if sites.is_fermion:
            neighbors = np.concatenate([neighbors, neighbors + sites.nsites], axis=0)
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
        propose_prob = to_global_array(jnp.ones(nsamples, dtype=get_default_dtype()))
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
        if state.Nparticle is None:
            raise ValueError("`Nparticle` of 'state' should be specified.")

        sites = get_sites()
        Nup = np.sum(state.Nparticle)
        if 2 * Nup <= state.nstates:
            self._hopping_particle = 1
            self._nhopping = Nup
        else:
            self._hopping_particle = -1
            self._nhopping = state.nstates - Nup

        n_neighbor = [n_neighbor] if isinstance(n_neighbor, int) else n_neighbor
        neighbors = sites.get_neighbor(n_neighbor)
        neighbors = np.concatenate(neighbors, axis=0)

        neighbor_matrix = np.zeros((state.nsites, state.nsites), dtype=np.bool_)
        neighbor_matrix[neighbors[:, 0], neighbors[:, 1]] = True
        neighbor_matrix = neighbor_matrix | neighbor_matrix.T
        neighbor_count = np.sum(neighbor_matrix, axis=1)
        if not np.all(neighbor_count == neighbor_count[0]):
            raise RuntimeError("Different sites have different amount of neighbors.")

        neighbor_idx = np.nonzero(neighbor_matrix)[1].reshape(sites.nsites, -1)
        if sites.is_fermion:
            neighbor_idx = np.concatenate(
                [neighbor_idx, neighbor_idx + sites.nsites], axis=0
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
        key1, key2 = jr.split(key, 2)
        nsamples = old_spins.shape[0]
        arange = jnp.arange(nsamples)

        hopping_particles = jnp.nonzero(
            old_spins == self._hopping_particle, size=nsamples * self._nhopping
        )[1].reshape(nsamples, self._nhopping)
        hopping_idx = jr.choice(key1, self._nhopping, (nsamples,))
        hopping_particles = hopping_particles[arange, hopping_idx]

        neighbors = self._neighbor_idx[hopping_particles]
        neighbor_idx = jr.choice(key2, neighbors.shape[1], (nsamples,))
        neighbors = neighbors[arange, neighbor_idx]

        pairs = jnp.stack([hopping_particles, neighbors], axis=1)
        arange = jnp.tile(arange, (2, 1)).T
        s_exchange = old_spins[arange, pairs[:, ::-1]]
        new_spins = old_spins.at[arange, pairs].set(s_exchange)
        propose_prob = to_global_array(jnp.ones(nsamples, dtype=get_default_dtype()))
        return new_spins, propose_prob
