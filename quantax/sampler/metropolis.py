from typing import Optional, Tuple, Union, Sequence
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
from .sampler import Sampler
from .status import SamplerStatus, Samples
from ..state import State
from ..global_defs import get_subkeys, get_sites, get_default_dtype
from ..utils import to_global_array, to_replicate_array, rand_states


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
            default to be ``nsites * 20``.

        :param sweep_steps:
            The number of steps for generating new samples, default to be ``nsites * 2``

        :param initial_spins:
            The initial spins for every Markov chain before the thermalization steps,
            default to be random spins.
        """
        super().__init__(state, nsamples, reweight)
        self._reweight = to_replicate_array(reweight)

        if thermal_steps is None:
            self._thermal_steps = 20 * self.nsites
        else:
            self._thermal_steps = thermal_steps
        if sweep_steps is None:
            self._sweep_steps = 2 * self.nsites
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

    def reset(self) -> None:
        """
        Reset all Markov chains to ``initial_spins`` and thermalize them
        """
        if self._initial_spins is None:
            spins = rand_states(self.nsamples, self.state.Nparticle, distributed=True)
        else:
            spins = self._initial_spins

        spins, propose_prob = self._propose(get_subkeys(), spins)
        self._status = SamplerStatus(spins, propose_prob=propose_prob)
        self.sweep(self._thermal_steps)

    def sweep(self, nsweeps: Optional[int] = None) -> Samples:
        """
        Generate new samples

        :param nsweeps:
            Number of sweeps for generating the new samples, default to be
            ``self._sweep_steps``
        """
        spins = self.current_spins
        wf = self._state(spins)
        prob = jnp.abs(wf) ** self._reweight
        self._status = SamplerStatus(spins, wf, prob, self._status.propose_prob)

        if nsweeps is None:
            nsweeps = self._sweep_steps
        keys_propose = to_replicate_array(get_subkeys(nsweeps))
        keys_update = to_replicate_array(get_subkeys(nsweeps))
        for keyp, keyu in zip(keys_propose, keys_update):
            new_spins, new_propose_prob = self._propose(keyp, self.current_spins)
            new_wf = self._state(new_spins)
            new_prob = jnp.abs(new_wf) ** self._reweight
            new_status = SamplerStatus(new_spins, new_wf, new_prob, new_propose_prob)
            self._status = self._update(keyu, self._status, new_status)
        return Samples(self.current_spins, self.current_wf, self._reweight)

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
    @jax.jit
    def _update(
        key: jax.Array, old_status: SamplerStatus, new_status: SamplerStatus
    ) -> SamplerStatus:
        nsamples, nstates = old_status.spins.shape
        dtype = old_status.prob.dtype
        rand = 1.0 - jr.uniform(key, (nsamples,), dtype)
        rate_accept = new_status.prob * old_status.propose_prob
        rate_reject = old_status.prob * new_status.propose_prob * rand
        selected = rate_accept > rate_reject
        selected = jnp.where(old_status.prob == 0., True, selected)

        selected_spins = jnp.tile(selected, (nstates, 1)).T
        spins = jnp.where(selected_spins, new_status.spins, old_status.spins)
        wf = jnp.where(selected, new_status.wave_function, old_status.wave_function)
        prob = jnp.where(selected, new_status.prob, old_status.prob)
        p_prob = jnp.where(selected, new_status.propose_prob, old_status.propose_prob)
        return SamplerStatus(spins, wf, prob, p_prob)


class LocalFlip(Metropolis):
    """
    Generate Monte Carlo samples by locally flipping spins.
    """
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
    Generate Monte Carlo samples by exchanging neighbor spins.
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
            default to 2.0

        :param thermal_steps:
            The number of thermalization steps in the beginning of each Markov chain,
            default to be ``nsites * 20``.

        :param sweep_steps:
            The number of steps for generating new samples, default to be ``nsites * 2``

        :param initial_spins:
            The initial spins for every Markov chain before the thermalization steps,
            default to be random spins.

        :param n_neighbor:
            The neighbors to be considered by spin exchanges, default to nearest neighbors.
        """
        if state.Nparticle is None:
            raise ValueError("`Nparticle` of 'state' should be specified.")
        n_neighbor = [n_neighbor] if isinstance(n_neighbor, int) else n_neighbor
        sites = get_sites()
        neighbors = sites.get_neighbor(n_neighbor)
        self._neighbors = jnp.concatenate(neighbors, axis=0)
        if sites.is_fermion:
            self._neighbors = [self._neighbors, self._neighbors + sites.nsites]
            self._neighbors = jnp.concatenate(self._neighbors, axis=0)

        super().__init__(
            state, nsamples, reweight, thermal_steps, sweep_steps, initial_spins
        )

    @partial(jax.jit, static_argnums=0)
    def _propose(
        self, key: jax.Array, old_spins: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        nsamples = old_spins.shape[0]
        pos = jr.choice(key, self._neighbors.shape[0], (nsamples,))
        pairs = self._neighbors[pos]

        arange = jnp.arange(nsamples)
        arange = jnp.tile(arange, (2, 1)).T
        spins_exchange = old_spins[arange, pairs[:, ::-1]]
        new_spins = old_spins.at[arange, pairs].set(spins_exchange)
        propose_prob = to_global_array(jnp.ones(nsamples, dtype=get_default_dtype()))
        return new_spins, propose_prob
