from __future__ import annotations
from typing import Optional, Tuple, Union, Sequence
from jaxtyping import Key, PyTree
from warnings import warn
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from .sampler import Sampler
from .samples import Samples
from ..state import State, Variational
from ..global_defs import get_subkeys, get_sites, get_real_dtype
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
    def is_balanced(self) -> bool:
        """
        Whether the sampler has balanced proposal rate P(s'|s) = P(s|s'), default to True
        """
        return True

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
            self._spins = rand_states(self.nsamples)
        else:
            self._spins = self._initial_spins

        if self._thermal_steps > 0:
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
                self._partial_sweep, in_axes=(None, 0), chunk_size=chunk_size
            )
        else:
            fn_sweep = self._partial_sweep

        samples = fn_sweep(nsweeps, self._spins)

        self._spins = samples.spins
        if isinstance(self._state, Variational):
            new_max = jnp.max(jnp.abs(samples.wave_function))
            old_max = self._state._maximum
            self._state._maximum = jnp.where(new_max > old_max, new_max, old_max)
        return samples

    def _partial_sweep(self, nsweeps: int, spins: jax.Array) -> Samples:
        wf = self._state(spins)
        if self.nflips is None:
            state_internal = None
        else:
            state_internal = self._state.init_internal(spins)
        samples = Samples(spins, wf, state_internal)

        keys_propose = to_replicate_array(get_subkeys(nsweeps))
        keys_update = to_replicate_array(get_subkeys(nsweeps))
        for keyp, keyu in zip(keys_propose, keys_update):
            samples = self._single_sweep(keyp, keyu, samples)

        if samples.state_internal is not None:
            samples = eqx.tree_at(lambda tree: tree.state_internal, samples, None)
            wf = self._state(samples.spins)
            is_wf_close = jnp.isclose(wf, samples.wave_function)
            if not jnp.all(is_wf_close):
                warn(
                    "The following wavefunctions are different in direct forward pass and local updates. "
                    f"Spin configuration: {samples.spins[~is_wf_close]}; "
                    f"Direct forward wavefunction: {wf[~is_wf_close]}; "
                    f"Local update wavefunction: {samples.wave_function[~is_wf_close]}."
                )
        else:
            wf = samples.wave_function

        return Samples(samples.spins, wf, None, self._get_reweight_factor(wf))

    def _single_sweep(
        self, keyp: Key, keyu: Key, samples: Samples
    ) -> Samples:
        new_spins, propose_ratio = self._propose(keyp, samples.spins)
        if self.nflips is None:
            new_wf = self._state(new_spins)
            state_internal = None
        else:
            new_wf, state_internal = self._state.ref_forward_with_updates(
                new_spins, samples.spins, self.nflips, samples.state_internal
            )
        new_samples = Samples(new_spins, new_wf, state_internal=state_internal)
        samples = self._update(keyu, propose_ratio, samples, new_samples)
        return samples

    def _propose(
        self, key: jax.Array, old_spins: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Propose new spin configurations, return spin and proposal weight

        :return:
            spins:
                The proposed spin configurations

            propose_ratio:
                The ratio of proposal rate P(s|s') / P(s'|s)
        """

    @partial(jax.jit, static_argnums=0, donate_argnums=3)
    def _update(
        self,
        key: jax.Array,
        propose_ratio: jax.Array,
        old_samples: Samples,
        new_samples: Samples,
    ) -> Samples:
        nsamples, nstates = old_samples.spins.shape
        old_prob = jnp.abs(old_samples.wave_function) ** self._reweight
        new_prob = jnp.abs(new_samples.wave_function) ** self._reweight
        rand = 1.0 - jr.uniform(key, (nsamples,), old_prob.dtype)
        rate_accept = new_prob * propose_ratio
        rate_reject = old_prob * rand

        if self._is_fermion and not self._double_occ:
            s = new_samples.spins.reshape(nsamples, 2, nstates // 2)
            occ_allowed = jnp.all(jnp.any(s <= 0, axis=1), axis=1)
        else:
            occ_allowed = True

        accepted = (rate_accept > rate_reject) | (old_prob == 0.0)
        updated = jnp.any(old_samples.spins != new_samples.spins, axis=1)

        cond = accepted & updated & occ_allowed

        def f_select(new, old):
            cond_expand = cond.reshape([-1] + [1] * (new.ndim - 1))
            return jnp.where(cond_expand, new, old)
            
        return filter_tree_map(f_select, new_samples, old_samples)


class LocalFlip(Metropolis):
    """
    Generate Monte Carlo samples by locally flipping spins.
    In fermion systems, it's equivalent to creating and annihilating particles locally.
    """

    @property
    def nflips(self) -> int:
        return 1

    @partial(jax.jit, static_argnums=0)
    def _propose(
        self, key: jax.Array, old_spins: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        nsamples, N = old_spins.shape
        pos = jr.choice(key, N, (nsamples,))
        new_spins = old_spins.at[jnp.arange(nsamples), pos].multiply(-1)

        ratio = jnp.ones(nsamples, dtype=get_real_dtype(), device=get_global_sharding())

        return new_spins, ratio


class NeighborExchange(Metropolis):
    """
    Generate Monte Carlo samples by exchanging neighbor spins or fermions.
    In fermion systems, it is similar to `quantax.sampler.ParticleHop`,
    but different to `quantax.sampler.SiteExchange`.
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
        n_neighbors = self._neighbors.shape[0]
        pos = jr.choice(key, n_neighbors, (nsamples,))
        pairs = self._neighbors[pos]

        arange = jnp.arange(nsamples)
        arange = jnp.tile(arange, (2, 1)).T
        s_exchange = old_spins[arange, pairs[:, ::-1]]
        new_spins = old_spins.at[arange, pairs].set(s_exchange)

        ratio = jnp.ones(nsamples, dtype=get_real_dtype(), device=get_global_sharding())

        return new_spins, ratio


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
        n_neighbors = neighbors.shape[1]
        neighbor_idx = jr.choice(key[-1], n_neighbors, (nsamples,))
        arange = jnp.arange(nsamples)
        neighbors = neighbors[arange, neighbor_idx]

        pairs = jnp.stack([hopping_particles, neighbors], axis=1)
        arange = jnp.tile(arange, (2, 1)).T
        s_exchange = old_spins[arange, pairs[:, ::-1]]
        new_spins = old_spins.at[arange, pairs].set(s_exchange)

        ratio = jnp.ones(nsamples, dtype=get_real_dtype(), device=get_global_sharding())

        return new_spins, ratio


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
        if not sites.is_fermion:
            raise ValueError("`SiteExchange` should be used for fermion systems")
        if sites.Nparticle is None:
            raise ValueError("`Nparticle` should be specified for `SiteExchange`.")

        n_neighbor = [n_neighbor] if isinstance(n_neighbor, int) else n_neighbor
        neighbors = sites.get_neighbor(n_neighbor)
        neighbors = np.concatenate(neighbors, axis=0)
        self._neighbors = jnp.asarray(neighbors, dtype=jnp.uint16)

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
        n_neighbors = self._neighbors.shape[0]
        pos = jr.choice(key, n_neighbors, (nsamples,))
        pairs = self._neighbors[pos]

        N = get_sites().N
        arange = jnp.arange(nsamples)
        arange = jnp.tile(arange, (2, 1)).T
        s_exchange_up = old_spins[arange, pairs[:, ::-1]]
        new_spins = old_spins.at[arange, pairs].set(s_exchange_up)
        s_exchange_dn = old_spins[arange, pairs[:, ::-1] + N]
        new_spins = new_spins.at[arange, pairs + N].set(s_exchange_dn)

        ratio = jnp.ones(nsamples, dtype=get_real_dtype(), device=get_global_sharding())

        return new_spins, ratio


class MixSampler(Metropolis):
    """
    A mixture of several metropolis samplers. New samples are proposed randomly by
    every sampler.

    .. warning::

        This sampler only
    """

    def __init__(
        self,
        samplers: Sequence[Metropolis],
        reweight: float = 2.0,
        thermal_steps: Optional[int] = None,
        sweep_steps: Optional[int] = None,
        initial_spins: Optional[jax.Array] = None,
    ):
        state = samplers[0].state
        for sampler in samplers[1:]:
            if sampler.state is not state:
                raise ValueError(
                    "The states of component samplers should be the same in `MixSampler`."
                )

        if not all(sampler.is_balanced for sampler in samplers):
            raise NotImplementedError(
                "The `MixSampler` is only implemented for samplers with"
                "balanced proposal rate P(s'|s) = P(s|s')."
            )

        self._samplers = tuple(samplers)
        ndevices = jax.device_count()
        nsamples_each = tuple(sampler.nsamples // ndevices for sampler in samplers)
        sections = tuple(np.cumsum(nsamples_each).tolist())
        self._sections = sections[:-1]
        nsamples = sections[-1] * ndevices

        super().__init__(
            state, nsamples, reweight, thermal_steps, sweep_steps, initial_spins
        )

    @property
    def nflips(self) -> int:
        nflips = tuple(sampler.nflips for sampler in self._samplers)
        return None if None in nflips else max(nflips)

    @partial(jax.jit, static_argnums=0)
    def _propose(
        self, key: jax.Array, old_spins: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        ndevices = jax.device_count()
        nsamples = old_spins.shape[0]
        keys = jr.split(key, len(self._samplers) + 1)

        indices = jnp.arange(nsamples // ndevices)
        indices = jnp.stack([indices] * ndevices, axis=0)
        indices = jr.permutation(keys[0], indices, axis=1, independent=True)
        indices = jax.lax.with_sharding_constraint(indices, get_global_sharding())

        f_slicing = lambda spins, indices: jnp.split(spins[indices], self._sections)
        f_slicing = jax.vmap(f_slicing)
        spins = old_spins.reshape(ndevices, -1, old_spins.shape[-1])
        spins = f_slicing(spins, indices)

        new_spins = []
        for sampler, key, s in zip(self._samplers, keys[1:], spins):
            s = s.reshape(-1, s.shape[-1])
            s, p = sampler._propose(key, s)
            new_spins.append(s.reshape(ndevices, -1, s.shape[-1]))

        @jax.vmap
        def f_combine(spins, indices):
            spins = jnp.concatenate(spins)
            spins = spins.at[indices].set(spins)
            return spins

        new_spins = f_combine(new_spins, indices)
        new_spins = new_spins.reshape(-1, new_spins.shape[-1])

        ratio = jnp.ones(nsamples, dtype=get_real_dtype(), device=get_global_sharding())

        return new_spins, ratio
