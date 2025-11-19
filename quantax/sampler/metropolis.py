from __future__ import annotations
from typing import Optional, Tuple, Sequence, Union
from jaxtyping import Key
from warnings import warn
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from .sampler import Sampler
from .samples import Samples
from ..state import State
from ..global_defs import PARTICLE_TYPE, get_subkeys, get_sites
from ..utils import (
    array_set,
    to_distribute_array,
    to_replicate_array,
    rand_states,
    filter_tree_map,
    chunk_map,
    PsiArray,
)


@jax.jit
def _get_update_size(is_updated: jax.Array, chunk_size: int) -> jax.Array:
    is_updated = is_updated.reshape(jax.device_count(), -1)
    n_updated = jnp.max(jnp.sum(is_updated, axis=1))
    n_chunks = (n_updated - 1) // chunk_size + 1
    size = n_chunks * chunk_size
    return size


@partial(jax.jit, static_argnames=("size",))
def _get_updated_spins(spins: jax.Array, is_updated: jax.Array, size: int) -> jax.Array:
    ndevices = jax.device_count()
    is_updated = is_updated.reshape(ndevices, -1)
    spins = spins.reshape(ndevices, -1, spins.shape[-1])

    def get_updated(spins, is_updated, size):
        idx = jnp.flatnonzero(is_updated, size=size, fill_value=-1)
        return spins[idx], idx

    get_updated = jax.vmap(get_updated, in_axes=(0, 0, None))
    s_updated, idx = get_updated(spins, is_updated, size)
    return s_updated.reshape(-1, spins.shape[-1]), idx


@jax.jit
def _get_new_psi(
    old_psi: PsiArray, new_psi: PsiArray, is_updated: jax.Array, idx: jax.Array
) -> PsiArray:
    ndevices = jax.device_count()
    old_psi = old_psi.reshape(ndevices, -1)
    new_psi = new_psi.reshape(ndevices, -1)
    is_updated = is_updated.reshape(ndevices, -1)
    idx = idx.reshape(ndevices, -1)

    def select_psi(old_psi, new_psi, is_updated, idx):
        old_psi, treedef = jax.tree.flatten(old_psi)
        new_psi, _ = jax.tree.flatten(new_psi)
        new_list = []
        for old, new in zip(old_psi, new_psi):
            new = array_set(old, idx, new)
            new = jnp.where(is_updated, new, old)
            new_list.append(new)
        new_psi = jax.tree.unflatten(treedef, new_list)
        return new_psi

    select_psi = jax.vmap(select_psi)
    psi = select_psi(old_psi, new_psi, is_updated, idx)
    return psi.flatten()


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
        r"""
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

        particle_type = get_sites().particle_type
        if particle_type not in tuple(self.particle_type):
            name = particle_type.name
            raise ValueError(
                f"Particle type {name} is not supported by {self.__class__.__name__}."
            )

        if thermal_steps is None:
            self._thermal_steps = 20 * self.Nmodes
        else:
            self._thermal_steps = thermal_steps
        if sweep_steps is None:
            self._sweep_steps = 2 * self.Nmodes
        else:
            self._sweep_steps = sweep_steps

        if initial_spins is not None:
            if initial_spins.ndim == 1:
                initial_spins = jnp.tile(initial_spins, (self.nsamples, 1))
            else:
                initial_spins = initial_spins.reshape(self.nsamples, self.Nmodes)
            initial_spins = to_distribute_array(initial_spins.astype(jnp.int8))
        self._initial_spins = initial_spins

        self.reset()

    @property
    def particle_type(self) -> Tuple[PARTICLE_TYPE, ...]:
        return (
            PARTICLE_TYPE.spin,
            PARTICLE_TYPE.spinful_fermion,
            PARTICLE_TYPE.spinless_fermion,
        )

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

        state = self._state
        use_ref = state.use_ref and self.nflips is not None
        if use_ref:
            has_chunk = hasattr(state, "ref_chunk") and state.ref_chunk is not None
            chunk_size = state.ref_chunk
        else:
            has_chunk = (
                hasattr(state, "forward_chunk") and state.forward_chunk is not None
            )
            chunk_size = state.forward_chunk
        is_chunked = has_chunk and chunk_size < self.nsamples // jax.device_count()

        if is_chunked:
            if use_ref:
                fn_sweep = chunk_map(
                    self._partial_sweep, in_axes=(None, 0), chunk_size=chunk_size
                )
                samples = fn_sweep(nsweeps, self._spins)
            else:
                samples = self._chunk_sweep(nsweeps, chunk_size)
        else:
            samples = self._partial_sweep(nsweeps, self._spins)
        self._spins = samples.spins

        if use_ref:
            psi = self._state(samples.spins)
            cond1 = jnp.abs(samples.psi - psi) < 1e-8
            cond2 = jnp.abs(samples.psi / psi - 1) < 1e-3
            is_psi_close = cond1 | cond2
            ndiff = jnp.sum(~is_psi_close)
            if ndiff > 0:
                if jax.process_index() == 0:
                    warn(
                        f"{ndiff} out of {self.nsamples} wavefunctions are not close in "
                        "direct forward pass and local updates. This may indicate inaccurate local updates."
                    )
            samples = eqx.tree_at(lambda tree: tree.psi, samples, psi)

        return samples

    def _chunk_sweep(self, nsweeps: int, chunk_size: int) -> Samples:
        """
        Generate new samples in chunks for states with large memory consumption.
        Every sweep step is chunked into several sub-steps.
        """
        keys_propose = get_subkeys(nsweeps)
        keys_update = get_subkeys(nsweeps)
        psi = self._state(self._spins)
        samples = Samples(self._spins, psi)

        for keyp, keyu in zip(keys_propose, keys_update):
            proposal = self.propose(keyp, samples.spins)
            if isinstance(proposal, tuple):
                new_spins, propose_ratio = proposal
            else:
                new_spins = proposal
                propose_ratio = None

            is_updated = jnp.any(samples.spins != new_spins, axis=1)
            size = _get_update_size(is_updated, chunk_size).item()
            s_updated, idx = _get_updated_spins(new_spins, is_updated, size)
            new_psi = self._state(s_updated)
            new_psi = _get_new_psi(samples.psi, new_psi, is_updated, idx)
            new_samples = Samples(new_spins, new_psi)
            samples = self._update(keyu, propose_ratio, samples, new_samples)

        psi = samples.psi
        return Samples(samples.spins, psi, None, self._get_reweight_factor(psi))

    def _partial_sweep(self, nsweeps: int, spins: jax.Array) -> Samples:
        """
        Generate new samples for a given set of initial spins.
        """
        psi = self._state(spins)
        state_internal = self._state.init_internal(spins)
        samples = Samples(spins, psi, state_internal)

        keys_propose = get_subkeys(nsweeps)
        keys_update = get_subkeys(nsweeps)
        for keyp, keyu in zip(keys_propose, keys_update):
            samples = self._single_sweep(keyp, keyu, samples)

        psi = samples.psi
        return Samples(samples.spins, psi, None, self._get_reweight_factor(psi))

    def _single_sweep(self, keyp: Key, keyu: Key, samples: Samples) -> Samples:
        proposal = self.propose(keyp, samples.spins)
        if isinstance(proposal, tuple):
            new_spins, propose_ratio = proposal
        else:
            new_spins = proposal
            propose_ratio = None

        new_psi, state_internal = self._state.ref_forward_with_updates(
            new_spins, samples.spins, self.nflips, samples.state_internal
        )
        new_samples = Samples(new_spins, new_psi, state_internal)
        samples = self._update(keyu, propose_ratio, samples, new_samples)
        return samples

    def propose(
        self, key: Key, old_spins: jax.Array
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        r"""
        Propose new configurations.

        :return:
            Either a tuple of (new_spins, propose_ratio) or new_spins only,
            where new_spins is the proposed configurations,
            and propose_ratio is the ratio of proposal rate :math:`P(s|s') / P(s'|s)`.
            propose_ratio is set to 1 if not returned.
        """
        raise NotImplementedError

    @partial(eqx.filter_jit, donate="all-except-first")
    def _update(
        self,
        key: Key,
        propose_ratio: Optional[jax.Array],
        old_samples: Samples,
        new_samples: Samples,
    ) -> Samples:
        nsamples, Nmodes = old_samples.spins.shape
        rate_accept = jnp.abs(new_samples.psi / old_samples.psi) ** self._reweight
        if propose_ratio is not None:
            rate_accept *= propose_ratio
        rate_reject = 1.0 - jr.uniform(key, (nsamples,), rate_accept.dtype)
        accepted = (rate_accept > rate_reject) | (jnp.abs(old_samples.psi) == 0.0)

        sites = get_sites()
        is_spinful_fermion = sites.particle_type == PARTICLE_TYPE.spinful_fermion
        if is_spinful_fermion and not sites.double_occ:
            s = new_samples.spins.reshape(nsamples, 2, Nmodes // 2)
            occ_allowed = jnp.all(jnp.any(s <= 0, axis=1), axis=1)
        else:
            occ_allowed = True

        updated = jnp.any(old_samples.spins != new_samples.spins, axis=1)

        cond = accepted & updated & occ_allowed

        def f_select(new, old):
            cond_expand = cond.reshape([-1] + [1] * (new.ndim - 1))
            return jnp.where(cond_expand, new, old)

        return filter_tree_map(f_select, new_samples, old_samples)


class MixSampler(Metropolis):
    r"""
    A mixture of several metropolis samplers. New samples are proposed randomly by
    every sampler.
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

        self._samplers = tuple(samplers)
        nsamples = np.array([sampler.nsamples for sampler in samplers])
        total_nsamples = np.sum(nsamples)
        self._ratio = to_replicate_array(nsamples / total_nsamples)

        super().__init__(
            state, total_nsamples, reweight, thermal_steps, sweep_steps, initial_spins
        )

    @property
    def particle_type(self) -> Tuple[PARTICLE_TYPE, ...]:
        return (get_sites().particle_type,)

    @property
    def nflips(self) -> int:
        nflips = tuple(sampler.nflips for sampler in self._samplers)
        return None if None in nflips else max(nflips)

    def reset(self) -> None:
        if hasattr(self, "_spins") or self._initial_spins is not None:
            super().reset()
        else:
            ndevices = jax.device_count()
            Nmodes = get_sites().Nmodes
            s = [spl._spins.reshape(ndevices, -1, Nmodes) for spl in self._samplers]
            s = jnp.concatenate(s, axis=1)
            self._spins = to_distribute_array(s.reshape(-1, Nmodes))

            if self._thermal_steps > 0:
                self.sweep(self._thermal_steps)

    @eqx.filter_jit
    def _rand_sampler_idx(self, key: Key, num: Optional[int] = None) -> int:
        if num is None:
            return jr.choice(key, len(self._samplers), p=self._ratio)
        else:
            return jr.choice(key, len(self._samplers), (num,), p=self._ratio)
    
    def _chunk_sweep(self, nsweeps: int, chunk_size: int) -> Samples:
        """
        Generate new samples in chunks for states with large memory consumption.
        Every sweep step is chunked into several sub-steps.
        """
        idx_samplers = self._rand_sampler_idx(get_subkeys(), nsweeps)
        psi = self._state(self._spins)
        samples = Samples(self._spins, psi)

        keys_propose = get_subkeys(nsweeps)
        keys_update = get_subkeys(nsweeps)
        for i_sampler, keyp, keyu in zip(idx_samplers, keys_propose, keys_update):
            sampler = self._samplers[i_sampler]
            proposal = sampler.propose(keyp, samples.spins)
            new_spins = proposal
            propose_ratio = None

            is_updated = jnp.any(samples.spins != new_spins, axis=1)
            size = _get_update_size(is_updated, chunk_size).item()
            s_updated, idx = _get_updated_spins(new_spins, is_updated, size)
            new_psi = self._state(s_updated)
            new_psi = _get_new_psi(samples.psi, new_psi, is_updated, idx)
            new_samples = Samples(new_spins, new_psi)
            samples = self._update(keyu, propose_ratio, samples, new_samples)

        psi = samples.psi
        return Samples(samples.spins, psi, None, self._get_reweight_factor(psi))
    
    def _partial_sweep(self, nsweeps: int, spins: jax.Array) -> Samples:
        """
        Generate new samples for a given set of initial spins.
        """
        idx_samplers = self._rand_sampler_idx(get_subkeys(), nsweeps)
        psi = self._state(spins)
        state_internal = self._state.init_internal(spins)
        samples = Samples(spins, psi, state_internal)

        keys_propose = get_subkeys(nsweeps)
        keys_update = get_subkeys(nsweeps)
        for i_sampler, keyp, keyu in zip(idx_samplers, keys_propose, keys_update):
            sampler = self._samplers[i_sampler]
            samples = sampler._single_sweep(keyp, keyu, samples)

        psi = samples.psi
        return Samples(samples.spins, psi, None, self._get_reweight_factor(psi))
