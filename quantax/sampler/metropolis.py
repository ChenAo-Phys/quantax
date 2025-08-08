from __future__ import annotations
from typing import Optional, Tuple, Union, Sequence
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
from ..state import State, Variational
from ..global_defs import PARTICLE_TYPE, get_subkeys, get_sites
from ..utils import (
    to_global_array,
    to_replicate_array,
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

        self.reset()

    @property
    def particle_type(self) -> Tuple[PARTICLE_TYPE, ...]:
        return tuple()

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
                    "The following wavefunctions are different in direct forward pass and local updates.\n"
                    f"Spin configurations: {samples.spins[~is_wf_close]}\n"
                    f"Direct forward wavefunctions: {wf[~is_wf_close]}\n"
                    f"Local update wavefunctions: {samples.wave_function[~is_wf_close]}"
                )
        else:
            wf = samples.wave_function

        return Samples(samples.spins, wf, None, self._get_reweight_factor(wf))

    def _single_sweep(self, keyp: Key, keyu: Key, samples: Samples) -> Samples:
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
                The ratio of proposal rate P(s|s') / P(s'|s), which is usually 1.
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

        sites = get_sites()
        is_spinful_fermion = sites.particle_type == PARTICLE_TYPE.spinful_fermion
        if is_spinful_fermion and not sites.double_occ:
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
        self._ratio = nsamples / total_nsamples

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
            nstates = get_sites().nstates
            s = [spl._spins.reshape(ndevices, -1, nstates) for spl in self._samplers]
            s = jnp.concatenate(s, axis=1)
            self._spins = to_global_array(s.reshape(-1, nstates))

            if self._thermal_steps > 0:
                self.sweep(self._thermal_steps)

    def _single_sweep(self, keyp: Key, keyu: Key, samples: Samples) -> Samples:
        keyp1, keyp2 = jr.split(keyp, 2)
        i_sampler = jr.choice(keyp1, len(self._samplers), p=self._ratio)
        sampler = self._samplers[i_sampler]
        nflips = sampler.nflips

        new_spins, propose_ratio = sampler._propose(keyp2, samples.spins)
        if not jnp.allclose(propose_ratio, 1.0):
            raise ValueError(
                "`MixSampler` only supports balanced proposals P(s'|s) = P(s|s')."
            )
        if nflips is None:
            new_wf = self._state(new_spins)
            state_internal = None
        else:
            new_wf, state_internal = self._state.ref_forward_with_updates(
                new_spins, samples.spins, nflips, samples.state_internal
            )
        new_samples = Samples(new_spins, new_wf, state_internal=state_internal)
        samples = sampler._update(keyu, propose_ratio, samples, new_samples)
        return samples
