from __future__ import annotations
from typing import Sequence, Any
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
    to_distributed_array,
    to_replicated_array,
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
def _get_updated_spins(
    spins: jax.Array, is_updated: jax.Array, size: int
) -> tuple[jax.Array, jax.Array]:
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
        thermal_steps: int | None = None,
        sweep_steps: int | None = None,
        initial_spins: jax.Array | None = None,
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
            default to be random.
        """
        super().__init__(state, nsamples, reweight)
        self._reweight = to_replicated_array(reweight)

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

        use_ref = state.use_ref
        if use_ref:
            mode_keys = self.update_mode.keys()
            if not all(mode in mode_keys for mode in state.required_update_modes):
                warn(
                    "The update_modes required by the state are not all provided "
                    "in the sampler. The fast local updates are not utilized."
                )
                use_ref = False
        self._use_ref = use_ref

        self.reset(initial_spins=initial_spins)

    @property
    def particle_type(self) -> tuple[PARTICLE_TYPE, ...]:
        """The particle types of the systems that this sampler can be applied to."""
        return (
            PARTICLE_TYPE.spin,
            PARTICLE_TYPE.spinful_fermion,
            PARTICLE_TYPE.spinless_fermion,
        )

    @property
    def update_mode(self) -> dict[str, Any]:
        """
        The update mode of local updates generated in the sampler.
        """
        return {}

    @property
    def use_ref(self) -> bool:
        """
        Whether to use reference implementation for local updates.
        """
        return self._use_ref

    def reset(
        self, nsweeps: int | None = None, initial_spins: jax.Array | None = None
    ) -> None:
        """
        Reset all Markov chains and thermalize them.

        :param nsweeps:
            Number of sweeps for thermalizing the new samples, default to be
            ``self._thermal_steps``

        :param initial_spins:
            The initial spins for every Markov chain before the thermalization steps,
            default to be random.
        """
        if initial_spins is None:
            self._spins = rand_states(self.nsamples)
        else:
            if initial_spins.ndim == 1:
                initial_spins = jnp.tile(initial_spins, (self.nsamples, 1))
            else:
                initial_spins = initial_spins.reshape(self.nsamples, self.Nmodes)
            self._spins = to_distributed_array(initial_spins.astype(jnp.int8))
        self._psi = None

        if nsweeps is None:
            nsweeps = self._thermal_steps
        if nsweeps > 0:
            self.sweep(nsweeps)

    def sweep(self, nsweeps: int | None = None) -> Samples:
        """
        Generate new samples

        :param nsweeps:
            Number of sweeps for generating the new samples, default to be
            ``self._sweep_steps``
        """
        if nsweeps is None:
            nsweeps = self._sweep_steps

        attr = "ref_chunk" if self.use_ref else "forward_chunk"
        chunk_size = getattr(self._state, attr, None)
        ns = self.nsamples // jax.device_count()

        if chunk_size is not None and chunk_size < ns:
            if self.use_ref:
                fn_sweep = chunk_map(
                    self._partial_sweep, in_axes=(None, 0), chunk_size=chunk_size
                )
                samples = fn_sweep(nsweeps, self._spins)
            else:
                samples = self._chunk_sweep(nsweeps, chunk_size)
        else:
            samples = self._partial_sweep(nsweeps, self._spins)

        self._spins = samples.spins
        self._psi = samples.psi  # Not reusable at next iteration, as state might change
        reweight_factor = self._get_reweight_factor(samples.psi)
        return Samples(self._spins, None, None, reweight_factor)

    def _chunk_sweep(self, nsweeps: int, chunk_size: int) -> Samples:
        """
        Generate new samples in chunks for states with large memory consumption.
        Every sweep step is chunked into several sub-steps.
        """
        keys_propose = get_subkeys(nsweeps)
        keys_update = get_subkeys(nsweeps)
        psi = self._state.fast_forward(self._spins)
        samples = Samples(self._spins, psi)

        for keyp, keyu in zip(keys_propose, keys_update):
            new_spins, propose_ratio = self._propose_spins_and_ratio(
                keyp, samples.spins
            )

            is_updated = jnp.any(samples.spins != new_spins, axis=1)
            size = _get_update_size(is_updated, chunk_size).item()
            s_updated, idx = _get_updated_spins(new_spins, is_updated, size)
            new_psi = self._state.fast_forward(s_updated)
            new_psi = _get_new_psi(samples.psi, new_psi, is_updated, idx)
            new_samples = Samples(new_spins, new_psi)
            samples = self._update(keyu, propose_ratio, samples, new_samples)

        return Samples(samples.spins, samples.psi)

    def _partial_sweep(self, nsweeps: int, spins: jax.Array) -> Samples:
        """
        Generate new samples for a given set of initial spins.
        """
        if self.use_ref:
            psi, state_internal = self._state.init_internal(spins)
        else:
            psi = self._state.fast_forward(spins)
            state_internal = None
        samples = Samples(spins, psi, state_internal)

        keys_propose = get_subkeys(nsweeps)
        keys_update = get_subkeys(nsweeps)
        sweep_fn = self._single_sweep_ref if self.use_ref else self._single_sweep_direct
        for keyp, keyu in zip(keys_propose, keys_update):
            samples = sweep_fn(keyp, keyu, samples)

        return Samples(samples.spins, samples.psi)

    def _single_sweep_ref(self, keyp: Key, keyu: Key, samples: Samples) -> Samples:
        new_spins, propose_ratio = self._propose_spins_and_ratio(keyp, samples.spins)
        new_psi, state_internal = self._state.ref_forward(
            new_spins,
            samples.spins,
            self.update_mode,
            samples.state_internal,
            return_update=True,
        )
        new_samples = Samples(new_spins, new_psi, state_internal)
        samples = self._update(keyu, propose_ratio, samples, new_samples)
        return samples

    def _single_sweep_direct(self, keyp: Key, keyu: Key, samples: Samples) -> Samples:
        new_spins, propose_ratio = self._propose_spins_and_ratio(keyp, samples.spins)
        new_psi = self._state.fast_forward(new_spins)
        new_samples = Samples(new_spins, new_psi)
        samples = self._update(keyu, propose_ratio, samples, new_samples)
        return samples

    @partial(jax.jit, static_argnums=0)
    def propose(
        self, key: Key, old_spins: jax.Array
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        r"""
        Propose new configurations.

        :return:
            Either a tuple of (new_spins, propose_ratio) or new_spins only,
            where new_spins is the proposed configurations,
            and propose_ratio is the ratio of proposal rate :math:`P(s|s') / P(s'|s)`.
            propose_ratio is set to 1 if not returned.
        """
        raise NotImplementedError

    @eqx.filter_jit
    def _propose_spins_and_ratio(
        self, key: Key, old_spins: jax.Array
    ) -> tuple[jax.Array, jax.Array | None]:
        """
        Split the output of ``self.propose`` into ``(new_spins, propose_ratio)``, with
        ``propose_ratio`` set to None when the proposer returns ``new_spins`` only.
        """
        proposal = self.propose(key, old_spins)
        if isinstance(proposal, tuple):
            return proposal
        return proposal, None

    @partial(eqx.filter_jit, donate="all-except-first")
    def _update(
        self,
        key: Key,
        propose_ratio: jax.Array | None,
        old_samples: Samples,
        new_samples: Samples,
    ) -> Samples:
        if new_samples.psi is None or old_samples.psi is None:
            raise ValueError("samples.psi should not be None.")

        nsamples, Nmodes = old_samples.spins.shape
        ratio = jnp.asarray(new_samples.psi / old_samples.psi)
        rate_accept = jnp.abs(ratio) ** self._reweight
        if propose_ratio is not None:
            rate_accept *= propose_ratio
        rate_reject = 1.0 - jr.uniform(key, (nsamples,), rate_accept.dtype)
        was_zero = jnp.abs(jnp.asarray(old_samples.psi)) == 0.0
        accepted = (rate_accept > rate_reject) | was_zero

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
        thermal_steps: int | None = None,
        sweep_steps: int | None = None,
        initial_spins: jax.Array | None = None,
    ):
        r"""
        :param samplers:
            The component metropolis samplers to be mixed. In every sweep step, one
            of them is randomly chosen to propose new configurations, with probability
            proportional to its ``nsamples``. All component samplers must share the
            same ``state`` and ``reweight`` factor, otherwise a ``ValueError`` is
            raised. The number of samples generated per iteration is the sum of the
            ``nsamples`` of all component samplers.

        :param thermal_steps:
            The number of thermalization steps in the beginning of each Markov chain,
            default to be 20 * fock state length.

        :param sweep_steps:
            The number of steps for generating new samples, default to be 2 * fock state length.

        :param initial_spins:
            The initial spins for every Markov chain before the thermalization steps.
            By default, the first run inherits the already-thermalized spins of the
            component samplers; otherwise the spins are random.
        """
        state = samplers[0].state
        reweight = float(samplers[0].reweight)
        for sampler in samplers[1:]:
            if sampler.state is not state:
                raise ValueError(
                    "The states of component samplers should be the same in `MixSampler`."
                )
            if float(sampler.reweight) != reweight:
                raise ValueError(
                    "The reweight factors of component samplers should be the same in "
                    "`MixSampler`."
                )

        self._samplers = tuple(samplers)
        nsamples = np.array([sampler.nsamples for sampler in samplers])
        total_nsamples = np.sum(nsamples)
        self._ratio = to_replicated_array(nsamples / total_nsamples)

        keys = [sampler.update_mode.keys() for sampler in self._samplers]
        common_keys = set.intersection(*map(set, keys))
        self._update_mode = {}
        for key in common_keys:
            value = self._samplers[0].update_mode[key]
            same = all(s.update_mode[key] == value for s in self._samplers[1:])
            self._update_mode[key] = value if same else None

        super().__init__(
            state, total_nsamples, reweight, thermal_steps, sweep_steps, initial_spins
        )

    @property
    def particle_type(self) -> tuple[PARTICLE_TYPE, ...]:
        particle_types = [sampler.particle_type for sampler in self._samplers]
        return tuple(set.intersection(*map(set, particle_types)))

    @property
    def update_mode(self) -> dict[str, Any]:
        return self._update_mode

    @property
    def use_ref(self) -> bool:
        """
        Whether to use reference implementation for local updates.
        """
        return all(sampler.use_ref for sampler in self._samplers)

    def reset(
        self, nsweeps: int | None = None, initial_spins: jax.Array | None = None
    ) -> None:
        """
        Reset all Markov chains and thermalize them.

        :param nsweeps:
            Number of sweeps for thermalizing the new samples, default to be
            ``self._thermal_steps``

        :param initial_spins:
            The initial spins for every Markov chain before the thermalization steps,
            default to be random.
        """
        if initial_spins is None and not hasattr(self, "_spins"):
            # load sub-sampler spins in the first run
            ndevices = jax.device_count()
            Nmodes = get_sites().Nmodes
            s = [spl._spins.reshape(ndevices, -1, Nmodes) for spl in self._samplers]
            s = jnp.concatenate(s, axis=1)
            initial_spins = s.reshape(-1, Nmodes)

        super().reset(nsweeps, initial_spins)

    @eqx.filter_jit
    def _rand_sampler_idx(self, key: Key, num: int | None = None) -> jax.Array:
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
        psi = self._state.fast_forward(self._spins)
        samples = Samples(self._spins, psi)

        keys_propose = get_subkeys(nsweeps)
        keys_update = get_subkeys(nsweeps)
        for i_sampler, keyp, keyu in zip(idx_samplers, keys_propose, keys_update):
            sampler = self._samplers[i_sampler]
            new_spins, propose_ratio = sampler._propose_spins_and_ratio(
                keyp, samples.spins
            )

            is_updated = jnp.any(samples.spins != new_spins, axis=1)
            size = _get_update_size(is_updated, chunk_size).item()
            s_updated, idx = _get_updated_spins(new_spins, is_updated, size)
            new_psi = self._state.fast_forward(s_updated)
            new_psi = _get_new_psi(samples.psi, new_psi, is_updated, idx)
            new_samples = Samples(new_spins, new_psi)
            samples = self._update(keyu, propose_ratio, samples, new_samples)

        return Samples(samples.spins, samples.psi)

    def _partial_sweep(self, nsweeps: int, spins: jax.Array) -> Samples:
        """
        Generate new samples for a given set of initial spins.
        """
        idx_samplers = self._rand_sampler_idx(get_subkeys(), nsweeps)
        if self.use_ref:
            psi, state_internal = self._state.init_internal(spins)
        else:
            psi = self._state.fast_forward(spins)
            state_internal = None
        samples = Samples(spins, psi, state_internal)

        keys_propose = get_subkeys(nsweeps)
        keys_update = get_subkeys(nsweeps)
        if self.use_ref:
            sweep_fn = [sampler._single_sweep_ref for sampler in self._samplers]
        else:
            sweep_fn = [sampler._single_sweep_direct for sampler in self._samplers]
        for i_sampler, keyp, keyu in zip(idx_samplers, keys_propose, keys_update):
            samples = sweep_fn[i_sampler](keyp, keyu, samples)

        return Samples(samples.spins, samples.psi)
