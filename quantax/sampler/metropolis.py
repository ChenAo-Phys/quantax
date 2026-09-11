from __future__ import annotations
from typing import Sequence, Any, Callable
from jaxtyping import Key, ArrayLike
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
    to_distributed_array,
    to_replicated_array,
    rand_states,
    filter_tree_map,
    chunk_map,
    isnan,
)


class Metropolis(Sampler):
    """
    Abstract class for metropolis samplers.
    The samples are equally distributed on different machines.

    Every Markov chain takes exactly ``sweep_steps`` Metropolis steps per sweep.
    A proposal that leaves the configuration unchanged, e.g. exchanging two equal
    spins, is accepted with rate 1, so it is counted as a step without evaluating
    the wavefunction. The wavefunction is evaluated in batches only for the
    proposals that change the configurations, hence the number of batched
    evaluations per sweep is usually smaller than ``sweep_steps``.

    For `~quantax.state.Variational` states the whole sweep runs on the device in
    one jitted call (`_sweep_variational`); other states are swept by a
    host loop that dispatches one round at a time (`_sweep_host`).
    """

    #: Number of candidate proposals drawn per chain in every round of
    #: `_propose_free_steps`. ``None`` picks ``ceil(log2(nsamples)) + 3``: if a
    #: proposal changes the configuration with probability 1/2, on average 1/8 chain
    #: per round finds no changed candidate and only takes free steps in that round.
    #: Proposers that always change the configuration should set it to 1, in which
    #: case every round is exactly one step per chain.
    _n_candidates: int | None = None

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
        self._reweight_factor = None

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
        elif nsweeps <= 0:
            return Samples(self._spins, None, None, self._reweight_factor)

        attr = "ref_chunk" if self.use_ref else "forward_chunk"
        chunk_size = getattr(self._state, attr, None)
        ns = self.nsamples // jax.device_count()

        if chunk_size is not None and chunk_size < ns:
            # Sweep the Markov chains chunk by chunk, so that at most ``chunk_size``
            # chains per device are alive in the sweep loop at a time.
            fn_sweep = chunk_map(
                self._partial_sweep, in_axes=(None, 0), chunk_size=chunk_size
            )
            samples = fn_sweep(nsweeps, self._spins)
        else:
            samples = self._partial_sweep(nsweeps, self._spins)

        self._spins = samples.spins
        self._psi = samples.psi  # Not reusable at next iteration, as state might change
        self._reweight_factor = self._get_reweight_factor(samples.psi)
        return Samples(self._spins, None, None, self._reweight_factor)

    def _partial_sweep(self, nsweeps: int, spins: jax.Array) -> Samples:
        """
        Generate new samples for a given set of initial spins, on the device for
        `~quantax.state.Variational` states and by a host loop otherwise.
        """
        key = get_subkeys()
        # a traced numpy scalar: no recompilation for every ``nsweeps`` and no
        # separate device transfer, it is shipped with the call arguments
        nsweeps_arr = np.asarray(nsweeps, dtype=np.int32)
        if isinstance(self._state, Variational):
            return self._sweep_variational(key, nsweeps_arr, spins, self._state.model)
        else:
            return self._sweep_host(key, nsweeps_arr, spins)

    def _sweep_host(self, key: Key, nsweeps: ArrayLike, spins: jax.Array) -> Samples:
        """
        Generate new samples for a given set of initial spins by a host loop.

        Every chain takes exactly ``nsweeps`` Metropolis steps. Each round of the
        host loop below draws one configuration-changing proposal per chain through
        `_propose_free_steps`, counting the unchanged proposals encountered on the
        way as free steps, and evaluates the wavefunction of all proposals in a
        single batch. The chains therefore progress by different numbers of steps
        per round. A chain that has completed its ``nsweeps`` steps stays in the
        batch with a no-op proposal to keep the batch size fixed, and is never
        updated again, so the returned configuration is the one after exactly
        ``nsweeps`` steps.

        The loop runs on the host, so its per-round work is kept to the three
        dispatches of proposal, wavefunction and acceptance: the random key is
        advanced inside the jitted proposal and acceptance routines and carried
        from round to round, the wavefunction routine is chosen once before the
        loop, and the ``done`` flag returned by the proposal is the only host
        synchronization.
        """
        state = self._state
        if self.use_ref:
            psi, state_internal = state.init_internal(spins)
        else:
            psi = state.fast_forward(spins)
            state_internal = None
        samples = Samples(spins, psi, state_internal)

        def forward(new_spins: jax.Array, samples: Samples) -> Samples:
            if self.use_ref:
                new_psi, state_internal = state.ref_forward(
                    new_spins,
                    samples.spins,
                    self.update_mode,
                    samples.state_internal,
                    return_update=True,
                )
                return Samples(new_spins, new_psi, state_internal)
            else:
                return Samples(new_spins, state.fast_forward(new_spins))

        steps = to_distributed_array(jnp.zeros(spins.shape[0], jnp.int32))
        done = False
        while not done:
            key, new_spins, ratio, steps, done = self._propose_free_steps(
                key, samples.spins, steps, nsweeps
            )
            new_samples = forward(new_spins, samples)
            key, samples = self._update(key, ratio, samples, new_samples)

        return Samples(samples.spins, samples.psi)

    @eqx.filter_jit
    def _sweep_variational(
        self, key: Key, nsweeps: ArrayLike, spins: jax.Array, model: Callable
    ) -> Samples:
        """
        The whole loop of rounds of `_sweep_host` run on the device in a single
        jitted call, for `~quantax.state.Variational` states.

        The jittable forward passes of the state are called with ``model`` passed
        explicitly, so updating the variational parameters doesn't trigger a
        recompilation. A ``while_loop`` runs the rounds until every chain has
        completed its steps. The number of rounds is traced, so no host
        synchronization happens during the sweep, and the ``done`` reduction of a
        round is only consumed by the loop condition after the wavefunction
        evaluation of the same round, which lets XLA overlap the two.
        """
        state = self._state
        assert isinstance(state, Variational)
        use_ref = self.use_ref

        if use_ref:
            psi, state_internal = state._init_internal(model, spins)
        else:
            psi = state._fulljit_forward(model, spins)
            state_internal = None
        samples = Samples(spins, psi, state_internal)

        def forward(new_spins: jax.Array, samples: Samples) -> Samples:
            if use_ref:
                new_psi, state_internal = state._ref_forward(
                    model,
                    new_spins,
                    samples.spins,
                    self.update_mode,
                    samples.state_internal,
                    True,
                )
                return Samples(new_spins, new_psi, state_internal)
            else:
                return Samples(new_spins, state._fulljit_forward(model, new_spins))

        steps = jnp.zeros(spins.shape[0], jnp.int32)
        nsweeps = jnp.asarray(nsweeps, steps.dtype)

        def round_cond(carry):
            done = carry[3]
            return ~done

        def round_body(carry):
            key, samples, steps, done = carry
            key, new_spins, ratio, steps, done = self._propose_free_steps(
                key, samples.spins, steps, nsweeps
            )
            new_samples = forward(new_spins, samples)
            key, samples = self._update(key, ratio, samples, new_samples)
            return key, samples, steps, done

        done = jnp.all(steps >= nsweeps)  # only for nsweeps <= 0
        carry = (key, samples, steps, done)
        key, samples, steps, done = jax.lax.while_loop(round_cond, round_body, carry)

        return Samples(samples.spins, samples.psi)

    @eqx.filter_jit
    def _propose_free_steps(
        self, key: Key, spins: jax.Array, steps: jax.Array, nsweeps: ArrayLike
    ) -> tuple[Key, jax.Array, jax.Array | None, jax.Array, jax.Array]:
        """
        Draw candidate proposals for every chain and pick the first one that changes
        its configuration, counting the unchanged candidates before it as free steps.

        A proposal identical to the current configuration is accepted with rate 1,
        so it is counted as a step of the chain right away without evaluating the
        wavefunction. Every chain draws ``self._n_candidates`` independent proposals
        at once. A chain whose candidates are all unchanged only takes free steps in
        this round, and a chain that has completed its ``nsweeps`` steps takes no
        step; both keep their own configuration as a no-op proposal, so that the
        batch size of the subsequent wavefunction evaluation stays fixed. Nothing is
        iterated until all chains hold a changed proposal, hence the only
        cross-device reduction is the ``done`` flag.

        :param key:
            The random key carried through the sweep, advanced on the device.

        :param spins:
            The current configurations of the chains.

        :param steps:
            The number of steps taken so far by every chain.

        :param nsweeps:
            The total number of steps to take in every chain.

        :return:
            A tuple ``(key, new_spins, propose_ratio, steps, done)``. ``key`` is the
            advanced random key. ``new_spins`` are the proposals to be evaluated and
            ``propose_ratio`` their proposal ratios (None if ``self.propose`` doesn't
            provide them). ``steps`` includes the free steps and the pending step of
            the proposals to be evaluated, and ``done`` tells whether every chain has
            completed its ``nsweeps`` steps after the pending step.
        """
        K = self._n_candidates
        if K is None:
            K = int(np.ceil(np.log2(spins.shape[0]))) + 3
        # keep every step count in the dtype of ``steps``, so that the carry of the
        # device loop isn't promoted (e.g. to int64 by argmax or scalars under x64)
        nsweeps = jnp.asarray(nsweeps, steps.dtype)
        key, key_propose = jr.split(key)

        propose = jax.vmap(lambda k: self._propose_spins_and_ratio(k, spins))
        candidates, candidate_ratios = propose(jr.split(key_propose, K))
        changed = jnp.any(candidates != spins[None], axis=2)  # (K, nsamples)
        any_changed = jnp.any(changed, axis=0)
        # index of the first changed candidate
        first = jnp.argmax(changed, axis=0).astype(steps.dtype)
        remaining = nsweeps - steps  # 0 for the chains that have completed the sweep
        # the unchanged candidates before the first changed one are free steps,
        # capped by the remaining steps of the chain
        free = jnp.minimum(jnp.where(any_changed, first, K), remaining)
        # keep the first changed candidate if its step is still within the sweep
        take = any_changed & (first < remaining)
        chosen = jnp.take_along_axis(candidates, first[None, :, None], axis=0)[0]
        new_spins = jnp.where(take[:, None], chosen, spins)
        if candidate_ratios is None:
            ratio = None
        else:
            chosen_ratio = jnp.take_along_axis(candidate_ratios, first[None], axis=0)[0]
            ratio = jnp.where(take, chosen_ratio, 1)

        steps = steps + free + take
        done = jnp.all(steps >= nsweeps)
        return key, new_spins, ratio, steps, done

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
    ) -> tuple[Key, Samples]:
        """
        Accept or reject the proposed samples by the Metropolis-Hastings rule.

        :param key:
            The random key carried through the sweep, advanced on the device.

        :return:
            A tuple of the advanced random key and the updated samples.
        """
        if new_samples.psi is None or old_samples.psi is None:
            raise ValueError("samples.psi should not be None.")

        key, key_accept = jr.split(key)
        nsamples, Nmodes = old_samples.spins.shape
        ratio = jnp.asarray(new_samples.psi / old_samples.psi)
        rate_accept = jnp.abs(ratio) ** self._reweight
        if propose_ratio is not None:
            rate_accept *= propose_ratio
        rate_reject = jr.uniform(key_accept, (nsamples,), rate_accept.dtype)  # [0, 1)

        # Table for special acceptance conditions ("*": needs special rules):
        # old\new   0   1   nan inf
        # 0         Y*  Y   N   Y
        # 1         N   Y   N   Y
        # nan       Y*  Y*  ?   Y*
        # inf       N   N   N   Y*

        special_cond = jnp.isnan(rate_accept) & ~isnan(new_samples.psi)
        accepted = (rate_accept > rate_reject) | special_cond

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

        return key, filter_tree_map(f_select, new_samples, old_samples)


class MixSampler(Metropolis):
    r"""
    A mixture of several metropolis samplers. Every proposal is generated by a
    randomly chosen component sampler.
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
            The component metropolis samplers to be mixed. For every proposal, one
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

        # Candidates per round: the default (None) if any component needs it,
        # otherwise the largest requirement among the components.
        n_candidates = [sampler._n_candidates for sampler in self._samplers]
        if any(n is None for n in n_candidates):
            self._n_candidates = None
        else:
            self._n_candidates = max(n for n in n_candidates if n is not None)

        # The proposals evaluated in one batch come from different components, as
        # every candidate slot of `_propose_free_steps` draws its own component, so
        # the flip numbers are merged into their upper bounds over the components.
        # The states treat them as upper bounds with no-op fills.
        keys = [sampler.update_mode.keys() for sampler in self._samplers]
        common_keys = set.intersection(*map(set, keys))
        self._update_mode = {}
        for key in common_keys:
            values = [sampler.update_mode[key] for sampler in self._samplers]
            if all(value == values[0] for value in values):
                self._update_mode[key] = values[0]
            elif key in ("nflips", "nflips_up", "nflips_dn"):
                if state.use_ref:
                    warn(
                        f"The update mode '{key}' differs among the component "
                        f"samplers and is merged into its maximum {max(values)}. "
                        f"The local updates of the state may be less efficient "
                        f"than necessary."
                    )
                self._update_mode[key] = max(values)
            else:
                if state.use_ref:
                    warn(
                        f"The update mode '{key}' differs among the component "
                        f"samplers and can't be merged, so it is set to None."
                    )
                self._update_mode[key] = None

        super().__init__(
            state, total_nsamples, reweight, thermal_steps, sweep_steps, initial_spins
        )

    @property
    def particle_type(self) -> tuple[PARTICLE_TYPE, ...]:
        particle_types = [sampler.particle_type for sampler in self._samplers]
        return tuple(set.intersection(*map(set, particle_types)))

    @property
    def update_mode(self) -> dict[str, Any]:
        """
        The update mode of local updates generated in the sampler. The modes shared
        by all component samplers are kept, with integer modes merged into their
        upper bounds over the components.
        """
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
    def _propose_spins_and_ratio(
        self, key: Key, old_spins: jax.Array
    ) -> tuple[jax.Array, jax.Array | None]:
        """
        Propose new configurations by a component sampler chosen independently for
        every chain, with probability proportional to its ``nsamples``. All
        components propose and the proposals are selected afterwards, as this
        routine is vmapped over the candidate slots in `_propose_free_steps` where
        a ``lax.switch`` would evaluate all branches anyway. ``propose_ratio`` is
        None if no component provides it, otherwise the components without
        proposal ratios are assigned the ratio 1.
        """
        nsamples = old_spins.shape[0]
        n_samplers = len(self._samplers)
        key_choice, *keys = jr.split(key, n_samplers + 1)
        idx = jr.choice(key_choice, n_samplers, (nsamples,), p=self._ratio)

        new_spins = []
        ratios = []
        for sampler, key in zip(self._samplers, keys):
            new_spins_i, ratio_i = sampler._propose_spins_and_ratio(key, old_spins)
            new_spins.append(new_spins_i)
            ratios.append(ratio_i)

        # chain i takes the proposal of component idx[i]
        new_spins = jnp.stack(new_spins)
        new_spins = jnp.take_along_axis(new_spins, idx[None, :, None], axis=0)[0]
        if all(r is None for r in ratios):
            return new_spins, None

        dtype = jnp.result_type(*[r for r in ratios if r is not None])
        ratios = [
            jnp.ones(nsamples, dtype) if r is None else r.astype(dtype) for r in ratios
        ]
        ratios = jnp.stack(ratios)
        ratio = jnp.take_along_axis(ratios, idx[None, :], axis=0)[0]
        return new_spins, ratio
