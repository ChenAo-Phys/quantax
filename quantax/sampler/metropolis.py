from typing import Optional, Tuple, Union, Sequence
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
from .sampler import Sampler
from .status import SamplerStatus, Samples
from ..state import State
from ..global_defs import get_subkeys, get_sites, get_default_dtype
from ..utils import to_array_shard, rand_spins


class Metropolis(Sampler):
    def __init__(
        self,
        state: State,
        nsamples: int,
        reweight: float = 2.0,
        thermal_steps: Optional[int] = None,
        sweep_steps: Optional[int] = None,
        initial_spins: Optional[jax.Array] = None,
    ):
        if nsamples % jax.local_device_count():
            raise ValueError(
                "'nsamples' should be a multiple of the number of devices, got "
                f"{nsamples} samples and {jax.local_device_count()} devices."
            )
        super().__init__(state, nsamples, reweight)

        if thermal_steps is None:
            self._thermal_steps = 20 * self.nsites
        else:
            self._thermal_steps = thermal_steps
        if sweep_steps is None:
            self._sweep_steps = 2 * self.nsites
        else:
            self._sweep_steps = sweep_steps
        self._initial_spins = initial_spins
        self.reset()

    @property
    def nsites(self) -> int:
        return self.state.nsites

    def reset(self) -> None:
        if self._initial_spins is None:
            spins = rand_spins(self.nsamples, self.state.total_sz)
        else:
            if self._initial_spins.ndim == 1:
                spins = jnp.tile(self._initial_spins, (self.nsamples, 1))
            else:
                spins = self._initial_spins.reshape(self.nsamples, self.nsites)
        spins = to_array_shard(spins)

        spins, propose_prob = self._propose(get_subkeys(), spins)
        self._status = SamplerStatus(spins, propose_prob=propose_prob)
        self.sweep(self._thermal_steps)

    def sweep(self, nsweeps: Optional[int] = None) -> Samples:
        spins = self.current_spins
        wf = self._state(spins)
        prob = jnp.abs(wf) ** self._reweight
        self._status = SamplerStatus(spins, wf, prob, self._status.propose_prob)

        if nsweeps is None:
            nsweeps = self._sweep_steps
        keys_propose = get_subkeys(nsweeps)
        keys_update = get_subkeys(nsweeps)
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
        """Propose new spin configurations, return spin and proposal weight"""

    @staticmethod
    @jax.jit
    def _update(
        key: jax.Array, old_status: SamplerStatus, new_status: SamplerStatus
    ) -> SamplerStatus:
        nsamples, nsites = old_status.spins.shape
        dtype = old_status.prob.dtype
        rand = 1.0 - jr.uniform(key, (nsamples,), dtype)
        rate_accept = new_status.prob * old_status.propose_prob
        rate_reject = old_status.prob * new_status.propose_prob * rand
        selected = rate_accept > rate_reject
        selected = jnp.where(old_status.prob == 0., True, selected)

        selected_spins = jnp.tile(selected, (nsites, 1)).T
        spins = jnp.where(selected_spins, new_status.spins, old_status.spins)
        wf = jnp.where(selected, new_status.wave_function, old_status.wave_function)
        prob = jnp.where(selected, new_status.prob, old_status.prob)
        p_prob = jnp.where(selected, new_status.propose_prob, old_status.propose_prob)
        return SamplerStatus(spins, wf, prob, p_prob)


class LocalFlip(Metropolis):
    @partial(jax.jit, static_argnums=0)
    def _propose(
        self, key: jax.Array, old_spins: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        nsamples, nsites = old_spins.shape
        pos = jr.choice(key, nsites, (nsamples,))
        new_spins = old_spins.at[jnp.arange(nsamples), pos].multiply(-1)
        propose_prob = to_array_shard(jnp.ones(nsamples, dtype=get_default_dtype()))
        return new_spins, propose_prob


class NeighborExchange(Metropolis):
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
        if state.total_sz is None:
            raise ValueError("The 'total_sz' of 'state' should be specified.")
        n_neighbor = [n_neighbor] if isinstance(n_neighbor, int) else n_neighbor
        neighbors = get_sites().get_neighbor(n_neighbor)
        self._neighbors = jnp.concatenate(neighbors, axis=0)

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
        propose_prob = to_array_shard(jnp.ones(nsamples, dtype=get_default_dtype()))
        return new_spins, propose_prob
