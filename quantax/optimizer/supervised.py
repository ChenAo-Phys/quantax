from typing import Callable
import jax
import jax.numpy as jnp
from ..state import State, Variational
from ..sampler import Samples
from ..global_defs import get_default_dtype


class Supervised:
    def __init__(self, state: Variational, target_state: State, solver: Callable):
        self._state = state
        self._target_state = target_state
        self._solver = solver

    def get_epsilon(self, samples: Samples) -> jax.Array:
        phi = self._target_state(samples.spins)
        psi = samples.wave_function
        ratio = phi / psi
        reweight = samples.reweight_factor

        ratio_mean = jnp.mean(ratio * reweight)
        ratio = ratio / ratio_mean - 1
        epsilon = -ratio * jnp.sqrt(reweight / samples.nsamples)
        return epsilon

    def get_Obar(self, samples: Samples) -> jax.Array:
        Omat = self._state.jacobian(samples.spins).astype(get_default_dtype())
        Omat = Omat.reshape(samples.nsamples, -1)
        # should be pmean here
        self._Omean = jnp.mean(Omat * samples.reweight_factor[:, None], axis=0)
        Omat -= self._Omean[None, :]
        Omat *= jnp.sqrt(samples.reweight_factor / samples.nsamples)[:, None]
        return Omat

    def get_step(self, samples: Samples) -> jax.Array:
        epsilon = self.get_epsilon(samples)
        Obar = self.get_Obar(samples)
        step = self._solver(Obar, epsilon)
        return step.astype(get_default_dtype())