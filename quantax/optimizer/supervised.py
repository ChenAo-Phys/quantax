from typing import Callable
import jax
import jax.numpy as jnp
from ..state import State, Variational
from ..sampler import Samples
from ..global_defs import get_default_dtype, is_default_cpl


class Supervised:
    def __init__(self, state: Variational, target_state: State, solver: Callable):
        self._state = state
        self._target_state = target_state
        self._solver = solver
        self._theta0_initialized = False

    def target_eval(self, spins) -> jax.Array:
        return self._target_state(spins)

    @staticmethod
    def _get_step0(ratio: jax.Array) -> jax.Array:
        absr = jnp.abs(ratio)
        absrmean = jnp.mean(absr)
        step0 = -jnp.log(absrmean)

        if is_default_cpl():
            unitr = ratio / absr
            angle = jnp.angle(jnp.sum(unitr))
            step0 -= 1j * angle
        return step0

    # def initialize_for_theta0(self, samples: Samples) -> Samples:
    #     phi = self.target_eval(samples.spins)
    #     psi = samples.wave_function
    #     ratio = phi / psi
    #     step0 = self._get_step0(ratio * samples.reweight_factor)
    #     self._state.update_theta0(step0)
    #     new_psi = psi * jnp.exp(-step0)
    #     new_samples = Samples(samples.spins, new_psi, samples.reweight_factor)
    #     self._theta0_initialized = True
    #     return new_samples

    def get_epsilon(self, samples: Samples) -> jax.Array:
        phi = self.target_eval(samples.spins)
        psi = samples.wave_function
        ratio = phi / psi
        reweight = samples.reweight_factor
        self._step0 = self._get_step0(ratio * reweight)

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
        # if not self._theta0_initialized:
        #    samples = self.initialize_for_theta0(samples)

        epsilon = self.get_epsilon(samples)
        Obar = self.get_Obar(samples)
        step = self._solver(Obar, epsilon)
        if isinstance(step, tuple):
            step, self._solver_info = step

        step = jnp.insert(step, 0, self._step0)
        return step.astype(get_default_dtype())