import numpy as np
import jax
import jax.numpy as jnp

from . import State
from ..sampler import Samples
from ..utils import DataTracer


class StateLS(State):
    """State with a Lanczos step. Not tested yet..."""

    def __init__(self, operator, state: State, Eg: float, E0: float, sigma: float):
        super().__init__(state.symm)
        self._operator = operator
        self._state = state
        self._E0 = E0
        self._sigma = sigma
        self._alpha0 = (Eg - E0) / sigma

        self.energy_data = DataTracer()
        self.var_data = DataTracer()

    def __call__(self, fock_states: jax.Array) -> jax.Array:
        psi = self._state(fock_states)
        samples = Samples(fock_states, psi)
        Eloc = self._operator.Oloc(self._state, samples)
        ElocLS = psi * (1 + self._alpha0 * (Eloc - self._E0)) / self._sigma
        return ElocLS

    def measure(self, samples: Samples) -> None:
        Eloc = self._operator.Oloc(self, samples)
        self.energy_data.append(jnp.mean(Eloc).real)
        self.var_data.append(jnp.mean(jnp.abs(Eloc - self._E0) ** 2))

    def get_stats(self) -> dict:
        alpha0 = self._alpha0
        E0 = self._E0
        sigma = self._sigma

        Ealpha0 = np.mean(self.energy_data)
        mu3 = ((1 + alpha0**2) * (Ealpha0 - E0) / sigma - 2 * alpha0) / alpha0**2
        alpha = (mu3 - np.sqrt(mu3**2 + 4)) / 2
        Ealpha = E0 + sigma * alpha

        var = np.mean(self.var_data) / sigma**2
        mu4 = ((1 + alpha0**2) * var - 2 * alpha0 * mu3 - 1) / alpha0**2
        VarEalpha = (
            sigma**2 * (alpha**2 * (mu4 + 2) - 1) / (1 + alpha**2) - (E0 - Ealpha) ** 2
        )

        return {"alpha": alpha, "E": Ealpha, "VarE": VarEalpha}
