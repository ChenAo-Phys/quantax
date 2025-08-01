import numpy as np
import jax.numpy as jnp
from .sr import QNGD
from .time_evol import TimeEvol
from ..state import Variational
from ..sampler import Sampler
from ..utils import DataTracer


class Driver():
    def __init__(
        self,
        state: Variational,
        sampler: Sampler,
        optimizer: QNGD,
        step_length: float,
    ) -> None:
        self._state = state
        self._sampler = sampler
        self._optimizer = optimizer
        self._step_length = step_length
        self._time = 0.0
        self.energy = DataTracer()
        self.VarE = DataTracer()

    def step(self) -> None:
        """iteration"""

class Euler(Driver):
    """First order Euler driver"""
    def step(self) -> None:
        samples = self._sampler.sweep()
        step = self._optimizer.get_step(samples)
        self._state.update(self._step_length * step)
        self._time += self._step_length
        self.energy.append(self._optimizer.energy, self._time)
        self.VarE.append(self._optimizer.VarE, self._time)


class AdaptiveHeunEvolution(Driver):
    """Adaptive second order Heun driver, designed for unitary time-evolution"""
    def __init__(
        self,
        state: Variational,
        sampler: Sampler,
        tdvp: TimeEvol,
        step_length: float = 1e-3,
        integ_threshold: float = 1e-3,
    ):
        super().__init__(state, sampler, tdvp, step_length)
        self._integ_threshold = integ_threshold
        self.step_size = DataTracer()

    def step(self) -> None:
        tdvp: TimeEvol = self._optimizer
        dt = self._step_length

        samples = self._sampler.sweep()
        stepi = tdvp.get_step(samples)
        self._state.update(dt * stepi, rescale=False)
        dtheta0_i = 1j * tdvp.energy - jnp.dot(tdvp._Omean, stepi)
        
        samples = self._sampler.sweep()
        stepf = tdvp.get_step(samples)
        step1 = (stepi + stepf) / 2
        self._state.update(-dt / 2 * stepi, rescale=False)

        samples = self._sampler.sweep()
        stepm = tdvp.get_step(samples)
        self._state.update(dt / 4 * (stepm - stepi))
        dtheta0_m = 1j * tdvp.energy - jnp.dot(tdvp._Omean, stepm)

        samples = self._sampler.sweep()
        stepmm = tdvp.get_step(samples)
        self._state.update(dt / 2 * stepmm, rescale=False)
        dtheta0_mm = 1j * tdvp.energy - jnp.dot(tdvp._Omean, stepmm)

        samples = self._sampler.sweep()
        Smat, Fvec = tdvp.get_SF(samples)
        stepff = tdvp.solve(Smat, Fvec)
        self._state.update(dt / 4 * (stepff - stepmm), rescale=True)
        dtheta0_ff = 1j * tdvp.energy - jnp.dot(tdvp._Omean, stepff)
        step2 = (stepi + stepm + stepmm + stepff) / 4

        dtheta0 = dt * (dtheta0_i + dtheta0_m + dtheta0_mm + dtheta0_ff) / 4
        self._state.rescale(dtheta0)

        diff = step1 - step2
        new_err = jnp.einsum("k,kq,q", diff.conj(), Smat, diff).real
        new_err = jnp.sqrt(new_err) * dt
        ratio = (self._integ_threshold / new_err) ** (1 / 3)
        ratio = np.clip(ratio, 0.2, 2)
        new_step_length = np.clip(dt * ratio, 1e-4, 1e-2)
        self._time += new_step_length
        self._step_length = new_step_length

        self.step_size.append(new_step_length, self._time)
        self.energy.append(self._optimizer.energy, self._time)
        self.VarE.append(self._optimizer.VarE, self._time)
