import numpy as np
import jax.numpy as jnp
from .tdvp import TDVP, TimeEvol
from ..state import Variational
from ..sampler import Sampler
from ..utils import DataTracer


class Driver():
    def __init__(
        self,
        state: Variational,
        sampler: Sampler,
        tdvp: TDVP,
        step_length: float,
    ) -> None:
        self._state = state
        self._sampler = sampler
        self._tdvp = tdvp
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
        step = self._tdvp.get_step(samples)
        self._state.update(self._step_length * step)
        self._time += self._step_length
        self.energy.append(self._tdvp.energy, self._time)
        self.VarE.append(self._tdvp.VarE, self._time)


class AdaptiveHeunEvolution(Driver):
    """Adaptive second order Heun driver, designed for unitary time-evolution"""
    def __init__(
        self,
        state: Variational,
        sampler: Sampler,
        tdvp: TimeEvol,
        step_length: float,
        integ_threshold: float = 1e-3,
    ):
        super().__init__(state, sampler, tdvp, step_length)
        self._integ_threshold = integ_threshold
        self.step_size = DataTracer()

    def step(self) -> None:
        tdvp: TimeEvol = self._tdvp

        samples = self._sampler.sweep()
        stepi = tdvp.get_step(samples)
        self._state.update(self._step_length * stepi, rescale=False)
        
        samples = self._sampler.sweep()
        stepf = tdvp.get_step(samples)
        self._state.update(-self._step_length / 2 * stepi, rescale=False)
        step1 = (stepi + stepf) / 2

        samples = self._sampler.sweep()
        stepm = tdvp.get_step(samples)
        self._state.update(self._step_length / 4 * (stepm - stepi))

        samples = self._sampler.sweep()
        stepmm = tdvp.get_step(samples)
        self._state.update(self._step_length / 2 * stepmm, rescale=False)

        samples = self._sampler.sweep()
        Smat, Fvec = tdvp.get_SF(samples)
        stepff = tdvp.solve(Smat, Fvec)
        self._state.update(self._step_length / 4 * (stepff - stepmm), rescale=False)
        step2 = (stepi + stepm + stepmm + stepff) / 4

        diff = step1 - step2
        new_err = jnp.einsum("k,kq,q", diff.conj(), Smat, diff).real
        new_err = jnp.sqrt(new_err) * self._step_length
        ratio = (self._integ_threshold / new_err) ** (1 / 3)
        ratio = np.clip(ratio, 0.2, 2)
        new_step_length = np.clip(self._step_length * ratio, 1e-4, 1e-2)
        self._time += new_step_length
        self._step_length = new_step_length

        self.step_size.append(new_step_length, self._time)
        self.energy.append(self._tdvp.energy, self._time)
        self.VarE.append(self._tdvp.VarE, self._time)
