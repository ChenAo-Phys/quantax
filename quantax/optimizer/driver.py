import abc
from numbers import Number

from .tdvp import TDVP
from ..state import Variational
from ..sampler import Sampler
from ..utils import DataTracer


class Driver(abc.ABC):
    def __init__(
        self,
        state: Variational,
        sampler: Sampler,
        tdvp: TDVP,
        step_length: Number,
    ) -> None:
        self._state = state
        self._sampler = sampler
        self._tdvp = tdvp
        self._step_length = step_length
        self._time = 0.0
        self._energy = DataTracer()
        self._VarE = DataTracer()

    @abc.abstractmethod
    def step(self) -> None:
        """iteration"""

    def energy(self) -> DataTracer:
        return self._energy

    def VarE(self) -> DataTracer:
        return self._VarE


class Euler(Driver):
    def step(self) -> None:
        samples = self._sampler.sweep()
        step = self._tdvp.get_step(samples)
        self._state.update(self._step_length * step)
        self._time += self._step_length
        self._energy.append(self._tdvp.energy)
        self._VarE.append(self._tdvp.VarE)
