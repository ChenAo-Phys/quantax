from typing import Optional
import jax
import jax.numpy as jnp
import jax.random as jr

from .status import SamplerStatus, Samples
from ..state import State
from ..symmetry import Symmetry
from ..utils import ints_to_array, rand_spins
from ..global_defs import get_subkeys


class Sampler:
    """Abstract class for sampler"""

    def __init__(self, state: State, nsamples: int, reweight: float = 2.0):
        self._state = state
        self._nsamples = nsamples
        self._reweight = reweight
        self._status = SamplerStatus()

    @property
    def state(self) -> State:
        return self._state

    @property
    def nsites(self) -> int:
        return self._state.nsites

    @property
    def nsamples(self) -> int:
        return self._nsamples

    @property
    def reweight(self) -> float:
        return self._reweight

    @property
    def current_spins(self) -> Optional[jax.Array]:
        return self._status.spins

    @property
    def current_wf(self) -> Optional[jax.Array]:
        return self._status.wave_function

    @property
    def current_prob(self) -> Optional[jax.Array]:
        return self._status.prob

    def sweep(self) -> Samples:
        """Generate samples. Return samples with sampled spin configuration,
        wave function and rescale factor"""


class ExactSampler(Sampler):
    """Exact sampling"""

    def __init__(
        self,
        state: State,
        nsamples: int,
        reweight: float = 2.0,
        symm: Optional[Symmetry] = None,
    ):
        super().__init__(state, nsamples, reweight)
        self._symm = symm if symm is not None else state.symm

    def sweep(self) -> Samples:
        state = self._state.todense(self._symm)
        prob = jnp.abs(state.wave_function) ** self._reweight
        basis = self._symm.basis
        basis_ints = basis.states.copy()
        basis_ints = basis_ints[prob > 0.0]
        prob = prob[prob > 0.0]
        basis_ints = jr.choice( # works only for one node
            get_subkeys(), basis_ints, shape=(self.nsamples,), p=prob
        )
        spins = ints_to_array(basis_ints)

        spins = self._symm.get_symm_spins(spins)
        idx = jr.choice(get_subkeys(), spins.shape[1], shape=(spins.shape[0],))
        arange = jnp.arange(spins.shape[0])
        spins = spins[arange, idx]
        wf = state(spins)
        prob = jnp.abs(wf) ** self._reweight
        self._status = SamplerStatus(spins, wf, prob)
        return Samples(spins, wf, self._reweight)


class RandomSampler(Sampler):
    def __init__(self, state: State, nsamples: int):
        super().__init__(state, nsamples, reweight=0.0)

    def sweep(self) -> Samples:
        spins = rand_spins(self.nsamples, self.state.total_sz)
        wf = self._state(spins)
        prob = jnp.ones_like(wf)
        self._status = SamplerStatus(spins, wf, prob)
        return Samples(spins, wf, self._reweight)
