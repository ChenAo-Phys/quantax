from typing import Optional
import jax
import jax.numpy as jnp
import jax.random as jr

from .status import SamplerStatus, Samples
from ..state import State
from ..symmetry import Symmetry
from ..utils import ints_to_array, rand_states
from ..global_defs import get_subkeys


class Sampler:
    """Abstract class for samplers"""

    def __init__(self, state: State, nsamples: int, reweight: float = 2.0):
        """
        :param state:
            The state used for computing the wave function and probability

        :param nsamples:
            Number of samples generated per iteration

        :param reweight:
            The reweight factor n defining the sample probability :math:`|\psi|^n`,
            default to 2.0
        """
        if nsamples % jax.device_count() != 0:
            raise ValueError(
                "`nsamples` should be a multiple of the number of devices, but got "
                f"{nsamples} samples and {jax.device_count()} devices."
            )

        self._state = state
        self._nsamples = nsamples
        self._reweight = reweight
        self._status = SamplerStatus()

    @property
    def state(self) -> State:
        """The state used for computing the wave function and probability"""
        return self._state

    @property
    def nsites(self) -> int:
        return self.state.nsites

    @property
    def nstates(self) -> int:
        return self.state.nstates

    @property
    def nsamples(self) -> int:
        """Number of samples generated per iteration"""
        return self._nsamples

    @property
    def reweight(self) -> float:
        """The reweight factor n defining the sample probability :math:`|\psi|^n`"""
        return self._reweight

    @property
    def current_spins(self) -> Optional[jax.Array]:
        """The current spin configurations stored in the sampler"""
        return self._status.spins

    @property
    def current_wf(self) -> Optional[jax.Array]:
        """The wave function of the current spin configurations"""
        return self._status.wave_function

    @property
    def current_prob(self) -> Optional[jax.Array]:
        """The probability of the current spin configurations"""
        return self._status.prob

    def sweep(self) -> Samples:
        """Generate new samples"""
        return NotImplemented


class ExactSampler(Sampler):
    """Generate samples directly from exact probability"""

    def __init__(
        self,
        state: State,
        nsamples: int,
        reweight: float = 2.0,
        symm: Optional[Symmetry] = None,
    ):
        """
        :param state:
            The state used for computing the wave function and probability

        :param nsamples:
            Number of samples generated per iteration

        :param reweight:
            The reweight factor n defining the sample probability :math:`|\psi|^n`,
            default to 2.0

        :param symm:
            The symmetry for computing the full wave function,
            default to the symmetry of the ``state``.
        """
        super().__init__(state, nsamples, reweight)
        self._symm = symm if symm is not None else state.symm

    def sweep(self) -> Samples:
        """
        Generate new samples by computing the full wave function
        """
        state = self._state.todense(self._symm)
        prob = jnp.abs(state.wave_function) ** self._reweight
        basis = self._symm.basis
        basis_ints = basis.states.copy()
        basis_ints = basis_ints[prob > 0.0]
        prob = prob[prob > 0.0]
        basis_ints = jr.choice(  # works only for one node
            get_subkeys(), basis_ints, shape=(self.nsamples,), p=prob
        )
        spins = ints_to_array(basis_ints)

        spins = jax.vmap(self._symm.get_symm_spins)(spins)
        idx = jr.choice(get_subkeys(), spins.shape[1], shape=(spins.shape[0],))
        arange = jnp.arange(spins.shape[0])
        spins = spins[arange, idx]
        wf = state(spins)
        prob = jnp.abs(wf) ** self._reweight
        self._status = SamplerStatus(spins, wf, prob)
        return Samples(spins, wf, self._reweight)


class RandomSampler(Sampler):
    r"""
    Generate random samples with equal probability for all possible spin configurations.
    The reweight factor is 0 because :math:`P \propto |\psi|^0`.
    """

    def __init__(self, state: State, nsamples: int):
        """
        :param state:
            The state used for computing the wave function and probability

        :param nsamples:
            Number of samples generated per iteration
        """
        super().__init__(state, nsamples, reweight=0.0)

    def sweep(self) -> Samples:
        spins = rand_states(self.nsamples, self.state.Nparticle)
        wf = self._state(spins)
        prob = jnp.ones_like(wf)
        self._status = SamplerStatus(spins, wf, prob)
        return Samples(spins, wf, self._reweight)
