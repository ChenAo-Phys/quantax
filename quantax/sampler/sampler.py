from typing import Optional
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr

from .samples import Samples
from ..state import State
from ..symmetry import Symmetry
from ..utils import ints_to_array, rand_states, PsiArray
from ..global_defs import get_subkeys


class Sampler:
    """Abstract class for samplers"""

    def __init__(self, state: State, nsamples: int, reweight: float = 2.0):
        r"""
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

    @property
    def state(self) -> State:
        """The state used for computing the wave function and probability"""
        return self._state

    @property
    def Nsites(self) -> int:
        return self.state.Nsites

    @property
    def Nmodes(self) -> int:
        return self.state.Nmodes

    @property
    def nsamples(self) -> int:
        """Number of samples generated per iteration"""
        return self._nsamples

    @property
    def reweight(self) -> float:
        r"""The reweight factor n defining the sample probability :math:`|\psi|^n`"""
        return self._reweight

    def sweep(self) -> Samples:
        """Generate new samples"""
        return NotImplemented

    @partial(jax.jit, static_argnums=0)
    def _get_reweight_factor(self, psi: PsiArray) -> jax.Array:
        reweight_factor = abs(psi) ** (2 - self._reweight)
        return jnp.asarray(reweight_factor / reweight_factor.mean())


class ExactSampler(Sampler):
    """Generate samples directly from exact probability"""

    def __init__(
        self,
        state: State,
        nsamples: int,
        reweight: float = 2.0,
        symm: Optional[Symmetry] = None,
    ):
        r"""
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
        prob = jnp.abs(state.psi) ** self._reweight
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
        psi = state(spins)
        prob = abs(psi) ** self._reweight

        return Samples(spins, psi, None, self._get_reweight_factor(psi))


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
        spins = rand_states(self.nsamples)
        psi = self._state(spins)
        return Samples(spins, psi, None, self._get_reweight_factor(psi))
