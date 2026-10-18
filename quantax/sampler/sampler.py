import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from .samples import Samples
from ..state import State
from ..symmetry import Symmetry
from ..utils import (
    ints_to_array,
    rand_states,
    to_replicated_array,
    to_distributed_array,
)
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
        self._reweight = to_replicated_array(reweight)

    @property
    def state(self) -> State:
        """The state used for computing the wave function and probability"""
        return self._state

    @property
    def Nsites(self) -> int:
        """Number of sites"""
        return self.state.Nsites

    @property
    def Nmodes(self) -> int:
        """Number of modes (fock state length)"""
        return self.state.Nmodes

    @property
    def nsamples(self) -> int:
        """Number of samples generated per iteration"""
        return self._nsamples

    @property
    def reweight(self) -> jax.Array:
        r"""The reweight factor n defining the sample probability :math:`|\psi|^n`"""
        return self._reweight

    def sweep(self) -> Samples:
        r"""
        Generate new samples.

        The returned `~quantax.sampler.Samples` carry ``spins`` and ``prob``, the
        probability the samples are actually drawn from, up to a constant. The wave
        function ``psi`` and the reweighting factor ``r`` towards :math:`|\psi|^2`
        are also returned if ``psi`` evaluated in the sampling is consistent with the
        forward pass of the state. Otherwise, they are filled in later, e.g. by
        `~quantax.operator.Operator.Oloc`.
        """
        raise NotImplementedError


class ExactSampler(Sampler):
    """Generate samples directly from exact probability"""

    def __init__(
        self,
        state: State,
        nsamples: int,
        reweight: float = 2.0,
        symm: Symmetry | None = None,
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
        r"""
        Generate new samples by computing the full wave function.

        A representative :math:`s_0` of the symmetry basis is drawn with probability
        :math:`P(s_0) \propto |\psi_G(s_0)|^n`, where
        :math:`\psi_G(s_0) = \psi(s_0) \sqrt{|G s_0|}` is the amplitude of the
        normalized symmetric basis state with :math:`|G s_0|` the size of the orbit of
        :math:`s_0`, and :math:`\psi` is the wave function symmetrized by ``symm``.
        A random symmetry operation then maps it to the sample :math:`s`.
        The returned ``prob`` is the actual sampling probability
        :math:`q_s = P(s_0) m_s / |G|`, with :math:`m_s = |G| / |G s_0|` the number of
        operations mapping :math:`s_0` to :math:`s`, i.e.
        :math:`q_s \propto |\psi(s)|^n |G s|^{n/2 - 1}`, which is :math:`|\psi(s)|^2`
        for ``reweight=2``. The returned ``psi`` is the wave function of the samples
        evaluated by the state itself, so that ``r`` also corrects the difference
        from the symmetrized wave function if the state doesn't carry ``symm``.
        """
        state = self._state.todense(self._symm)
        # The normalization is only applied to the sampling probability, where
        # `|psi| ** reweight` of an unnormalized state easily overflows. The psi of
        # the samples is kept unnormalized, so that it stays consistent with the
        # wave function of connected configurations in `quantax.operator.Operator.Oloc`.
        psi_dense = jnp.asarray(state.normalize().psi)
        # promote psi below single precision to float32 before computing prob
        psi_dense = psi_dense.astype(jnp.promote_types(psi_dense.dtype, jnp.float32))
        prob = jnp.abs(psi_dense) ** self._reweight
        basis = self._symm.basis
        nonzero = prob > 0.0
        basis_ints = basis.states[nonzero]
        prob = prob[nonzero]
        idx = jr.choice(  # works only for one node
            get_subkeys(), prob.size, shape=(self.nsamples,), p=prob
        )
        # Draw indices rather than the uint64 basis integers, which jax would
        # truncate to uint32 for more than 32 modes when x64 is disabled.
        spins = ints_to_array(basis_ints[np.asarray(idx)])
        prob = prob[idx]

        # Apply a random symmetry operation to each representative. A configuration
        # appearing several times among the symmetry images of its representative is
        # drawn more often, so its probability is multiplied by the multiplicity.
        symm_spins = jax.vmap(self._symm.get_symm_spins)(spins)
        nsymm = symm_spins.shape[1]
        idx = jr.choice(get_subkeys(), nsymm, shape=(spins.shape[0],))
        spins = symm_spins[jnp.arange(spins.shape[0]), idx]
        multiplicity = jnp.sum(jnp.all(symm_spins == spins[:, None, :], axis=2), axis=1)
        prob = prob * multiplicity.astype(prob.dtype) / nsymm

        spins = to_distributed_array(spins)
        prob = to_distributed_array(prob)
        psi = self._state(spins)
        samples = Samples(spins, psi=psi, prob=prob)
        samples.update_r()
        return samples


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
        r"""
        Generate new samples with uniform probability. The returned ``prob`` is 1
        for every sample, and ``psi`` is the wave function evaluated by the state.
        """
        spins = rand_states(self.nsamples)
        psi = self._state(spins)
        dtype = jnp.promote_types(jnp.finfo(psi.dtype).dtype, jnp.float32)
        prob = jnp.ones_like(jnp.asarray(psi), dtype=dtype)
        samples = Samples(spins, psi=psi, prob=prob)
        samples.update_r()
        return samples
