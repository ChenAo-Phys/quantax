from typing import Optional
from jax.typing import ArrayLike
import jax
import jax.numpy as jnp

from ..state import State, Variational, DenseState
from ..sampler import Samples
from ..operator import Operator
from ..symmetry import Symmetry
from ..utils import array_extend
from ..global_defs import get_default_dtype


class EnergyGrad:
    r"""
    Energy gradient source defining :math:`\bar \epsilon` from the local energies
    of a Hamiltonian, used for ground-state search and time evolution.
    """

    def __init__(self, hamiltonian: Operator):
        r"""
        :param hamiltonian:
            The Hamiltonian for the evolution.
        """
        self._hamiltonian = hamiltonian
        self._energy = None
        self._VarE = None

    @property
    def hamiltonian(self) -> Operator:
        """The Hamiltonian for the evolution."""
        return self._hamiltonian

    @property
    def energy(self) -> ArrayLike | None:
        """Energy of the current step."""
        return self._energy

    @property
    def VarE(self) -> ArrayLike | None:
        r"""Energy variance :math:`\left< (H - E)^2 \right>` of the current step."""
        return self._VarE

    def ebar(self, state: Variational, samples: Samples) -> jax.Array:
        r"""
        Compute :math:`\bar \epsilon` for given samples. The local energy is
        :math:`E_{loc, s} = \sum_{s'} \frac{\psi_{s'}}{\psi_s} \left< s|H|s' \right>`,
        and :math:`\bar \epsilon` is defined as
        :math:`\bar \epsilon = \frac{1}{\sqrt{N_s}} (E_{loc, s} - \left<E_{loc, s}\right>)`.
        """
        if samples.reweight_factor is None:
            reweight_factor = 1
        else:
            reweight_factor = samples.reweight_factor

        Eloc = self._hamiltonian.Oloc(state, samples).astype(get_default_dtype())
        Emean = jnp.mean(Eloc * reweight_factor)
        self._energy = Emean.real
        Evar = jnp.abs(Eloc - Emean) ** 2
        self._VarE = jnp.mean(Evar * reweight_factor).real

        Eloc -= jnp.mean(Eloc)
        Eloc *= jnp.sqrt(reweight_factor / samples.nsamples)
        return Eloc

    def ebar_dense(self, psi: jax.Array, symm: Symmetry, Ns: int) -> jax.Array:
        r"""
        Compute :math:`\bar \epsilon = (H - E) \psi` for the normalized wave
        function ``psi`` given in the full Hilbert space of ``symm``, whose first
        ``Ns`` entries are the physical amplitudes.
        """
        dense = DenseState(psi[:Ns], symm)
        H_psi = self._hamiltonian @ dense
        energy = dense @ H_psi
        self._energy = jnp.asarray(energy.real)
        Ebar = H_psi - dense * energy
        Ebar = jnp.asarray(Ebar.psi)
        return array_extend(Ebar, psi.size)


class OverlapGrad:
    r"""
    Overlap gradient source defining :math:`\bar \epsilon` from the amplitude
    ratios with a target state, used for supervised wave function optimization.
    """

    def __init__(self, target_state: State, clip: Optional[float] = None):
        r"""
        :param target_state:
            The target state to be approximated.

        :param clip:
            The clipping value of the centered amplitude ratios, default to no
            clipping.
        """
        self._target_state = target_state
        self._clip = clip
        self._target_psi = None

    @property
    def target_state(self) -> State:
        """The target state to be approximated."""
        return self._target_state

    def ebar(self, state: Variational, samples: Samples) -> jax.Array:
        r"""
        Compute :math:`\bar \epsilon` from the normalized amplitude ratios
        :math:`\phi_s / \psi_s` for given samples.
        """
        if samples.psi is None:
            psi = state(samples.spins)
        else:
            psi = samples.psi
        if samples.reweight_factor is None:
            reweight = 1
        else:
            reweight = samples.reweight_factor

        phi = self._target_state(samples.spins)
        ratio = phi / psi
        ratio_mean = (ratio * reweight).mean()
        ratio = jnp.asarray(ratio / ratio_mean) - 1
        if self._clip is not None:
            ratio = jnp.clip(ratio, -self._clip, self._clip)
        Ebar = -ratio * jnp.sqrt(reweight / samples.nsamples)
        return Ebar

    def ebar_dense(
        self, psi: jax.Array, symm: Symmetry, Ns: int | None = None
    ) -> jax.Array:
        r"""
        Compute :math:`\bar \epsilon = \psi - \phi / \left<\psi|\phi\right>` for
        the normalized wave function ``psi`` given in the full Hilbert space of
        ``symm``, whose first ``Ns`` entries are the physical amplitudes.
        """
        if self._target_psi is None:
            self._target_psi = jnp.asarray(self._target_state.todense(symm).psi)
        if Ns is None:
            Ns = psi.size

        target_psi = self._target_psi
        Ebar = psi[:Ns] - target_psi / jnp.vdot(psi[:Ns], target_psi)
        return array_extend(Ebar, psi.size)
