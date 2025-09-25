from typing import Optional, Callable, Union, BinaryIO
from pathlib import Path
from functools import partial
import jax
import jax.numpy as jnp
import equinox as eqx

from .solver import auto_pinv_eig
from ..state import DenseState, Variational, VS_TYPE
from ..sampler import Samples
from ..operator import Operator
from ..symmetry import Symmetry
from ..utils import ints_to_array, get_replicate_sharding
from ..global_defs import get_default_dtype, is_default_cpl


class QNGD:
    r"""
    Abstract class of quantum natural gradient descent.

    The key function of the class is `~quantax.optimizer.QNGD.get_step`, which provides
    the update of parameters by solving the quantum natural gradient descent equation
    :math:`\bar O \dot \theta = \bar \epsilon`,
    in which :math:`\bar O = \frac{1}{\sqrt{N_s}}(\frac{1}{\psi} \frac{\partial \psi}{\partial \theta} - \left< \frac{1}{\psi} \frac{\partial \psi}{\partial \theta} \right>)`
    and :math:`\bar \epsilon` should be defined in the child class.
    """

    def __init__(
        self,
        state: Variational,
        imag_time: bool = True,
        solver: Optional[Callable[[jax.Array, jax.Array], jax.Array]] = None,
    ):
        r"""
        :param state:
            Variational state to be optimized.

        :param imag_time:
            Whether to use imaginary-time evolution.

        :param solver:
            The numerical solver for the matrix inverse, default to `~quantax.optimizer.auto_pinv_eig`.
        """
        self._state = state
        self._imag_time = imag_time
        if solver is None:
            solver = auto_pinv_eig()
        self._solver = solver
        self._Omean = None

    @property
    def state(self) -> Variational:
        """Variational state to be optimized."""
        return self._state

    @property
    def holomorphic(self) -> bool:
        """Whether the state is holomorphic."""
        return self._state._holomorphic

    @property
    def vs_type(self) -> int:
        """The vs_type of the state, see `~quantax.state.VS_TYPE`."""
        return self._state.vs_type

    @property
    def imag_time(self) -> bool:
        """Whether to use imaginary-time evolution."""
        return self.imag_time

    def get_Ebar(self, samples: Samples) -> jax.Array:
        r"""Method for computing :math:`\bar \epsilon` in QNGD equaion, specified by the child class."""

    @staticmethod
    @partial(jax.jit, donate_argnums=0)
    def _Omat_to_Obar(Omat: jax.Array, factor: jax.Array) -> jax.Array:
        return (Omat - jnp.mean(Omat, axis=0, keepdims=True)) * factor

    def get_Obar(self, samples: Samples) -> jax.Array:
        r"""
        Calculate
        :math:`\bar O = \frac{1}{\sqrt{N_s}}(\frac{1}{\psi} \frac{\partial \psi}{\partial \theta} - \left< \frac{1}{\psi} \frac{\partial \psi}{\partial \theta} \right>)`
        for given samples.
        """
        Omat = self._state.jacobian(samples.spins)
        self._Omean = jnp.mean(Omat * samples.reweight_factor[:, None], axis=0)
        factor = jnp.sqrt(samples.reweight_factor / samples.nsamples)[:, None]
        return self._Omat_to_Obar(Omat, factor)

    @partial(jax.jit, static_argnums=0)
    def solve(self, Obar: jax.Array, Ebar: jax.Array) -> jax.Array:
        r"""
        Solve the equation :math:`\bar O \dot \theta = \bar \epsilon` for given
        :math:`\bar O` and :math:`\bar \epsilon`.
        """
        if self.vs_type == VS_TYPE.real_or_holomorphic:
            if not self._imag_time:
                Ebar *= 1j
        else:
            Obar = jnp.concatenate([Obar.real, Obar.imag], axis=0)
            if self._imag_time:
                Ebar = jnp.concatenate([Ebar.real, Ebar.imag])
            else:
                Ebar = jnp.concatenate([-Ebar.imag, Ebar.real])

        step = self._solver(Obar, Ebar)

        if self.vs_type == VS_TYPE.non_holomorphic:
            step = step.reshape(2, -1)
            step = step[0] + 1j * step[1]
        step = step.astype(get_default_dtype())

        return step

    def get_step(self, samples: Samples) -> jax.Array:
        r"""
        Obtain the optimization step by solving the equation :math:`\bar O \dot \theta = \bar \epsilon`
        for given samples.
        """
        Ebar = self.get_Ebar(samples)
        Obar = self.get_Obar(samples)
        step = self.solve(Obar, Ebar)
        return step

    def save(self, file: Union[str, Path, BinaryIO]) -> None:
        r"""
        Save the optimizer internal quantities to a file.
        """


class SR(QNGD):
    r"""
    Stochastic reconfiguration (SR). This optimizer automatically chooses between
    `SR <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.80.4558>`_ and
    `MinSR <https://www.nature.com/articles/s41567-024-02566-1>`_
    based on the the number of samples and parameters.
    """

    def __init__(
        self,
        state: Variational,
        hamiltonian: Operator,
        imag_time: bool = True,
        solver: Optional[Callable] = None,
    ):
        r"""
        :param state:
            Variational state to be optimized.

        :param hamiltonian:
            The Hamiltonian for the evolution.

        :param imag_time:
            Whether to use imaginary-time evolution.

        :param solver:
            The numerical solver for the matrix inverse, default to `~quantax.optimizer.auto_pinv_eig`.
        """
        super().__init__(state, imag_time, solver)
        self._hamiltonian = hamiltonian

        self._energy = None
        self._VarE = None

    @property
    def hamiltonian(self) -> Operator:
        """The Hamiltonian for the evolution."""
        return self._hamiltonian

    @property
    def energy(self) -> Optional[float]:
        """Energy of the current step."""
        return self._energy

    @property
    def VarE(self) -> Optional[float]:
        r"""Energy variance :math:`\left< (H - E)^2 \right>` of the current step."""
        return self._VarE

    def get_Ebar(self, samples: Samples) -> jax.Array:
        r"""
        Compute :math:`\bar \epsilon` for given samples. The local energy is
        :math:`E_{loc, s} = \sum_{s'} \frac{\psi_{s'}}{\psi_s} \left< s|H|s' \right>`,
        and :math:`\bar \epsilon` is defined as
        :math:`\bar \epsilon = \frac{1}{\sqrt{N_s}} (E_{loc, s} - \left<E_{loc, s}\right>)`.
        """
        Eloc = self._hamiltonian.Oloc(self._state, samples).astype(get_default_dtype())
        Emean = jnp.mean(Eloc * samples.reweight_factor)
        self._energy = Emean.real
        Evar = jnp.abs(Eloc - Emean) ** 2
        self._VarE = jnp.mean(Evar * samples.reweight_factor).real

        Eloc -= jnp.mean(Eloc)
        Eloc *= jnp.sqrt(samples.reweight_factor / samples.nsamples)
        return Eloc


class SPRING(SR):
    r"""
    `SPRING optimizer <https://doi.org/10.1016/j.jcp.2024.113351>`_.
    This is a variant of SR with momentum. When using the default value of `mu=0.9`,
    the learning rate should be roughly 1/5 of the one in SR.
    """

    def __init__(
        self,
        state: Variational,
        hamiltonian: Operator,
        imag_time: bool = True,
        solver: Optional[Callable] = None,
        mu: float = 0.9,
        file: Union[None, str, Path, BinaryIO] = None,
    ):
        r"""
        Initialize the SPRING optimizer.

        :param state:
            Variational state to be optimized.

        :param hamiltonian:
            The Hamiltonian for the evolution.

        :param imag_time:
            Whether to use imaginary-time evolution.

        :param solver:
            The numerical solver for the matrix inverse,
            default to `~quantax.optimizer.auto_pinv_eig`.

        :param mu:
            The momentum factor.

        :param file:
            File to load the optimizer internal quantities.
        """

        super().__init__(state, hamiltonian, imag_time, solver)

        self._mu = mu
        self._last_step = jnp.zeros(
            state.nparams, state.dtype, device=get_replicate_sharding()
        )
        if file is not None:
            val = eqx.tree_deserialise_leaves(file, (self._mu, self._last_step))
            self._mu, self._last_step = val

    def solve(self, Obar: jax.Array, Ebar: jax.Array) -> jax.Array:
        r"""
        Solve the SPRING optimization step.
        """
        Ebar -= self._mu * (Obar @ self._last_step.astype(Obar.dtype))
        step = super().solve(Obar, Ebar)
        step += self._mu * self._last_step.astype(step.dtype)
        self._last_step = step
        return step

    def save(self, file: Union[str, Path, BinaryIO]) -> None:
        r"""
        Save the optimizer internal quantities to a file.
        """
        val = (self._mu, self._last_step)
        if jax.process_index() == 0:
            eqx.tree_serialise_leaves(file, val)


class MARCH(SR):
    r"""
    `MARCH optimizer <https://arxiv.org/abs/2507.02644>`_.
    This is a variant of SR with first and second order momentum (like Adam).
    When using the default value of `mu=0.95` and `beta=0.995`,
    the learning rate should be roughly 1/5 of the one in SR.
    """

    def __init__(
        self,
        state: Variational,
        hamiltonian: Operator,
        imag_time: bool = True,
        solver: Optional[Callable] = None,
        mu: float = 0.95,
        beta: float = 0.995,
        file: Union[None, str, Path, BinaryIO] = None,
    ):
        r"""
        Initialize the MARCH optimizer.

        :param state:
            Variational state to be optimized.

        :param hamiltonian:
            The Hamiltonian for the evolution.

        :param imag_time:
            Whether to use imaginary-time evolution.

        :param solver:
            The numerical solver for the matrix inverse,
            default to `~quantax.optimizer.auto_pinv_eig`.

        :param mu:
            The first order momentum factor.

        :param beta:
            The second order momentum factor.
        """

        super().__init__(state, hamiltonian, imag_time, solver)
        self._mu = mu
        self._beta = beta
        sharding = get_replicate_sharding()
        self._last_step = jnp.zeros(state.nparams, state.dtype, device=sharding)
        real_dtype = jnp.finfo(state.dtype).dtype
        self._V = jnp.zeros(state.nparams, real_dtype, device=sharding)
        self._t = 0
        if file is not None:
            val = eqx.tree_deserialise_leaves(
                file, (self._mu, self._beta, self._last_step, self._V, self._t)
            )
            self._mu, self._beta, self._last_step, self._V, self._t = val

    def solve(self, Obar: jax.Array, Ebar: jax.Array) -> jax.Array:
        r"""
        Solve the MARCH optimization step.
        """
        self._t += 1

        Ebar -= self._mu * (Obar @ self._last_step.astype(Obar.dtype))
        if jnp.allclose(self._V, 0):
            V = jnp.ones_like(self._V)
        else:
            V = self._V / (1 - self._beta**self._t)
            V = V**0.25 + 1e-8

        Obar /= V[None, :]
        step = super().solve(Obar, Ebar)
        step = (step / V + self._mu * self._last_step).astype(step.dtype)

        dtheta2 = jnp.abs(step - self._last_step) ** 2
        self._V = self._beta * self._V + (1 - self._beta) * dtheta2
        self._last_step = step
        return step

    def save(self, file: Union[str, Path, BinaryIO]) -> None:
        r"""
        Save the optimizer internal quantities to a file.
        """
        val = (self._mu, self._beta, self._last_step, self._V, self._t)
        if jax.process_index() == 0:
            eqx.tree_serialise_leaves(file, val)


class AdamSR(SR):
    r"""
    AdamSR optimizer.
    This is a variant of SR with first and second order momentum (like Adam).
    """

    def __init__(
        self,
        state: Variational,
        hamiltonian: Operator,
        imag_time: bool = True,
        solver: Optional[Callable] = None,
        mu: float = 0.95,
        beta: float = 0.995,
        file: Union[None, str, Path, BinaryIO] = None,
    ):
        r"""
        Initialize the AdamSR optimizer.

        :param state:
            Variational state to be optimized.

        :param hamiltonian:
            The Hamiltonian for the evolution.

        :param imag_time:
            Whether to use imaginary-time evolution.

        :param solver:
            The numerical solver for the matrix inverse,
            default to `~quantax.optimizer.auto_pinv_eig`.

        :param mu:
            The first order momentum factor.

        :param beta:
            The second order momentum factor.
        """

        super().__init__(state, hamiltonian, imag_time, solver)
        self._mu = mu
        self._beta = beta
        sharding = get_replicate_sharding()
        self._m = jnp.zeros(state.nparams, state.dtype, device=sharding)
        real_dtype = jnp.finfo(state.dtype).dtype
        self._v = jnp.zeros(state.nparams, real_dtype, device=sharding)
        self._t = 0
        if file is not None:
            val = eqx.tree_deserialise_leaves(
                file, (self._mu, self._beta, self._m, self._v, self._t)
            )
            self._mu, self._beta, self._m, self._v, self._t = val

    def solve(self, Obar: jax.Array, Ebar: jax.Array) -> jax.Array:
        r"""
        Solve the AdamSR optimization step. The time cost is roughly twice of SR.
        """
        self._t += 1
        g = super().solve(Obar, Ebar)
        self._m = self._mu * self._m + (1 - self._mu) * g
        self._v = self._beta * self._v + (1 - self._beta) * jnp.abs(g) ** 2
        m = self._m / (1 - self._mu**self._t)
        v = self._v / (1 - self._beta**self._t)
        V = v**0.25 + 1e-8

        Ebar -= Obar @ m.astype(Obar.dtype)
        Obar /= V[None, :]
        step = super().solve(Obar, Ebar)
        step = (step / V + m).astype(step.dtype)
        return step

    def save(self, file: Union[str, Path, BinaryIO]) -> None:
        r"""
        Save the optimizer internal quantities to a file.
        """
        val = (self._mu, self._beta, self._m, self._v, self._t)
        if jax.process_index() == 0:
            eqx.tree_serialise_leaves(file, val)


class ER(QNGD):
    r"""
    Exact reconfiguration, performed by a full summation in the whole Hilbert space.
    This is only available in small systems.
    """

    def __init__(
        self,
        state: Variational,
        hamiltonian: Operator,
        imag_time: bool = True,
        solver: Optional[Callable] = None,
        symm: Optional[Symmetry] = None,
    ):
        r"""
        :param state:
            Variational state to be optimized.

        :param hamiltonian:
            The Hamiltonian for the evolution.

        :param imag_time:
            Whether to use imaginary-time evolution, default to True.

        :param solver:
            The numerical solver for the matrix inverse, default to `~quantax.optimizer.auto_pinv_eig`.

        :param symm:
            Symmetry used to construct the Hilbert space, default to be the symmetry
            of the variational state.
        """
        super().__init__(state, imag_time, solver)

        self._hamiltonian = hamiltonian
        self._energy = None
        self._Omean = None

        self._symm = state.symm if symm is None else symm
        self._symm.basis_make()
        basis = self._symm.basis
        self._spins = ints_to_array(basis.states)
        self._symm_norm = jnp.asarray(basis.get_amp(basis.states))
        if not is_default_cpl():
            self._symm_norm = self._symm_norm.real

    @property
    def hamiltonian(self) -> Operator:
        """The Hamiltonian for the evolution."""
        return self._hamiltonian

    @property
    def energy(self) -> Optional[float]:
        """Energy of the current step."""
        return self._energy

    def get_Ebar(self, psi: jax.Array) -> jax.Array:
        r"""Compute :math:`\bar \epsilon` in the full Hilbert space."""
        dense = DenseState(psi, self._symm)
        H_psi = self._hamiltonian @ dense
        energy = dense @ H_psi
        Ebar = H_psi - dense * energy
        self._energy = energy.real
        return Ebar.psi

    def get_Obar(self, psi: jax.Array) -> jax.Array:
        r"""Compute :math:`\bar O` in the full Hilbert space."""
        Omat = self._state.jacobian(self._spins) * psi[:, None]
        Omat = jnp.where(jnp.isnan(Omat), 0, Omat)
        self._Omean = jnp.einsum("s,sk->k", psi.conj(), Omat)
        Omean = jnp.einsum("s,k->sk", psi, self._Omean)
        return Omat - Omean

    def get_step(self) -> jax.Array:
        r"""
        Obtain the optimization step by solving the equation :math:`\bar O \dot \theta = \bar \epsilon`.
        """
        psi = self._state(self._spins) / self._symm_norm
        psi /= jnp.linalg.norm(psi)
        Ebar = self.get_Ebar(psi)
        Obar = self.get_Obar(psi)
        step = self.solve(Obar, Ebar)
        return step
