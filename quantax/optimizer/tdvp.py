from typing import Optional, Callable, Tuple
from functools import partial
import jax
import jax.numpy as jnp

from .solver import auto_pinv_eig, pinvh_solve
from ..state import DenseState, Variational, VS_TYPE
from ..sampler import Samples
from ..operator import Operator
from ..symmetry import Symmetry
from ..utils import (
    array_extend,
    ints_to_array,
    get_replicate_sharding,
    get_global_sharding,
    to_replicate_array,
)
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
        kazcmarz_mu: float = 0.0,
    ):
        r"""
        :param state:
            Variational state to be optimized.

        :param imag_time:
            Whether to use imaginary-time evolution, default to True.

        :param solver:
            The numerical solver for the matrix inverse, default to `~quantax.optimizer.auto_pinv_eig`.

        :param use_kazcmarz:
            Whether to use the `kazcmarz <https://arxiv.org/abs/2401.10190>`_ scheme, default to False.
        """
        self._state = state
        self._imag_time = imag_time
        if solver is None:
            solver = auto_pinv_eig()
        self._solver = solver
        self._kazcmarz_mu = to_replicate_array(kazcmarz_mu)
        self._last_step = jnp.zeros(
            state.nparams, state.dtype, device=get_replicate_sharding()
        )
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
        """The vs_type of the state."""
        return self._state.vs_type

    @property
    def imag_time(self) -> bool:
        """Whether to use imaginary-time evolution."""
        return self.imag_time

    @property
    def kazcmarz_mu(self) -> float:
        """Whether to use the `kazcmarz <https://arxiv.org/abs/2401.10190>`_ scheme."""
        return self._kazcmarz_mu

    def get_Ebar(self) -> jax.Array:
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
    def _solve(
        self, Obar: jax.Array, Ebar: jax.Array, last_step: jax.Array
    ) -> jax.Array:
        Ebar -= self._kazcmarz_mu * Obar @ last_step.astype(Obar.dtype)

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

        step += self._kazcmarz_mu * last_step.astype(step.dtype)
        return step

    def solve(self, Obar: jax.Array, Ebar: jax.Array) -> jax.Array:
        r"""
        Solve the equation :math:`\bar O \dot \theta = \bar \epsilon` for given
        :math:`\bar O` and :math:`\bar \epsilon`.
        """
        step = self._solve(Obar, Ebar, self._last_step)
        self._last_step = step
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


class TDVP(QNGD):
    r"""
    Time-dependent variational principle TDVP, or stochastic reconfiguration (SR).
    """

    def __init__(
        self,
        state: Variational,
        hamiltonian: Operator,
        imag_time: bool = True,
        solver: Optional[Callable] = None,
        kazcmarz_mu: float = 0.0,
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

        :param use_kazcmarz:
            Whether to use the `kazcmarz <https://arxiv.org/abs/2401.10190>`_ scheme, default to False.
        """
        super().__init__(state, imag_time, solver, kazcmarz_mu)
        self._hamiltonian = hamiltonian

        self._energy = None
        self._VarE = None
        self._Omean = None

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


class TDVP_exact(QNGD):
    r"""
    Exact TDVP evolution, performed by a full summation in the whole Hilbert space.
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

    def get_Ebar(self, wave_function: jax.Array) -> jax.Array:
        r"""Compute :math:`\bar \epsilon` in the full Hilbert space."""
        psi = DenseState(wave_function, self._symm)
        H_psi = self._hamiltonian @ psi
        energy = psi @ H_psi
        Ebar = H_psi - energy * psi
        self._energy = energy.real
        return Ebar.wave_function

    def get_Obar(self, wave_function: jax.Array) -> jax.Array:
        r"""Compute :math:`\bar O` in the full Hilbert space."""
        Omat = self._state.jacobian(self._spins) * wave_function[:, None]
        Omat = jnp.where(jnp.isnan(Omat), 0, Omat)
        self._Omean = jnp.einsum("s,sk->k", wave_function.conj(), Omat)
        Omean = jnp.einsum("s,k->sk", wave_function, self._Omean)
        return Omat - Omean

    def get_step(self) -> jax.Array:
        r"""
        Obtain the optimization step by solving the equation :math:`\bar O \dot \theta = \bar \epsilon`.
        """
        wave_function = self._state(self._spins) / self._symm_norm
        wave_function /= jnp.linalg.norm(wave_function)
        Ebar = self.get_Ebar(wave_function)
        Obar = self.get_Obar(wave_function)
        step = self.solve(Obar, Ebar)
        return step


@jax.jit
def _AconjB(A: jax.Array, B: jax.Array) -> jax.Array:
    matmul = lambda x, y: x.conj().T @ y
    if A.ndim == 2:
        return matmul(A, B)
    elif A.ndim == 3:
        return jax.vmap(matmul)(A, B)
    else:
        raise NotImplementedError


class TimeEvol(TDVP):
    def __init__(
        self,
        state: Variational,
        hamiltonian: Operator,
        solver: Optional[Callable] = None,
    ):
        if solver is None:
            solver = pinvh_solve()
        super().__init__(state, hamiltonian, imag_time=False, solver=solver)
        self._max_parallel = state._backward_chunk

    def get_SF(self, samples: Samples) -> Tuple[jax.Array, jax.Array]:
        if (
            self._max_parallel is None
            or samples.nsamples <= self._max_parallel * jax.device_count()
        ):
            Ebar = self.get_Ebar(samples)
            Obar = self.get_Obar(samples)
            Smat = _AconjB(Obar, Obar)
            Fvec = _AconjB(Obar, Ebar)
            return Smat, Fvec
        else:
            return self._get_SF_indirect(samples)

    def _get_SF_indirect(self, samples: Samples) -> Tuple[jax.Array, jax.Array]:
        ndevices = jax.device_count()
        Eloc = self._hamiltonian.Oloc(self._state, samples)
        Emean = jnp.mean(Eloc)
        self._energy = Emean.real.item()
        Evar = jnp.abs(Eloc - Emean) ** 2
        self._VarE = jnp.mean(Evar).real.item()
        Eloc = Eloc.reshape(ndevices, -1)
        Eloc = array_extend(Eloc, self._max_parallel, axis=1)
        nsplits = Eloc.shape[1] // self._max_parallel
        Eloc = jnp.split(Eloc, nsplits, axis=1)

        nsamples, N = samples.spins.shape
        spins = samples.spins.reshape(ndevices, -1, N)
        spins = array_extend(spins, self._max_parallel, 1, padding_values=1)
        spins = jnp.split(spins, nsplits, axis=1)

        nparams = self._state.nparams
        dtype = get_default_dtype()
        sharding = get_global_sharding()
        Smat = jnp.zeros((ndevices, nparams, nparams), dtype, device=sharding)
        Fvec = jnp.zeros((ndevices, nparams), dtype, device=sharding)
        Omean = jnp.zeros((ndevices, nparams), dtype, device=sharding)
        for s, e in zip(spins, Eloc):
            Omat = self._state.jacobian(s.reshape(-1, N))
            Omat = Omat.reshape(ndevices, -1, nparams).astype(dtype)
            Omean += jnp.sum(Omat, axis=1)
            newS = _AconjB(Omat, Omat)
            newF = _AconjB(Omat, e)
            Smat += newS
            Fvec += newF
        # psum here, nsamples definition should be modified to all samples across nodes
        Smat = jnp.sum(Smat, axis=0) / nsamples
        Fvec = jnp.sum(Fvec, axis=0) / nsamples
        Omean = jnp.sum(Omean, axis=0) / nsamples
        self._Omean = Omean

        Smat = Smat - jnp.outer(Omean.conj(), Omean)
        Fvec = Fvec - Omean.conj() * Emean
        return Smat, Fvec

    @partial(jax.jit, static_argnums=0)
    def solve(self, Smat: jax.Array, Fvec: jax.Array) -> jax.Array:
        if self.vs_type == VS_TYPE.real_or_holomorphic:
            Fvec *= 1j
        else:
            Smat = Smat.real
            Fvec = -Fvec.imag
        step = self._solver(Smat, Fvec)

        if self.vs_type == VS_TYPE.non_holomorphic:
            step = step.reshape(2, -1)
            step = step[0] + 1j * step[1]
        step = step.astype(get_default_dtype())
        return step

    def get_step(self, samples: Samples) -> jax.Array:
        if not jnp.allclose(samples.reweight_factor, 1.0):
            raise ValueError("TimeEvol is only for non-reweighted samples")

        Smat, Fvec = self.get_SF(samples)
        step = self.solve(Smat, Fvec)
        return step
