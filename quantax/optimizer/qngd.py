from typing import Callable, BinaryIO
from jax.typing import ArrayLike
from pathlib import Path
from functools import partial
from warnings import warn
import inspect
import jax
import jax.numpy as jnp
import equinox as eqx

from .solver import auto_shift_eig
from .updater import Updater, PlainUpdater
from .gradient import EnergyGrad, OverlapGrad
from ..state import Variational, VS_TYPE
from ..sampler import Samples
from ..operator import Operator
from ..symmetry import Symmetry
from ..utils import (
    ints_to_array,
    get_replicated_sharding,
    to_replicated_numpy,
    to_replicated_array,
    filter_tree_map,
    array_extend,
    to_distributed_array,
)
from ..global_defs import get_default_dtype, is_default_cpl


def _accepts_diag_preconditioner(solver: Callable) -> bool:
    """Whether ``solver`` declares ``diag_preconditioner`` as an explicit argument."""
    try:
        params = inspect.signature(solver).parameters
    except (TypeError, ValueError):
        return False
    return "diag_preconditioner" in params


class QNGD:
    r"""
    Base class of quantum natural gradient descent. It solves the linear equation
    :math:`\bar O \dot \theta = \bar \epsilon`, in which :math:`\bar O` is the
    centered Jacobian matrix and :math:`\bar \epsilon` is defined by the
    gradient source ``grad``. The behavior is composed from three pluggable
    parts: the gradient source (e.g. `~quantax.optimizer.EnergyGrad`), the
    numerical ``solver``, and the ``updater`` strategy
    (e.g. `~quantax.optimizer.SpringUpdater`).
    """

    def __init__(
        self,
        state: Variational,
        grad: EnergyGrad | OverlapGrad,
        imag_time: bool = True,
        solver: Callable[..., jax.Array] | None = None,
        updater: Updater | None = None,
        file: str | Path | BinaryIO | None = None,
    ):
        r"""
        :param state:
            Variational state to be optimized.

        :param grad:
            The gradient source defining :math:`\bar \epsilon`, e.g.
            `~quantax.optimizer.EnergyGrad` or `~quantax.optimizer.OverlapGrad`.

        :param imag_time:
            Whether to use imaginary-time evolution.

        :param solver:
            The numerical solver for the matrix inverse, default to `~quantax.optimizer.auto_shift_eig`.

        :param updater:
            The update strategy applied around the equation solve, default to
            `~quantax.optimizer.PlainUpdater`.

        :param file:
            The file with stored buffers of the optimizer.
        """
        self._state = state
        self._grad = grad
        self._imag_time = imag_time
        if solver is None:
            solver = auto_shift_eig()
        self._solver = solver
        if updater is None:
            updater = PlainUpdater()
        self._updater = updater

        self._buffers = updater.init(state.nparams)
        if file is not None:
            self._buffers = eqx.tree_deserialise_leaves(file, self._buffers)
        self._buffers = filter_tree_map(to_replicated_array, self._buffers)

    @property
    def state(self) -> Variational:
        """Variational state to be optimized."""
        return self._state

    @property
    def holomorphic(self) -> bool:
        """Whether the state is holomorphic."""
        return self._state._holomorphic

    @property
    def vs_type(self) -> VS_TYPE:
        """The vs_type of the state, see `~quantax.state.VS_TYPE`."""
        return self._state.vs_type

    @property
    def imag_time(self) -> bool:
        """Whether to use imaginary-time evolution."""
        return self._imag_time

    @property
    def hamiltonian(self) -> Operator | None:
        """
        The Hamiltonian for the evolution, ``None`` when the gradient source
        doesn't define it.
        """
        return getattr(self._grad, "hamiltonian", None)

    @property
    def energy(self) -> ArrayLike | None:
        """
        Energy of the current step, ``None`` when the gradient source doesn't
        define it.
        """
        return getattr(self._grad, "energy", None)

    @property
    def VarE(self) -> ArrayLike | None:
        r"""
        Energy variance :math:`\left< (H - E)^2 \right>` of the current step,
        ``None`` when the gradient source doesn't define it.
        """
        return getattr(self._grad, "VarE", None)

    @partial(eqx.filter_jit, donate="all-except-first")
    def solve(
        self, Obar: jax.Array, Ebar: jax.Array, buffers: dict
    ) -> tuple[jax.Array, dict]:
        r"""
        Generate the optimization step for given :math:`\bar O` and
        :math:`\bar \epsilon` by applying the updater strategy around
        `~quantax.optimizer.QNGD.solve_equation`.
        """
        return self._updater.update(self.solve_equation, Obar, Ebar, buffers)

    def solve_equation(
        self, Obar: jax.Array, Ebar: jax.Array, **solver_kwargs
    ) -> jax.Array:
        r"""
        Solve the linear equation :math:`\bar O \dot \theta = \bar \epsilon` for given
        :math:`\bar O` and :math:`\bar \epsilon`. Real and imaginary parts are
        stacked for non-holomorphic states, and extra keyword arguments are
        forwarded to the numerical solver.

        A ``diag_preconditioner`` keyword is handled here: if the solver declares
        it as an argument (e.g. `~quantax.optimizer.lsmr`), it is passed through;
        otherwise the right preconditioning is emulated by solving with
        :math:`\bar O / d` and rescaling the output by :math:`1 / d`.
        """
        diag_preconditioner = solver_kwargs.pop("diag_preconditioner", None)

        if self.vs_type == VS_TYPE.real_or_holomorphic:
            if not self._imag_time:
                Ebar = Ebar * 1j
        else:
            Obar = jnp.concatenate([Obar.real, Obar.imag], axis=0)
            if self._imag_time:
                Ebar = jnp.concatenate([Ebar.real, Ebar.imag])
            else:
                Ebar = jnp.concatenate([-Ebar.imag, Ebar.real])

        if diag_preconditioner is None:
            step = self._solver(Obar, Ebar, **solver_kwargs)
        elif _accepts_diag_preconditioner(self._solver):
            step = self._solver(
                Obar, Ebar, diag_preconditioner=diag_preconditioner, **solver_kwargs
            )
        else:
            step = self._solver(Obar / diag_preconditioner, Ebar, **solver_kwargs)
            step = step / diag_preconditioner

        if self.vs_type == VS_TYPE.non_holomorphic:
            step = step.reshape(2, -1)
            step = step[0] + 1j * step[1]
        step = step.astype(get_default_dtype())
        step = jax.lax.with_sharding_constraint(step, get_replicated_sharding())
        return step

    def save(self, file: str | Path | BinaryIO) -> None:
        r"""
        Save the optimizer buffers to a file.
        """
        buffers = filter_tree_map(to_replicated_numpy, self._buffers)
        if jax.process_index() == 0:
            eqx.tree_serialise_leaves(file, buffers)


@jax.jit
def _Omat_stats(Omat: jax.Array) -> tuple[jax.Array, jax.Array]:
    # Reduce-only pass: NaN row count and column mean of the NaN-zeroed matrix.
    # ``n_nan_rows`` is returned so the caller can warn cheaply (scalar host
    # transfer).
    n_nan_rows = jnp.count_nonzero(jnp.any(jnp.isnan(Omat), axis=1))
    Omean = jnp.mean(jnp.where(jnp.isnan(Omat), 0.0, Omat), axis=0, keepdims=True)
    return Omean, n_nan_rows


@partial(jax.jit, donate_argnums=0)
def _Omat_center(Omat: jax.Array, Omean: jax.Array, factor: jax.Array) -> jax.Array:
    # Elementwise-only pass donating Omat, so it runs in place.
    Omat = jnp.where(jnp.isnan(Omat), 0.0, Omat)
    return (Omat - Omean) * factor


class StochasticQNGD(QNGD):
    r"""
    Stochastic quantum natural gradient descent.

    The key function of the class is `~quantax.optimizer.StochasticQNGD.get_step`, which provides
    the update of parameters by solving the quantum natural gradient descent equation
    :math:`\bar O \dot \theta = \bar \epsilon`,
    in which :math:`\bar O = \frac{1}{\sqrt{N_s}}(\frac{1}{\psi} \frac{\partial \psi}{\partial \theta} - \left< \frac{1}{\psi} \frac{\partial \psi}{\partial \theta} \right>)`
    and :math:`\bar \epsilon` is estimated on the samples by the gradient source.
    """

    def get_Obar(self, samples: Samples | jax.Array) -> jax.Array:
        r"""
        Calculate
        :math:`\bar O = \frac{1}{\sqrt{N_s}}(\frac{1}{\psi} \frac{\partial \psi}{\partial \theta} - \left< \frac{1}{\psi} \frac{\partial \psi}{\partial \theta} \right>)`
        for given samples.
        """
        if not isinstance(samples, Samples):
            samples = Samples(to_distributed_array(samples))

        Omat = self._state.jacobian(samples.spins)

        if samples.reweight_factor is None:
            reweight_factor = 1
        else:
            reweight_factor = samples.reweight_factor[:, None]
        factor = jnp.sqrt(reweight_factor / samples.nsamples)
        Omean, n_nan_rows = _Omat_stats(Omat)
        Obar = _Omat_center(Omat, Omean, factor)
        if jax.process_index() == 0 and n_nan_rows > 0:
            warn(f"{n_nan_rows} NaN row(s) detected in the Jacobian matrix.")
        return Obar

    def get_Ebar(self, samples: Samples | jax.Array) -> jax.Array:
        r"""Compute :math:`\bar \epsilon` of the gradient source for given samples."""
        return self._grad.ebar(self._state, samples)

    def get_step(self, samples: Samples | jax.Array) -> jax.Array:
        r"""
        Obtain the optimization step by solving the equation :math:`\bar O \dot \theta = \bar \epsilon`
        for given samples.
        """
        Ebar = self.get_Ebar(samples)
        Obar = self.get_Obar(samples)
        step, self._buffers = self.solve(Obar, Ebar, self._buffers)
        return step


class ExactQNGD(QNGD):
    r"""
    Exact quantum natural gradient descent, performed by a full summation in the
    whole Hilbert space.

    The key function of the class is `~quantax.optimizer.ExactQNGD.get_step`, which provides
    the update of parameters by solving the quantum natural gradient descent equation
    :math:`\bar O \dot \theta = \bar \epsilon`.
    """

    def __init__(
        self,
        state: Variational,
        grad: EnergyGrad | OverlapGrad,
        imag_time: bool = True,
        solver: Callable[..., jax.Array] | None = None,
        symm: Symmetry | None = None,
    ):
        r"""
        :param state:
            Variational state to be optimized.

        :param grad:
            The gradient source defining :math:`\bar \epsilon`, e.g.
            `~quantax.optimizer.EnergyGrad` or `~quantax.optimizer.OverlapGrad`.

        :param imag_time:
            Whether to use imaginary-time evolution, default to True.

        :param solver:
            The numerical solver for the matrix inverse, default to `~quantax.optimizer.auto_shift_eig`.

        :param symm:
            Symmetry used to construct the Hilbert space, default to be the symmetry
            of the variational state.
        """
        QNGD.__init__(self, state, grad, imag_time, solver)

        self._Omean = None

        self._symm = state.symm if symm is None else symm
        self._symm.basis_make()
        basis = self._symm.basis
        self._Ns = basis.Ns
        spins = jnp.asarray(ints_to_array(basis.states))
        ndevices = jax.device_count()
        self._spins = to_distributed_array(array_extend(spins, ndevices))
        symm_norm = jnp.asarray(basis.get_amp(basis.states))
        if not is_default_cpl():
            symm_norm = symm_norm.real
        self._symm_norm = to_distributed_array(array_extend(symm_norm, ndevices))

    def get_Ebar(self, psi: jax.Array) -> jax.Array:
        r"""Compute :math:`\bar \epsilon` of the gradient source in the full Hilbert space."""
        return self._grad.ebar_dense(psi, self._symm, self._Ns)

    def get_Obar(self, psi: jax.Array) -> jax.Array:
        r"""Compute :math:`\bar O` in the full Hilbert space."""
        Omat = self._state.jacobian(self._spins) * psi[:, None]
        Omat = jnp.where(jnp.isfinite(Omat), Omat, 0)
        self._Omean = jnp.einsum("s,sk->k", psi.conj(), Omat)
        Omean = jnp.einsum("s,k->sk", psi, self._Omean)
        return Omat - Omean

    def get_step(self) -> jax.Array:
        r"""
        Obtain the optimization step by solving the equation :math:`\bar O \dot \theta = \bar \epsilon`.
        """
        psi = jnp.asarray(self._state(self._spins) / self._symm_norm)
        psi = to_replicated_array(psi)
        psi = psi.at[self._Ns :].set(0)
        psi /= jnp.linalg.norm(psi)
        Ebar = self.get_Ebar(psi)
        Obar = self.get_Obar(psi)
        step, self._buffers = self.solve(Obar, Ebar, self._buffers)
        return step
