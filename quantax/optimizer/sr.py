from typing import Callable, BinaryIO
from pathlib import Path
import jax

from .qngd import StochasticQNGD, ExactQNGD
from .updater import SpringUpdater, MarchUpdater, AdamUpdater
from .gradient import EnergyGrad
from ..state import Variational
from ..operator import Operator
from ..symmetry import Symmetry


class SR(StochasticQNGD):
    r"""
    Stochastic reconfiguration (SR). By default, this optimizer automatically chooses between
    `SR <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.80.4558>`_ and
    `MinSR <https://www.nature.com/articles/s41567-024-02566-1>`_
    based on the number of samples and parameters.
    """

    def __init__(
        self,
        state: Variational,
        hamiltonian: Operator,
        imag_time: bool = True,
        solver: Callable[..., jax.Array] | None = None,
        file: str | Path | BinaryIO | None = None,
    ):
        r"""
        :param state:
            Variational state to be optimized.

        :param hamiltonian:
            The Hamiltonian for the evolution.

        :param imag_time:
            Whether to use imaginary-time evolution.

        :param solver:
            The numerical solver for the matrix inverse, default to `~quantax.optimizer.auto_shift_eig`.

        :param file:
            The file with stored buffers of the optimizer.
        """
        grad = EnergyGrad(hamiltonian)
        StochasticQNGD.__init__(self, state, grad, imag_time, solver, file=file)


class SPRING(StochasticQNGD):
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
        solver: Callable[..., jax.Array] | None = None,
        file: str | Path | BinaryIO | None = None,
        mu: float = 0.9,
        norm_clip: float | None = None,
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
            default to `~quantax.optimizer.auto_shift_eig`.

        :param file:
            The file with stored buffers of the optimizer.

        :param mu:
            The momentum factor.

        :param norm_clip:
            The maximum norm of the step to be accumulated in momentum.
            If not None, the raw step will be clipped to this value.
        """
        grad = EnergyGrad(hamiltonian)
        updater = SpringUpdater(mu, norm_clip)
        StochasticQNGD.__init__(self, state, grad, imag_time, solver, updater, file)


class MARCH(StochasticQNGD):
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
        solver: Callable[..., jax.Array] | None = None,
        file: str | Path | BinaryIO | None = None,
        mu: float = 0.95,
        beta: float = 0.995,
        norm_clip: float | None = None,
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
            default to `~quantax.optimizer.auto_shift_eig`.

        :param file:
            The file with stored buffers of the optimizer.

        :param mu:
            The first order momentum factor.

        :param beta:
            The second order momentum factor.

        :param norm_clip:
            The maximum norm of the step to be accumulated in the first and second order momentum.
            If not None, the raw step will be clipped to this value.
        """
        grad = EnergyGrad(hamiltonian)
        updater = MarchUpdater(mu, beta, norm_clip)
        StochasticQNGD.__init__(self, state, grad, imag_time, solver, updater, file)


class AdamSR(StochasticQNGD):
    r"""
    AdamSR optimizer.
    This is a variant of SR with first and second order momentum (like Adam).
    The time cost is roughly twice of SR.
    """

    def __init__(
        self,
        state: Variational,
        hamiltonian: Operator,
        imag_time: bool = True,
        solver: Callable[..., jax.Array] | None = None,
        file: str | Path | BinaryIO | None = None,
        mu: float = 0.95,
        beta: float = 0.995,
        norm_clip: float | None = None,
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
            default to `~quantax.optimizer.auto_shift_eig`.

        :param file:
            The file with stored buffers of the optimizer.

        :param mu:
            The first order momentum factor.

        :param beta:
            The second order momentum factor.

        :param norm_clip:
            The maximum norm of the step to be accumulated.
            If not None, the raw step will be clipped to this value.
        """
        grad = EnergyGrad(hamiltonian)
        updater = AdamUpdater(mu, beta, norm_clip)
        StochasticQNGD.__init__(self, state, grad, imag_time, solver, updater, file)


class ER(ExactQNGD):
    r"""
    Exact reconfiguration, performed by a full summation in the whole Hilbert space.
    This is only available in small systems.
    """

    def __init__(
        self,
        state: Variational,
        hamiltonian: Operator,
        imag_time: bool = True,
        solver: Callable[..., jax.Array] | None = None,
        symm: Symmetry | None = None,
    ):
        r"""
        :param state:
            Variational state to be optimized.

        :param hamiltonian:
            The Hamiltonian for the evolution.

        :param imag_time:
            Whether to use imaginary-time evolution, default to True.

        :param solver:
            The numerical solver for the matrix inverse, default to `~quantax.optimizer.auto_shift_eig`.

        :param symm:
            Symmetry used to construct the Hilbert space, default to be the symmetry
            of the variational state.
        """
        grad = EnergyGrad(hamiltonian)
        ExactQNGD.__init__(self, state, grad, imag_time, solver, symm)
