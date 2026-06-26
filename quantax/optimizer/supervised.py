from typing import Callable, BinaryIO
from pathlib import Path
import jax

from .qngd import StochasticQNGD, ExactQNGD
from .updater import AdamUpdater
from .gradient import OverlapGrad
from ..symmetry import Symmetry
from ..state import State, Variational


class Supervised(StochasticQNGD):
    r"""
    Supervised optimization of the wave function towards a target state by
    stochastic reconfiguration on the overlap gradient.
    """

    def __init__(
        self,
        state: Variational,
        target_state: State,
        solver: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
        file: str | Path | BinaryIO | None = None,
        clip: float | None = None,
    ):
        r"""
        :param state:
            Variational state to be optimized.

        :param target_state:
            The target state to be approximated.

        :param solver:
            The numerical solver for the matrix inverse, default to `~quantax.optimizer.auto_shift_eig`.

        :param file:
            The file with stored buffers of the optimizer.

        :param clip:
            The clipping value of the centered amplitude ratios, default to no clipping.
        """
        grad = OverlapGrad(target_state, clip)
        StochasticQNGD.__init__(self, state, grad, solver=solver, file=file)


class SupervisedAdam(StochasticQNGD):
    r"""
    Supervised optimization towards a target state with the Adam-like update of
    `~quantax.optimizer.AdamSR`.
    """

    def __init__(
        self,
        state: Variational,
        target_state: State,
        solver: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
        file: str | Path | BinaryIO | None = None,
        clip: float | None = None,
        mu: float = 0.95,
        beta: float = 0.995,
        norm_clip: float | None = None,
    ):
        r"""
        :param state:
            Variational state to be optimized.

        :param target_state:
            The target state to be approximated.

        :param solver:
            The numerical solver for the matrix inverse, default to `~quantax.optimizer.auto_shift_eig`.

        :param file:
            The file with stored buffers of the optimizer.

        :param clip:
            The clipping value of the centered amplitude ratios, default to no clipping.

        :param mu:
            The first order momentum factor.

        :param beta:
            The second order momentum factor.

        :param norm_clip:
            The maximum norm of the step to be accumulated.
            If not None, the raw step will be clipped to this value.
        """
        grad = OverlapGrad(target_state, clip)
        updater = AdamUpdater(mu, beta, norm_clip)
        StochasticQNGD.__init__(
            self, state, grad, solver=solver, file=file, updater=updater
        )


class SupervisedExact(ExactQNGD):
    r"""
    Supervised optimization towards a target state by a full summation in the
    whole Hilbert space. This is only available in small systems.
    """

    def __init__(
        self,
        state: Variational,
        target_state: State,
        solver: Callable | None = None,
        symm: Symmetry | None = None,
    ):
        r"""
        :param state:
            Variational state to be optimized.

        :param target_state:
            The target state to be approximated.

        :param solver:
            The numerical solver for the matrix inverse, default to `~quantax.optimizer.auto_shift_eig`.

        :param symm:
            Symmetry used to construct the Hilbert space, default to be the symmetry
            of the variational state.
        """
        grad = OverlapGrad(target_state)
        ExactQNGD.__init__(self, state, grad, solver=solver, symm=symm)
