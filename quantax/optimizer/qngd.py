from typing import Callable, BinaryIO
from pathlib import Path
from functools import partial
from warnings import warn
import jax
import jax.numpy as jnp
import equinox as eqx

from .solver import auto_shift_eig
from ..state import Variational, VS_TYPE
from ..sampler import Samples
from ..symmetry import Symmetry
from ..utils import (
    ints_to_array,
    to_replicated_numpy,
    to_replicated_array,
    filter_tree_map,
    array_extend,
    to_distributed_array,
)
from ..global_defs import get_default_dtype, is_default_cpl


class QNGD:
    r"""
    Abstract class of quantum natural gradient descent.
    """

    def __init__(
        self,
        state: Variational,
        imag_time: bool = True,
        solver: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
        file: str | Path | BinaryIO | None = None,
    ):
        r"""
        :param state:
            Variational state to be optimized.

        :param imag_time:
            Whether to use imaginary-time evolution.

        :param solver:
            The numerical solver for the matrix inverse, default to `~quantax.optimizer.auto_shift_eig`.

        :param file:
            The file with stored buffers of the optimizer.
        """
        self._state = state
        self._imag_time = imag_time
        if solver is None:
            solver = auto_shift_eig()
        self._solver = solver
        self._Omean = None
        if not hasattr(self, "_buffers"):
            self._buffers = {}
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
        return self.imag_time

    @partial(eqx.filter_jit, donate="all-except-first")
    def solve(
        self, Obar: jax.Array, Ebar: jax.Array, buffers: dict
    ) -> tuple[jax.Array, dict]:
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

        return step, buffers

    def save(self, file: str | Path | BinaryIO) -> None:
        r"""
        Save the optimizer buffers to a file.
        """
        buffers = filter_tree_map(to_replicated_numpy, self._buffers)
        if jax.process_index() == 0:
            eqx.tree_serialise_leaves(file, buffers)


@partial(jax.jit, donate_argnums=0)
def _Omat_to_Obar(Omat: jax.Array, factor: jax.Array) -> jax.Array:
    return (Omat - jnp.mean(Omat, axis=0, keepdims=True)) * factor


class StochasticQNGD(QNGD):
    r"""
    Abstract class of stochastic quantum natural gradient descent.

    The key function of the class is `~quantax.optimizer.StochasticQNGD.get_step`, which provides
    the update of parameters by solving the quantum natural gradient descent equation
    :math:`\bar O \dot \theta = \bar \epsilon`,
    in which :math:`\bar O = \frac{1}{\sqrt{N_s}}(\frac{1}{\psi} \frac{\partial \psi}{\partial \theta} - \left< \frac{1}{\psi} \frac{\partial \psi}{\partial \theta} \right>)`
    and :math:`\bar \epsilon` should be defined in the child class.
    """

    def get_Obar(self, samples: Samples) -> jax.Array:
        r"""
        Calculate
        :math:`\bar O = \frac{1}{\sqrt{N_s}}(\frac{1}{\psi} \frac{\partial \psi}{\partial \theta} - \left< \frac{1}{\psi} \frac{\partial \psi}{\partial \theta} \right>)`
        for given samples.
        """
        Omat = self._state.jacobian(samples.spins)
        has_nan = jnp.any(jnp.isnan(Omat), axis=1)
        if jnp.any(has_nan):
            nan_count = jnp.sum(has_nan)
            if jax.process_index() == 0:
                warn(f"{nan_count} NaN row(s) detected in the Jacobian matrix.")
            Omat = jnp.where(jnp.isnan(Omat), 0, Omat)

        if samples.reweight_factor is None:
            reweight_factor = 1
        else:
            reweight_factor = samples.reweight_factor[:, None]
        self._Omean = jnp.mean(Omat * reweight_factor, axis=0)
        factor = jnp.sqrt(reweight_factor / samples.nsamples)
        return _Omat_to_Obar(Omat, factor)

    def get_Ebar(self, samples: Samples) -> jax.Array:
        r"""Compute :math:`\bar \epsilon` for given samples."""
        raise NotImplementedError

    def get_step(self, samples: Samples) -> jax.Array:
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
    Abstract class of exact quantum natural gradient descent.

    The key function of the class is `~quantax.optimizer.ExactQNGD.get_step`, which provides
    the update of parameters by solving the quantum natural gradient descent equation
    :math:`\bar O \dot \theta = \bar \epsilon`.
    """

    def __init__(
        self,
        state: Variational,
        imag_time: bool = True,
        solver: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
        symm: Symmetry | None = None,
    ):
        r"""
        :param state:
            Variational state to be optimized.

        :param imag_time:
            Whether to use imaginary-time evolution, default to True.

        :param solver:
            The numerical solver for the matrix inverse, default to `~quantax.optimizer.auto_pinv_eig`.

        :param symm:
            Symmetry used to construct the Hilbert space, default to be the symmetry
            of the variational state.
        """
        QNGD.__init__(self, state, imag_time, solver)

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
        r"""Compute :math:`\bar \epsilon` in the full Hlbert space."""
        raise NotImplementedError

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
