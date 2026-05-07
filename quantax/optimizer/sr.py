from typing import Callable, BinaryIO
from jax.typing import ArrayLike
from pathlib import Path
from functools import partial
from warnings import warn
import jax
import jax.numpy as jnp
import equinox as eqx

from .solver import auto_shift_eig
from ..state import DenseState, Variational, VS_TYPE
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

    def get_Ebar(self, samples: Samples) -> jax.Array:
        r"""Method for computing :math:`\bar \epsilon` in QNGD equaion, specified by the child class."""
        raise NotImplementedError

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
        return self._Omat_to_Obar(Omat, factor)

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

    def get_step(self, samples: Samples) -> jax.Array:
        r"""
        Obtain the optimization step by solving the equation :math:`\bar O \dot \theta = \bar \epsilon`
        for given samples.
        """
        Ebar = self.get_Ebar(samples)
        Obar = self.get_Obar(samples)
        step, self._buffers = self.solve(Obar, Ebar, self._buffers)
        return step

    def save(self, file: str | Path | BinaryIO) -> None:
        r"""
        Save the optimizer buffers to a file.
        """
        buffers = filter_tree_map(to_replicated_numpy, self._buffers)
        if jax.process_index() == 0:
            eqx.tree_serialise_leaves(file, buffers)


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
        solver: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
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
            The numerical solver for the matrix inverse, default to `~quantax.optimizer.auto_pinv_eig`.

        :param file:
            The file with stored buffers of the optimizer.
        """
        QNGD.__init__(self, state, imag_time, solver, file)
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

    def get_Ebar(self, samples: Samples) -> jax.Array:
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

        Eloc = self._hamiltonian.Oloc(self._state, samples).astype(get_default_dtype())
        Emean = jnp.mean(Eloc * reweight_factor)
        self._energy = Emean.real
        Evar = jnp.abs(Eloc - Emean) ** 2
        self._VarE = jnp.mean(Evar * reweight_factor).real

        Eloc -= jnp.mean(Eloc)
        Eloc *= jnp.sqrt(reweight_factor / samples.nsamples)
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
        solver: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
        file: str | Path | BinaryIO | None = None,
        mu: float = 0.9,
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

        :param file:
            The file with stored buffers of the optimizer.

        :param mu:
            The momentum factor.
        """

        self._mu = mu
        dtype = get_default_dtype()
        sharding = get_replicated_sharding()
        phi = jnp.zeros(state.nparams, dtype=dtype, device=sharding)
        self._buffers = {"phi": phi}
        SR.__init__(self, state, hamiltonian, imag_time, solver, file)

    @partial(eqx.filter_jit, donate="all-except-first")
    def solve(
        self, Obar: jax.Array, Ebar: jax.Array, buffers: dict
    ) -> tuple[jax.Array, dict]:
        r"""
        Solve the SPRING optimization step.
        """
        phi = buffers["phi"]
        Ebar -= self._mu * (Obar @ phi)
        step, buffers = SR.solve(self, Obar, Ebar, buffers)
        step = step + self._mu * phi
        buffers["phi"] = step
        return step, buffers


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
        solver: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
        file: str | Path | BinaryIO | None = None,
        mu: float = 0.95,
        beta: float = 0.995,
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

        :param file:
            The file with stored buffers of the optimizer.

        :param mu:
            The first order momentum factor.

        :param beta:
            The second order momentum factor.
        """

        self._mu = mu
        self._beta = beta
        dtype = get_default_dtype()
        sharding = get_replicated_sharding()
        phi = jnp.zeros(state.nparams, dtype=dtype, device=sharding)
        real_dtype = jnp.finfo(dtype).dtype
        v = jnp.zeros(state.nparams, dtype=real_dtype, device=sharding)
        self._buffers = {"phi": phi, "v": v}
        super().__init__(state, hamiltonian, imag_time, solver, file)

    @partial(eqx.filter_jit, donate="all-except-first")
    def solve(
        self, Obar: jax.Array, Ebar: jax.Array, buffers: dict
    ) -> tuple[jax.Array, dict]:
        r"""
        Solve the MARCH optimization step.
        """
        phi = buffers["phi"]
        v = buffers["v"]
        Ebar -= self._mu * (Obar @ phi)
        V = jnp.where(jnp.allclose(v, 0), jnp.ones_like(v), v**0.25 + 1e-8)

        Obar /= V[None, :]
        step, buffers = SR.solve(self, Obar, Ebar, buffers)
        step = step / V + self._mu * phi

        buffers["phi"] = step
        buffers["v"] = self._beta * v + jnp.abs(step - phi) ** 2
        return step, buffers


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
        solver: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
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
            default to `~quantax.optimizer.auto_pinv_eig`.

        :param file:
            The file with stored buffers of the optimizer.

        :param mu:
            The first order momentum factor.

        :param beta:
            The second order momentum factor.

        :param norm_clip:
            The maximum norm of the gradient.
            If not None, the raw gradient will be clipped to this value.
        """

        self._mu = mu
        self._beta = beta
        self._norm_clip = norm_clip
        dtype = get_default_dtype()
        sharding = get_replicated_sharding()
        m = jnp.zeros(state.nparams, dtype=dtype, device=sharding)
        real_dtype = jnp.finfo(dtype).dtype
        v = jnp.zeros(state.nparams, dtype=real_dtype, device=sharding)
        t = jnp.zeros((), jnp.int32, device=sharding)
        self._buffers = {"m": m, "v": v, "t": t}
        SR.__init__(self, state, hamiltonian, imag_time, solver, file)

    @partial(eqx.filter_jit, donate="all-except-first")
    def solve(
        self, Obar: jax.Array, Ebar: jax.Array, buffers: dict
    ) -> tuple[jax.Array, dict]:
        r"""
        Solve the AdamSR optimization step. The time cost is roughly twice of SR.
        """
        g, buffers = SR.solve(self, Obar, Ebar, buffers)
        if self._norm_clip is not None:
            g_norm = jnp.linalg.norm(g)
            g = jnp.where(g_norm > self._norm_clip, g * (self._norm_clip / g_norm), g)

        t = buffers["t"]
        m = buffers["m"]
        v = buffers["v"]
        t += 1
        m = self._mu * m + (1 - self._mu) * g
        v = self._beta * v + (1 - self._beta) * jnp.abs(g) ** 2
        buffers["t"] = t
        buffers["m"] = m
        buffers["v"] = v

        mhat = m / (1 - self._mu**t)
        vhat = v / (1 - self._beta**t)
        V = vhat**0.25 + 1e-8

        Ebar -= Obar @ mhat
        Obar /= V[None, :]
        step, buffers = SR.solve(self, Obar, Ebar, buffers)
        step = step / V + mhat
        return step, buffers


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
        solver: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
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
            The numerical solver for the matrix inverse, default to `~quantax.optimizer.auto_pinv_eig`.

        :param symm:
            Symmetry used to construct the Hilbert space, default to be the symmetry
            of the variational state.
        """
        SR.__init__(self, state, imag_time, solver)

        self._hamiltonian = hamiltonian
        self._energy = None
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

    @property
    def hamiltonian(self) -> Operator:
        """The Hamiltonian for the evolution."""
        return self._hamiltonian

    @property
    def energy(self) -> ArrayLike | None:
        """Energy of the current step."""
        return self._energy

    def get_full_Ebar(self, psi: jax.Array) -> jax.Array:
        r"""Compute :math:`\bar \epsilon` in the full Hilbert space."""
        dense = DenseState(psi[: self._Ns], self._symm)
        H_psi = self._hamiltonian @ dense
        energy = dense @ H_psi
        self._energy = jnp.asarray(energy.real)
        Ebar = H_psi - dense * energy
        Ebar = jnp.asarray(Ebar.psi)
        return array_extend(Ebar, psi.size)

    def get_full_Obar(self, psi: jax.Array) -> jax.Array:
        r"""Compute :math:`\bar O` in the full Hilbert space."""
        Omat = self._state.jacobian(self._spins) * psi[:, None]
        Omat = jnp.where(jnp.isfinite(Omat), Omat, 0)
        self._Omean = jnp.einsum("s,sk->k", psi.conj(), Omat)
        Omean = jnp.einsum("s,k->sk", psi, self._Omean)
        return Omat - Omean

    def get_exact_step(self) -> jax.Array:
        r"""
        Obtain the optimization step by solving the equation :math:`\bar O \dot \theta = \bar \epsilon`.
        """
        psi = jnp.asarray(self._state(self._spins) / self._symm_norm)
        psi = to_replicated_array(psi)
        psi = psi.at[self._Ns :].set(0)
        psi /= jnp.linalg.norm(psi)
        Ebar = self.get_full_Ebar(psi)
        Obar = self.get_full_Obar(psi)
        step, self._buffers = self.solve(Obar, Ebar, self._buffers)
        return step
