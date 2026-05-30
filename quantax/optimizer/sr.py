from typing import Callable, BinaryIO
from jax.typing import ArrayLike
from pathlib import Path
from functools import partial
import jax
import jax.numpy as jnp
import equinox as eqx

from .qngd import StochasticQNGD, ExactQNGD
from ..state import DenseState, Variational
from ..sampler import Samples
from ..operator import Operator
from ..symmetry import Symmetry
from ..utils import get_replicated_sharding, array_extend
from ..global_defs import get_default_dtype


class SR(StochasticQNGD):
    r"""
    Stochastic reconfiguration (SR). By default, this optimizer automatically chooses between
    `SR <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.80.4558>`_ and
    `MinSR <https://www.nature.com/articles/s41567-024-02566-1>`_
    based on the the number of samples and parameters.
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
            The numerical solver for the matrix inverse, default to `~quantax.optimizer.auto_pinv_eig`.

        :param file:
            The file with stored buffers of the optimizer.
        """
        StochasticQNGD.__init__(self, state, imag_time, solver, file)
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
            default to `~quantax.optimizer.auto_pinv_eig`.

        :param file:
            The file with stored buffers of the optimizer.

        :param mu:
            The momentum factor.

        :param norm_clip:
            The maximum norm of the step to be accumulated in momentum.
            If not None, the raw step will be clipped to this value.
        """

        self._mu = mu
        self._norm_clip = norm_clip
        dtype = get_default_dtype()
        sharding = get_replicated_sharding()
        phi = jnp.zeros(state.nparams, dtype=dtype, device=sharding)
        self._buffers = {"phi": phi}
        SR.__init__(self, state, hamiltonian, imag_time, solver, file)

    @partial(eqx.filter_jit, donate="all-except-first")
    def solve(
        self, Obar: jax.Array, Ebar: jax.Array, buffers: dict[str, jax.Array]
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        r"""
        Solve the SPRING optimization step.
        """
        phi = buffers["phi"]
        Ebar -= self._mu * (Obar @ phi)
        step, buffers = SR.solve(self, Obar, Ebar, buffers)
        if self._norm_clip is not None:
            norm = jnp.asarray(jnp.linalg.norm(step))
            step = jnp.where(
                norm > self._norm_clip, step * (self._norm_clip / norm), step
            )
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
            default to `~quantax.optimizer.auto_pinv_eig`.

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

        self._mu = mu
        self._beta = beta
        self._norm_clip = norm_clip
        dtype = get_default_dtype()
        sharding = get_replicated_sharding()
        phi = jnp.zeros(state.nparams, dtype=dtype, device=sharding)
        real_dtype = jnp.finfo(dtype).dtype
        v = jnp.zeros(state.nparams, dtype=real_dtype, device=sharding)
        self._buffers = {"phi": phi, "v": v}
        super().__init__(state, hamiltonian, imag_time, solver, file)

    @partial(eqx.filter_jit, donate="all-except-first")
    def solve(
        self, Obar: jax.Array, Ebar: jax.Array, buffers: dict[str, jax.Array]
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        r"""
        Solve the MARCH optimization step.
        """
        phi = buffers["phi"]
        v = buffers["v"]
        Ebar -= self._mu * (Obar @ phi)
        V = jnp.where(jnp.allclose(v, 0), jnp.ones_like(v), v**0.25 + 1e-8)

        Obar /= V[None, :]
        step, buffers = SR.solve(self, Obar, Ebar, buffers)
        step /= V
        if self._norm_clip is not None:
            norm = jnp.asarray(jnp.linalg.norm(step))
            step = jnp.where(
                norm > self._norm_clip, step * (self._norm_clip / norm), step
            )
        step = step + self._mu * phi

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
            default to `~quantax.optimizer.auto_pinv_eig`.

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

        self._mu = mu
        self._beta = beta
        self._norm_clip = norm_clip
        dtype = get_default_dtype()
        sharding = get_replicated_sharding()
        x0 = jnp.zeros(state.nparams, dtype=dtype, device=sharding)
        m = jnp.zeros(state.nparams, dtype=dtype, device=sharding)
        real_dtype = jnp.finfo(dtype).dtype
        v = jnp.zeros(state.nparams, dtype=real_dtype, device=sharding)
        t = jnp.zeros((), jnp.int32, device=sharding)
        self._buffers = {"x0": x0, "m": m, "v": v, "t": t}
        SR.__init__(self, state, hamiltonian, imag_time, solver, file)

    @partial(eqx.filter_jit, donate="all-except-first")
    def solve(
        self, Obar: jax.Array, Ebar: jax.Array, buffers: dict[str, jax.Array]
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        r"""
        Solve the AdamSR optimization step. The time cost is roughly twice of SR.
        """
        g, buffers = SR.solve(self, Obar, Ebar, buffers)
        if self._norm_clip is not None:
            g_norm = jnp.asarray(jnp.linalg.norm(g))
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
        del buffers["x0"]

        mhat = m / (1 - self._mu**t).astype(m.dtype)
        vhat = v / (1 - self._beta**t).astype(v.dtype)
        V = vhat**0.25 + 1e-8

        Ebar -= Obar @ mhat
        Obar /= V[None, :]
        step, buffers = SR.solve(self, Obar, Ebar, buffers)
        step = step / V + mhat

        buffers["x0"] = mhat
        return step, buffers


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
            The numerical solver for the matrix inverse, default to `~quantax.optimizer.auto_pinv_eig`.

        :param symm:
            Symmetry used to construct the Hilbert space, default to be the symmetry
            of the variational state.
        """
        ExactQNGD.__init__(self, state, imag_time, solver, symm)
        self._hamiltonian = hamiltonian
        self._energy = None

    @property
    def hamiltonian(self) -> Operator:
        """The Hamiltonian for the evolution."""
        return self._hamiltonian

    @property
    def energy(self) -> ArrayLike | None:
        """Energy of the current step."""
        return self._energy

    def get_Ebar(self, psi: jax.Array) -> jax.Array:
        r"""Compute :math:`\bar \epsilon` in the full Hilbert space."""
        dense = DenseState(psi[: self._Ns], self._symm)
        H_psi = self._hamiltonian @ dense
        energy = dense @ H_psi
        self._energy = jnp.asarray(energy.real)
        Ebar = H_psi - dense * energy
        Ebar = jnp.asarray(Ebar.psi)
        return array_extend(Ebar, psi.size)
