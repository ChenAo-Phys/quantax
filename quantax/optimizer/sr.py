from typing import Callable, BinaryIO
from jax.typing import ArrayLike
from pathlib import Path
from functools import partial
import jax
import jax.numpy as jnp

from .qngd import StochasticQNGD, ExactQNGD
from .updater import SpringUpdater, MarchUpdater, AdamUpdater
from .gradient import EnergyGrad
from .solver import pinvh_solve
from ..state import Variational, VS_TYPE
from ..operator import Operator
from ..sampler import Samples
from ..symmetry import Symmetry
from ..utils import get_distributed_sharding, array_extend
from ..global_defs import get_default_dtype


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
        clip: float | None = 5.0,
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

        :param clip:
            Local-energy clipping passed to `~quantax.optimizer.EnergyGrad`,
            default to 5.0. Set to None to disable, in particular for real-time
            evolution (``imag_time=False``), where clipping biases the dynamics.
        """
        grad = EnergyGrad(hamiltonian, clip)
        StochasticQNGD.__init__(self, state, grad, imag_time, solver, file=file)


class SPRING(StochasticQNGD):
    r"""
    `SPRING optimizer <https://doi.org/10.1016/j.jcp.2024.113351>`_.
    This is a variant of SR with momentum.

    The momentum makes large steps possible, so it is recommended to constrain
    the norm of the applied update in the training loop (see the note in
    `~quantax.optimizer.MARCH`).
    """

    def __init__(
        self,
        state: Variational,
        hamiltonian: Operator,
        imag_time: bool = True,
        solver: Callable[..., jax.Array] | None = None,
        file: str | Path | BinaryIO | None = None,
        mu: float = 0.9,
        clip: float | None = 5.0,
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

        :param clip:
            Local-energy clipping passed to `~quantax.optimizer.EnergyGrad`,
            default to 5.0. Set to None to disable.
        """
        grad = EnergyGrad(hamiltonian, clip)
        updater = SpringUpdater(mu)
        StochasticQNGD.__init__(self, state, grad, imag_time, solver, updater, file)


class MARCH(StochasticQNGD):
    r"""
    `MARCH optimizer <https://arxiv.org/abs/2507.02644>`_.
    This is a variant of SR with first and second order momentum (like Adam).

    The recommended usage follows the original paper: unit learning rate with a
    norm constraint on the applied update as the step-size control, e.g.

    .. code-block:: python

        step = optimizer.get_step(samples)
        c = C0 / (1 + max(t - t0, 0) / T)  # constraint schedule
        state.update(step * min(1, c / jnp.linalg.norm(step)))

    The constraint should be of order 1 (the paper uses ``C0=0.1``; values up to
    ~1 are typically stable, while unconstrained momentum steps can diverge).
    The momentum buffers intentionally track the unconstrained step.
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
        clip: float | None = 5.0,
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

        :param clip:
            Local-energy clipping passed to `~quantax.optimizer.EnergyGrad`,
            default to 5.0 as in the MARCH paper. Set to None to disable.
        """
        grad = EnergyGrad(hamiltonian, clip)
        updater = MarchUpdater(mu, beta)
        StochasticQNGD.__init__(self, state, grad, imag_time, solver, updater, file)


class AdamSR(StochasticQNGD):
    r"""
    AdamSR optimizer.
    This is a variant of SR with first and second order momentum (like Adam).
    The time cost is roughly twice of SR.

    A fixed learning rate leaves the parameters random-walking in a noise ball;
    for converged energies, decay the learning rate in the training loop, e.g.
    ``lr / (1 + max(t - t0, 0) / t0)``. Constraining the norm of the applied
    update (see `~quantax.optimizer.MARCH`) protects the ignition phase.
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
        clip: float | None = 5.0,
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

        :param clip:
            Local-energy clipping passed to `~quantax.optimizer.EnergyGrad`,
            default to 5.0. Set to None to disable.
        """
        grad = EnergyGrad(hamiltonian, clip)
        updater = AdamUpdater(mu, beta)
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


@jax.jit
def _AconjB(A: jax.Array, B: jax.Array) -> jax.Array:
    matmul = lambda x, y: x.conj().T @ y
    if A.ndim == 2:
        return matmul(A, B)
    elif A.ndim == 3:
        return jax.vmap(matmul)(A, B)
    else:
        raise NotImplementedError


class TimeEvol(SR):
    r"""
    Time evolution optimizer, equivalent to real-time `~quantax.optimizer.SR`.
    This optimizer assumes the number of samples is more than the number of parameters,
    and is more memory-efficient than `~quantax.optimizer.SR` when this is the case.
    """

    def __init__(
        self,
        state: Variational,
        hamiltonian: Operator,
        solver: Callable[..., jax.Array] | None = None,
    ):
        r"""
        :param state:
            Variational state to be evolved.

        :param hamiltonian:
            Hamiltonian operator for time evolution.

        :param solver:
            The numerical solver for :math:`Sx = F`, default to pseudo-inverse.
        """

        if solver is None:
            solver = pinvh_solve()
        # clip=None: local-energy clipping is a biased operation that would
        # introduce a systematic error in the real-time dynamics
        super().__init__(state, hamiltonian, imag_time=False, solver=solver, clip=None)
        self._hamiltonian = hamiltonian
        self._max_parallel = state._backward_chunk
        self._energy = None
        self._VarE = None

    @property
    def energy(self) -> ArrayLike | None:
        """Energy of the current step."""
        return self._energy

    @property
    def VarE(self) -> ArrayLike | None:
        r"""Energy variance :math:`\left< (H - E)^2 \right>` of the current step."""
        return self._VarE

    def _get_SF_direct(self, samples: Samples) -> tuple[jax.Array, jax.Array]:
        Ebar = self.get_Ebar(samples)
        self._energy = getattr(self._grad, "energy", None)
        self._VarE = getattr(self._grad, "VarE", None)
        Obar = self.get_Obar(samples)
        Smat = _AconjB(Obar, Obar)
        Fvec = _AconjB(Obar, Ebar)
        return Smat, Fvec

    def _get_SF_indirect(self, samples: Samples) -> tuple[jax.Array, jax.Array]:
        if self._max_parallel is None:
            return self._get_SF_direct(samples)

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

        nsamples, Nmodes = samples.spins.shape
        spins = samples.spins.reshape(ndevices, -1, Nmodes)
        spins = array_extend(spins, self._max_parallel, 1, padding_values=1)
        spins = jnp.split(spins, nsplits, axis=1)

        nparams = self._state.nparams
        dtype = get_default_dtype()
        sharding = get_distributed_sharding()
        Smat = jnp.zeros((ndevices, nparams, nparams), dtype, device=sharding)
        Fvec = jnp.zeros((ndevices, nparams), dtype, device=sharding)
        Omean = jnp.zeros((ndevices, nparams), dtype, device=sharding)
        for s, e in zip(spins, Eloc):
            Omat = self._state.jacobian(s.reshape(-1, Nmodes))
            Omat = Omat.reshape(ndevices, -1, nparams).astype(dtype)
            Omean += jnp.sum(Omat, axis=1)
            newS = _AconjB(Omat, Omat)
            newF = _AconjB(Omat, e)
            Smat += newS
            Fvec += newF
        Smat = jnp.sum(Smat, axis=0) / nsamples
        Fvec = jnp.sum(Fvec, axis=0) / nsamples
        Omean = jnp.sum(Omean, axis=0) / nsamples
        self._Omean = Omean

        Smat = Smat - jnp.outer(Omean.conj(), Omean)
        Fvec = Fvec - Omean.conj() * Emean
        return Smat, Fvec

    def get_SF(self, samples: Samples) -> tuple[jax.Array, jax.Array]:
        r"""
        Compute :math:`S = \bar O^\dagger \bar O` and :math:`F = \bar O^\dagger \bar \epsilon`
        with the given samples. When the number of samples is large, this function will
        automatically switch to a more memory-efficient implementation.
        """
        if (
            self._max_parallel is None
            or samples.nsamples <= self._max_parallel * jax.device_count()
        ):
            return self._get_SF_direct(samples)
        else:
            return self._get_SF_indirect(samples)

    @partial(jax.jit, static_argnums=0)
    def solve_SF(self, Smat: jax.Array, Fvec: jax.Array) -> jax.Array:
        r"""
        Solve the time evolution equation :math:`S \dot\theta = F` for the
        parameter update, given the matrix :math:`S` and vector :math:`F` from
        `~quantax.optimizer.TimeEvol.get_SF`.
        """
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

    def get_step(self, samples: Samples | jax.Array) -> jax.Array:
        if not isinstance(samples, Samples):
            samples = Samples(samples)

        reweight = samples.reweight_factor
        if reweight is not None and not jnp.allclose(reweight, 1.0):
            raise ValueError("TimeEvol is only for non-reweighted samples")

        Smat, Fvec = self.get_SF(samples)
        step = self.solve_SF(Smat, Fvec)
        return step
