from typing import Tuple, Optional, Callable
from functools import partial
import jax
import jax.numpy as jnp

from .sr import SR
from .solver import pinvh_solve
from ..state import Variational, VS_TYPE
from ..operator import Operator
from ..sampler import Samples
from ..utils import get_distribute_sharding, array_extend
from ..global_defs import get_default_dtype


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
        solver: Optional[Callable] = None,
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
        super().__init__(state, hamiltonian, imag_time=False, solver=solver)
        self._max_parallel = state._backward_chunk

    def get_SF(self, samples: Samples) -> Tuple[jax.Array, jax.Array]:
        r"""
        Compute :math:`S = \bar O^\dagger \bar O` and :math:`F = \bar O^\dagger \bar \epsilon`
        with the given samples. When the number of samples is large, this function will
        automatically switch to a more memory-efficient implementation.
        """
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

        nsamples, Nmodes = samples.spins.shape
        spins = samples.spins.reshape(ndevices, -1, Nmodes)
        spins = array_extend(spins, self._max_parallel, 1, padding_values=1)
        spins = jnp.split(spins, nsplits, axis=1)

        nparams = self._state.nparams
        dtype = get_default_dtype()
        sharding = get_distribute_sharding()
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
