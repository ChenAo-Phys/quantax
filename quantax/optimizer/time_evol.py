from typing import Union, Tuple, Optional, Callable
from numbers import Number
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

from .sr import SR
from .solver import pinvh_solve
from ..state import DenseState, Variational, VS_TYPE
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


class ExactTimeEvol:
    def __init__(self, init_state: DenseState, hamiltonian: Operator) -> None:
        self._init_state = init_state
        self._symm = self._init_state.symm
        self._eigs, self._U = hamiltonian.diagonalize(self._symm, "full")
        self._eigs = jnp.asarray(self._eigs)
        self._U = jnp.asarray(self._U)

    def get_evolved_psi(self, time: Union[float, jax.Array]) -> jax.Array:
        is_float = isinstance(time, float)
        if is_float:
            time = jnp.array([time])
        exp_eigs = jnp.exp(-1j * jnp.einsum("t,d->td", time, self._eigs))
        psi0 = self._init_state.psi
        psi = jnp.einsum("ij,tj,kj,k->ti", self._U, exp_eigs, self._U.conj(), psi0)
        if is_float:
            psi = psi[0]
        return psi

    def expectation(
        self, operator: Operator, time: Union[float, jax.Array]
    ) -> Union[Number, jax.Array]:
        psi = self.get_evolved_psi(time)
        psi /= jnp.linalg.norm(psi, axis=1, keepdims=True)
        op = operator.get_quspin_op(self._symm)

        # this hasn't been tested
        out = jnp.einsum("ti,it->t", psi.conj(), op.dot(np.ascontiguousarray(psi.T)))
        if isinstance(time, float):
            out = out.item()
        return out
