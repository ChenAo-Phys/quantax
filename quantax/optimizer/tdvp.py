from typing import Optional, Callable, Tuple
from functools import partial
import jax
import jax.numpy as jnp

from .solver import auto_pinv_eig, pinvh_solve
from ..state import DenseState, Variational, VS_TYPE
from ..sampler import Samples
from ..operator import Operator
from ..symmetry import Symmetry
from ..utils import to_global_array, array_extend, ints_to_array
from ..global_defs import get_default_dtype, is_default_cpl


class QNGD:
    """
    Quantum Natural Gradient Descent.
    """

    def __init__(
        self,
        state: Variational,
        imag_time: bool = True,
        solver: Optional[Callable] = None,
        use_kazcmarz: bool = False,
    ):
        self._state = state
        self._holomorphic = state._holomorphic
        self._imag_time = imag_time
        if solver is None:
            solver = auto_pinv_eig()
        self._solver = solver
        self._use_kazcmarz = use_kazcmarz
        self._last_step = None
        self._Omean = None

    @property
    def state(self) -> Variational:
        return self._state

    @property
    def holomorphic(self) -> bool:
        return self._holomorphic

    @property
    def vs_type(self) -> int:
        return self._state.vs_type

    @property
    def imag_time(self) -> bool:
        return self.imag_time

    @property
    def use_kazcmarz(self) -> bool:
        return self._use_kazcmarz

    def get_Ebar(self) -> jax.Array:
        """Method for computing epsilon in QNGD equaion, specified by the task."""

    def get_Obar(self, samples: Samples) -> jax.Array:
        Omat = self._state.jacobian(samples.spins).astype(get_default_dtype())
        self._Omean = jnp.mean(Omat * samples.reweight_factor[:, None], axis=0)
        Omat -= jnp.mean(Omat, axis=0, keepdims=True)
        Omat *= jnp.sqrt(samples.reweight_factor / samples.nsamples)[:, None]
        return Omat

    def solve(self, Obar: jax.Array, Ebar: jax.Array) -> jax.Array:
        use_kazcmarz = self._use_kazcmarz and self._last_step is not None
        if use_kazcmarz:
            Ebar -= Obar @ self._last_step

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

        if use_kazcmarz:
            step += self._last_step
            self._last_step = step
        return step

    def get_step(self, samples: Samples) -> jax.Array:
        Ebar = self.get_Ebar(samples)
        Obar = self.get_Obar(samples)
        step = self.solve(Obar, Ebar)
        return step


class TDVP(QNGD):
    def __init__(
        self,
        state: Variational,
        hamiltonian: Operator,
        imag_time: bool = True,
        solver: Optional[Callable] = None,
        use_kazcmarz: bool = False,
    ):
        super().__init__(state, imag_time, solver, use_kazcmarz)
        self._hamiltonian = hamiltonian

        self._energy = None
        self._VarE = None
        self._Omean = None

    @property
    def hamiltonian(self) -> Operator:
        return self._hamiltonian

    @property
    def energy(self) -> Optional[float]:
        return self._energy

    @property
    def VarE(self) -> Optional[float]:
        return self._VarE

    def get_Ebar(self, samples: Samples) -> jax.Array:
        Eloc = self._hamiltonian.Oloc(self._state, samples).astype(get_default_dtype())
        Emean = jnp.mean(Eloc * samples.reweight_factor)
        self._energy = Emean.real
        Evar = jnp.abs(Eloc - Emean) ** 2
        self._VarE = jnp.mean(Evar * samples.reweight_factor).real

        Eloc -= jnp.mean(Eloc)
        Eloc *= jnp.sqrt(samples.reweight_factor / samples.nsamples)
        return Eloc


class TDVP_exact(TDVP):
    def __init__(
        self,
        state: Variational,
        hamiltonian: Operator,
        imag_time: bool = True,
        solver: Optional[Callable] = None,
        symm: Optional[Symmetry] = None,
    ):
        super().__init__(state, hamiltonian, imag_time, solver)

        self._symm = state.symm if symm is None else symm
        self._symm.basis_make()
        basis = self._symm.basis
        self._spins = ints_to_array(basis.states)
        self._symm_norm = jnp.asarray(basis.get_amp(basis.states))
        if not is_default_cpl():
            self._symm_norm = self._symm_norm.real

    def get_Ebar(self, wave_function: jax.Array) -> jax.Array:
        psi = DenseState(wave_function, self._symm)
        H_psi = self._hamiltonian @ psi
        energy = psi @ H_psi
        Ebar = H_psi - energy * psi
        self._energy = energy.real
        return Ebar.wave_function

    def get_Obar(self, wave_function: jax.Array) -> jax.Array:
        Omat = self._state.jacobian(self._spins) * wave_function[:, None]
        self._Omean = jnp.einsum("s,sk->k", wave_function.conj(), Omat)
        Omean = jnp.einsum("s,k->sk", wave_function, self._Omean)
        return Omat - Omean

    def get_step(self) -> jax.Array:
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
        max_parallel: Optional[int] = None,
    ):
        if solver is None:
            solver = pinvh_solve()
        super().__init__(state, hamiltonian, imag_time=False, solver=solver)
        self._max_parallel = max_parallel

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

        nsamples, nsites = samples.spins.shape
        spins = samples.spins.reshape(ndevices, -1, nsites)
        spins = array_extend(spins, self._max_parallel, 1, padding_values=1)
        spins = jnp.split(spins, nsplits, axis=1)

        nparams = self._state.nparams
        dtype = get_default_dtype()
        Smat = to_global_array(jnp.zeros((ndevices, nparams, nparams), dtype))
        Fvec = to_global_array(jnp.zeros((ndevices, nparams), dtype))
        Omean = to_global_array(jnp.zeros((ndevices, nparams), dtype))
        for s, e in zip(spins, Eloc):
            Omat = self._state.jacobian(s.reshape(-1, nsites))
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

    def get_step(self, samples: Samples) -> jax.Array:
        if not jnp.allclose(samples.reweight_factor, 1.0):
            raise ValueError("TimeEvol is only for non-reweighted samples")

        Smat, Fvec = self.get_SF(samples)
        step = self.solve(Smat, Fvec)
        return step

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
