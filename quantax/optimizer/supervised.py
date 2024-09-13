from typing import Callable, Optional
from functools import partial
import jax
import jax.numpy as jnp
from .solver import auto_pinv_eig
from ..symmetry import Symmetry
from ..state import State, Variational, VS_TYPE
from ..sampler import Samples
from ..utils import ints_to_array
from ..global_defs import get_default_dtype, is_default_cpl


class Supervised:
    def __init__(
        self, state: Variational, target_state: State, solver: Optional[Callable] = None
    ):
        self._state = state
        self._target_state = target_state
        if solver is None:
            solver = auto_pinv_eig()
        self._solver = solver

    @property
    def vs_type(self) -> int:
        return self._state.vs_type

    def get_epsilon(self, samples: Samples) -> jax.Array:
        phi = self._target_state(samples.spins)
        psi = samples.wave_function
        ratio = phi / psi
        reweight = samples.reweight_factor

        ratio_mean = jnp.mean(ratio * reweight)
        ratio = ratio / ratio_mean - 1
        epsilon = -ratio * jnp.sqrt(reweight / samples.nsamples)
        return epsilon

    def get_Obar(self, samples: Samples) -> jax.Array:
        Omat = self._state.jacobian(samples.spins).astype(get_default_dtype())
        Omat = Omat.reshape(samples.nsamples, -1)
        self._Omean = jnp.mean(Omat * samples.reweight_factor[:, None], axis=0)
        Omat -= self._Omean[None, :]
        Omat *= jnp.sqrt(samples.reweight_factor / samples.nsamples)[:, None]
        return Omat

    def get_step(self, samples: Samples) -> jax.Array:
        epsilon = self.get_epsilon(samples)
        Obar = self.get_Obar(samples)
        step = self._solver(Obar, epsilon)
        return step

    @partial(jax.jit, static_argnums=0)
    def solve(self, Obar: jax.Array, Ebar: jax.Array) -> jax.Array:
        if not self.vs_type == VS_TYPE.real_or_holomorphic:
            Obar = jnp.concatenate([Obar.real, Obar.imag], axis=0)
            Ebar = jnp.concatenate([Ebar.real, Ebar.imag])

        step = self._solver(Obar, Ebar)

        if self.vs_type == VS_TYPE.non_holomorphic:
            step = step.reshape(2, -1)
            step = step[0] + 1j * step[1]
        step = step.astype(get_default_dtype())
        return step


class Supervised_exact(Supervised):
    def __init__(
        self,
        state: Variational,
        target_state: State,
        solver: Optional[Callable] = None,
        symm: Optional[Symmetry] = None,
        restricted_to: Optional[jax.Array] = None,
    ):
        super().__init__(state, target_state, solver)

        if symm is None:
            symm = state.symm
        self._symm = symm
        symm.basis_make()
        basis = symm.basis
        self._spins = ints_to_array(basis.states)
        self._symm_norm = jnp.asarray(basis.get_amp(basis.states))
        if not is_default_cpl():
            self._symm_norm = self._symm_norm.real

        if restricted_to is None:
            restricted_to = jnp.arange(basis.Ns)
        else:
            restricted_to = jnp.asarray(restricted_to).flatten()
        self._resctricted_to = restricted_to
        self._target_wf = target_state.todense(symm).wave_function[restricted_to]

    def get_epsilon(self, psi: jax.Array) -> jax.Array:
        return psi - self._target_wf / jnp.vdot(psi, self._target_wf)

    def get_Obar(self, psi: jax.Array) -> jax.Array:
        Omat = self._state.jacobian(self._spins[self._resctricted_to]) * psi[:, None]
        self._Omean = jnp.einsum("s,sk->k", psi.conj(), Omat)
        Omean = jnp.einsum("s,k->sk", psi, self._Omean)
        return Omat - Omean

    def get_step(self) -> jax.Array:
        psi = self._state(self._spins) / self._symm_norm
        self._psi = psi / jnp.linalg.norm(psi)
        psi = self._psi[self._resctricted_to]
        epsilon = self.get_epsilon(psi)
        Obar = self.get_Obar(psi)
        step = self.solve(Obar, epsilon)
        return step
