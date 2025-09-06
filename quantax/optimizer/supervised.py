from typing import Callable, Optional
import jax
import jax.numpy as jnp
from .sr import QNGD
from ..symmetry import Symmetry
from ..state import State, Variational
from ..sampler import Samples
from ..utils import ints_to_array
from ..global_defs import is_default_cpl


class Supervised(QNGD):
    def __init__(
        self,
        state: Variational,
        target_state: State,
        solver: Optional[Callable[[jax.Array, jax.Array], jax.Array]] = None,
    ):
        super().__init__(state, solver=solver)
        self._target_state = target_state

    def get_Ebar(self, samples: Samples) -> jax.Array:
        phi = self._target_state(samples.spins)
        psi = samples.psi
        ratio = phi / psi
        reweight = samples.reweight_factor

        ratio_mean = jnp.mean(ratio * reweight)
        ratio = ratio / ratio_mean - 1
        Ebar = -ratio * jnp.sqrt(reweight / samples.nsamples)
        return Ebar


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
        self._target_psi = target_state.todense(symm).psi[restricted_to]

    def get_epsilon(self, psi: jax.Array) -> jax.Array:
        return psi - self._target_psi / jnp.vdot(psi, self._target_psi)

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
