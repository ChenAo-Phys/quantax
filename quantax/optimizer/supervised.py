from typing import Callable, BinaryIO
from pathlib import Path
import jax
import jax.numpy as jnp
from .qngd import StochasticQNGD, ExactQNGD
from .sr import AdamSR
from ..symmetry import Symmetry
from ..state import State, Variational
from ..sampler import Samples
from ..utils import ints_to_array
from ..global_defs import is_default_cpl


class Supervised(StochasticQNGD):
    def __init__(
        self,
        state: Variational,
        target_state: State,
        solver: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
        file: str | Path | BinaryIO | None = None,
        clip: float | None = None,
    ):
        StochasticQNGD.__init__(self, state, solver=solver, file=file)
        self._target_state = target_state
        self._clip = clip

    def get_Ebar(self, samples: Samples) -> jax.Array:
        if samples.psi is None:
            psi = self.state(samples.spins)
        else:
            psi = samples.psi
        if samples.reweight_factor is None:
            reweight = 1
        else:
            reweight = samples.reweight_factor

        phi = self._target_state(samples.spins)
        ratio = phi / psi
        ratio_mean = (ratio * reweight).mean()
        ratio = jnp.asarray(ratio / ratio_mean) - 1
        if self._clip is not None:
            ratio = jnp.clip(ratio, -self._clip, self._clip)
        Ebar = -ratio * jnp.sqrt(reweight / samples.nsamples)
        return Ebar


class SupervisedAdam(Supervised, AdamSR):
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
        Supervised.__init__(self, state, target_state, solver, file, clip)
        AdamSR.__init__(self, state, None, True, solver, file, mu, beta, norm_clip)  # type: ignore


class SupervisedExact(ExactQNGD):
    def __init__(
        self,
        state: Variational,
        target_state: State,
        solver: Callable | None = None,
        symm: Symmetry | None = None,
    ):
        ExactQNGD.__init__(self, state, solver=solver)
        self._target_state = target_state

        if symm is None:
            symm = state.symm
        self._symm = symm
        symm.basis_make()
        basis = symm.basis
        self._spins = ints_to_array(basis.states)
        self._symm_norm = jnp.asarray(basis.get_amp(basis.states))
        if not is_default_cpl():
            self._symm_norm = self._symm_norm.real
        self._target_psi = jnp.asarray(target_state.todense(symm).psi)

    def get_Ebar(self, psi: jax.Array) -> jax.Array:
        return psi - self._target_psi / jnp.vdot(psi, self._target_psi)
