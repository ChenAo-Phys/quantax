from typing import Callable
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from ..utils import get_replicated_sharding
from ..global_defs import get_default_dtype


def _zeros(nparams: int, dtype: DTypeLike) -> jax.Array:
    return jnp.zeros(nparams, dtype=dtype, device=get_replicated_sharding())


class Updater:
    r"""
    Base class of update strategies for `~quantax.optimizer.QNGD` optimizers. It
    defines how an optimization step is generated around the core equation
    solve, including the persistent buffers it carries (momentum, initial
    guesses, ...). A concrete updater implements two methods:

    - `~quantax.optimizer.Updater.init` returns the buffers as a dict of arrays;
    - `~quantax.optimizer.Updater.update` returns ``(step, buffers)``, where
      ``core_solve(Obar, Ebar, **solver_kwargs)`` solves
      :math:`\bar O \dot \theta = \bar \epsilon` and forwards extra keyword
      arguments to the numerical solver.
    """

    def init(self, nparams: int) -> dict[str, jax.Array]:
        r"""Initialize the persistent buffers for ``nparams`` parameters."""
        raise NotImplementedError

    def update(
        self,
        core_solve: Callable[..., jax.Array],
        Obar: jax.Array,
        Ebar: jax.Array,
        buffers: dict[str, jax.Array],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        r"""
        Generate the optimization step and the new buffers, given the equation
        solver ``core_solve`` and the current ``buffers``.
        """
        raise NotImplementedError


class PlainUpdater(Updater):
    r"""
    The plain update :math:`\dot\theta = \mathrm{solve}(\bar O, \bar\epsilon)`.
    """

    def init(self, nparams: int) -> dict[str, jax.Array]:
        return {}

    def update(
        self,
        core_solve: Callable[..., jax.Array],
        Obar: jax.Array,
        Ebar: jax.Array,
        buffers: dict[str, jax.Array],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        step = core_solve(Obar, Ebar)
        return step, buffers


class SpringUpdater(Updater):
    r"""
    The `SPRING <https://doi.org/10.1016/j.jcp.2024.113351>`_ update, a variant
    of SR with momentum stored in the ``phi`` buffer.
    """

    def __init__(self, mu: float = 0.9):
        r"""
        :param mu:
            The momentum factor.
        """
        self.mu = mu

    def init(self, nparams: int) -> dict[str, jax.Array]:
        return {"phi": _zeros(nparams, get_default_dtype())}

    def update(
        self,
        core_solve: Callable[..., jax.Array],
        Obar: jax.Array,
        Ebar: jax.Array,
        buffers: dict[str, jax.Array],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        phi = buffers["phi"]
        Ebar = Ebar - self.mu * (Obar @ phi)
        step = core_solve(Obar, Ebar)
        step = step + self.mu * phi
        buffers["phi"] = step
        return step, buffers


class MarchUpdater(Updater):
    r"""
    The `MARCH <https://arxiv.org/abs/2507.02644>`_ update, a variant of SR with
    first and second order momentum stored in the ``phi`` and ``v`` buffers.
    """

    def __init__(
        self, mu: float = 0.95, beta: float = 0.995
    ):
        r"""
        :param mu:
            The first order momentum factor.

        :param beta:
            The second order momentum factor.
        """
        self.mu = mu
        self.beta = beta

    def init(self, nparams: int) -> dict[str, jax.Array]:
        dtype = get_default_dtype()
        real_dtype = jnp.finfo(dtype).dtype
        return {"phi": _zeros(nparams, dtype), "v": _zeros(nparams, real_dtype)}

    def update(
        self,
        core_solve: Callable[..., jax.Array],
        Obar: jax.Array,
        Ebar: jax.Array,
        buffers: dict[str, jax.Array],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        phi = buffers["phi"]
        v = buffers["v"]
        Ebar = Ebar - self.mu * (Obar @ phi)
        V = jnp.where(jnp.allclose(v, 0), jnp.ones_like(v), v**0.25 + 1e-8)
        step = core_solve(Obar, Ebar, diag_preconditioner=V)
        step = step + self.mu * phi
        buffers["phi"] = step
        buffers["v"] = self.beta * v + jnp.abs(step - phi) ** 2
        return step, buffers


class AdamUpdater(Updater):
    r"""
    The AdamSR update, a variant of SR with first and second order momentum
    (like Adam) stored in the ``m``, ``v`` and ``t`` buffers. The time cost is
    roughly twice of the plain update. The second order momentum enters as a
    ``diag_preconditioner`` argument: it is passed through to solvers that accept
    it (e.g. `~quantax.optimizer.lsmr`) and emulated by right preconditioning for
    those that don't.
    """

    def __init__(
        self, mu: float = 0.95, beta: float = 0.995
    ):
        r"""
        :param mu:
            The first order momentum factor.

        :param beta:
            The second order momentum factor.
        """
        self.mu = mu
        self.beta = beta

    def init(self, nparams: int) -> dict[str, jax.Array]:
        dtype = get_default_dtype()
        real_dtype = jnp.finfo(dtype).dtype
        return {
            "m": _zeros(nparams, dtype),
            "v": _zeros(nparams, real_dtype),
            "t": jnp.zeros((), jnp.int32, device=get_replicated_sharding()),
        }

    def update(
        self,
        core_solve: Callable[..., jax.Array],
        Obar: jax.Array,
        Ebar: jax.Array,
        buffers: dict[str, jax.Array],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        g = core_solve(Obar, Ebar)

        t = buffers["t"] + 1
        m = self.mu * buffers["m"] + (1 - self.mu) * g
        v = self.beta * buffers["v"] + (1 - self.beta) * jnp.abs(g) ** 2
        buffers["t"] = t
        buffers["m"] = m
        buffers["v"] = v

        mhat = m / (1 - self.mu**t).astype(m.dtype)
        vhat = v / (1 - self.beta**t).astype(v.dtype)
        V = vhat**0.25 + 1e-8

        Ebar = Ebar - Obar @ mhat
        step = core_solve(Obar, Ebar, diag_preconditioner=V) + mhat
        return step, buffers
