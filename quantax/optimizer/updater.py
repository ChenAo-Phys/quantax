from typing import Callable, Any
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from ..utils import get_replicated_sharding
from ..global_defs import get_default_dtype


def _zeros(shape: Any, dtype: DTypeLike) -> jax.Array:
    return jnp.zeros(shape, dtype=dtype, device=get_replicated_sharding())


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

    If the proposed step contains non-finite values, the ``phi`` buffer is left
    unchanged, so a single corrupted iteration can't poison the accumulated
    momentum. The non-finite step is still returned and is expected to be
    rejected by `~quantax.state.Variational.update`.
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
        finite = jnp.all(jnp.isfinite(step))
        buffers["phi"] = jnp.where(finite, step, phi)
        return step, buffers


class MarchUpdater(Updater):
    r"""
    The `MARCH <https://arxiv.org/abs/2507.02644>`_ update, a variant of SR with
    first and second order momentum stored in the ``phi`` and ``v`` buffers.

    If the proposed step or the new second-order momentum contains non-finite
    values, both buffers are left unchanged, so a single corrupted iteration
    can't poison the accumulated momentum. The non-finite step is still
    returned and is expected to be rejected by
    `~quantax.state.Variational.update`.
    """

    def __init__(self, mu: float = 0.95, beta: float = 0.995):
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
            "phi": _zeros(nparams, dtype),
            "v": jnp.ones(nparams, dtype=real_dtype, device=get_replicated_sharding()),
        }

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
        V = v**0.25 + 1e-8
        step = core_solve(Obar, Ebar, diag_preconditioner=V)
        step = step + self.mu * phi
        v_new = self.beta * v + jnp.abs(step - phi) ** 2
        finite = jnp.all(jnp.isfinite(step)) & jnp.all(jnp.isfinite(v_new))
        buffers["phi"] = jnp.where(finite, step, phi)
        buffers["v"] = jnp.where(finite, v_new, v)
        return step, buffers


class AdamUpdater(Updater):
    r"""
    The AdamSR update, a variant of SR with first and second order momentum
    (like Adam) stored in the ``m``, ``v`` and ``t`` buffers. The time cost is
    roughly twice of the plain update. The second order momentum enters as a
    ``diag_preconditioner`` argument: it is passed through to solvers that accept
    it (e.g. `~quantax.optimizer.lsmr`) and emulated by right preconditioning for
    those that don't.

    If the new momentum values contain non-finite entries (e.g. from a
    non-finite gradient solve), the ``m``, ``v`` and ``t`` buffers are left
    unchanged, so a single corrupted iteration can't poison the accumulated
    momentum. The resulting non-finite step is still returned and is expected
    to be rejected by `~quantax.state.Variational.update`.
    """

    def __init__(self, mu: float = 0.95, beta: float = 0.995):
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
            "t": _zeros((), jnp.int32),
        }

    def update(
        self,
        core_solve: Callable[..., jax.Array],
        Obar: jax.Array,
        Ebar: jax.Array,
        buffers: dict[str, jax.Array],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        g = core_solve(Obar, Ebar)

        m_new = self.mu * buffers["m"] + (1 - self.mu) * g
        v_new = self.beta * buffers["v"] + (1 - self.beta) * jnp.abs(g) ** 2
        finite = jnp.all(jnp.isfinite(m_new)) & jnp.all(jnp.isfinite(v_new))
        t = buffers["t"] + finite.astype(buffers["t"].dtype)
        m = jnp.where(finite, m_new, buffers["m"])
        v = jnp.where(finite, v_new, buffers["v"])
        buffers["t"] = t
        buffers["m"] = m
        buffers["v"] = v

        # t = 0 can only occur when every update so far was rejected; the
        # buffers are still zero there, so skip the 0/0 bias correction.
        mhat = jnp.where(t == 0, m, m / (1 - self.mu**t).astype(m.dtype))
        vhat = jnp.where(t == 0, v, v / (1 - self.beta**t).astype(v.dtype))
        V = vhat**0.25 + 1e-8

        Ebar = Ebar - Obar @ mhat
        step = core_solve(Obar, Ebar, diag_preconditioner=V) + mhat
        return step, buffers
