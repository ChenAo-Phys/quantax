from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

from ..state import Variational
from ..operator import Operator
from ..sampler import Samples
from ..utils import tree_fully_flatten
from ..global_defs import get_params_dtype


class CMAES:
    def __init__(
        self,
        state: Variational,
        hamiltonian: Operator,
        Nes: int = 200,
        sigma0: float = np.pi / 4,
    ) -> None:
        self._state = state
        hamiltonian.initialize()
        self._hamiltonian = hamiltonian

        import cma

        opts = cma.CMAOptions()
        opts.set("popsize", Nes)
        init_params = tree_fully_flatten(state._params)
        self._es = cma.CMAEvolutionStrategy(init_params, sigma0, opts)

        @partial(jax.jit, static_argnums=2)
        @partial(jax.vmap, in_axes=(0, None, None, None))
        def forward_fn(params, variables, static, spins):
            params = self._state.get_params_unflatten(params)
            psi = self._state._forward_fn(params, variables, static, spins)
            return psi

        self._forward = lambda params, spins: forward_fn(
            params, self._state._variables, self._state._static, spins
        )

    def Eloc(self, params: jax.Array, spins: jax.Array) -> jax.Array:
        Hz1, s_conn1, H_conn1, segment1 = self._hamiltonian._get_conn1(spins)
        Hz2, s_conn2, H_conn2, segment2 = self._hamiltonian._get_conn2(spins)
        Hz = Hz1 + Hz2

        n_conn = s_conn1.shape[0] + s_conn2.shape[0]
        if (
            hasattr(self._state, "max_parallel")
            and self._state.max_parallel is not None
        ):
            max_parallel = (
                self._state.max_parallel * jax.local_device_count() // self._state.nsymm
            )
            n_res = n_conn % max_parallel
            n_extend = max_parallel - n_res
        else:
            n_extend = 0
        s_extend = np.ones([n_extend, spins.shape[1]], spins.dtype)
        s_conn = jnp.asarray(np.concatenate([s_conn1, s_conn2, s_extend], axis=0))
        psi_conn = self._forward(params, s_conn)

        H_extend = np.zeros([n_extend], H_conn1.dtype)
        H_conn = jnp.asarray(np.concatenate([H_conn1, H_conn2, H_extend]))
        seg_extend = np.zeros([n_extend], segment1.dtype)
        segment = jnp.asarray(np.concatenate([segment1, segment2, seg_extend]))
        Hx = self._compute_Hx(psi_conn, H_conn, segment, spins.shape[0])
        psi = self._forward(params, spins)
        Hx /= psi
        return Hz + Hx

    @staticmethod
    @partial(jax.jit, static_argnums=3)
    @partial(jax.vmap, in_axes=(0, None, None, None))
    def _compute_Hx(psi_conn, H_conn, segment, num_segments):
        return jax.ops.segment_sum(
            psi_conn * H_conn, segment, num_segments=num_segments
        )

    def update(self, samples: Samples) -> None:
        dtype = get_params_dtype()

        es_ask = self._es.ask()
        new_params = jnp.asarray(np.stack(es_ask).astype(dtype))
        Eloc = self.Eloc(new_params, samples.spins)
        Emean = jnp.mean(Eloc, axis=1)
        self._energy = jnp.mean(Emean)
        self._es.tell(es_ask, np.asarray(Emean))
        self._state.set_params(jnp.asarray(self._es.result.xfavorite.astype(dtype)))
