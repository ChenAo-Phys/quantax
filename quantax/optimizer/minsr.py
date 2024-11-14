from typing import Optional, Callable, Tuple, List, Any
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from .tdvp import TDVP
from .solver import minsr_pinv_eig
from ..state import Variational, VS_TYPE
from ..nn import Sequential, filter_vjp, RawInputLayer
from ..sampler import Samples
from ..operator import Operator
from ..utils import (
    tree_fully_flatten,
    to_global_array,
    get_replicate_sharding,
    array_extend,
    chunk_map,
)
from ..global_defs import get_default_dtype


class MinSR(TDVP):
    """
    MinSR optimization, specifically designed for `~quantax.nn.Sequential` networks.
    The optimization utilizes gradient checkpointing method and structured derivatives
    to reduce the memory cost.
    See `MinSR paper <https://www.nature.com/articles/s41567-024-02566-1#Sec25>`_ for details.
    """

    def __init__(
        self,
        state: Variational,
        hamiltonian: Operator,
        solver: Optional[Callable] = None,
    ):
        r"""
        :param state:
            Variational state to be optimized.

        :param hamiltonian:
            The Hamiltonian for the evolution.

        :param solver:
            The numerical solver for the matrix inverse, default to `~quantax.optimizer.minsr_pinv_eig`.

        ...warning::

            The model must be `~quantax.nn.Sequential`, otherwise one should use
            `~quantax.optimizer.TDVP`.

            The vs_type of the variational state should be ``real_or_holomorphic`` or
            ``real_to_complex``. In the latter case, the complex neurons are only allowed
            in the last few unparametrized layers.
        """
        if solver is None:
            solver = minsr_pinv_eig()
        super().__init__(state, hamiltonian, imag_time=True, solver=solver)

        if self.vs_type == VS_TYPE.non_holomorphic:
            raise ValueError(
                "'MinSR' optimizer doesn't support non-holomorphic complex networks"
            )

        params, others = state.partition(state.model)
        nparams = state.nparams
        if nparams == 0:
            raise ValueError("The variational state has no parameter.")
        elif not isinstance(params, Sequential):
            raise ValueError("The variational state is not `Sequential`.")
        else:
            self._nodes = []
            idx_parametrized = []
            for i, layer in enumerate(params):
                vals = tree_fully_flatten(layer)
                if vals.size > 0:
                    idx_parametrized.append(i)

            block_len = np.sqrt(len(idx_parametrized)).astype(int)
            for i, l in enumerate(idx_parametrized):
                if i % block_len == block_len - 1 or i == len(idx_parametrized) - 1:
                    self._nodes.append(l)

    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, None, 0))
    def _forward(self, model: Sequential, s: jax.Array) -> Tuple[jax.Array, jax.Array]:
        s_symm = self.state.symm.get_symm_spins(s)
        x = s_symm
        neurons = [x]

        for i, layer in enumerate(model):
            if isinstance(layer, RawInputLayer):
                x = jax.vmap(layer)(x, s_symm)
            else:
                x = jax.vmap(layer)(x)
            if i == self._nodes[-1]:
                is_complex = jnp.iscomplexobj(x)
                if self.vs_type == VS_TYPE.real_to_complex and is_complex:
                    raise NotImplementedError(
                        "'MinSR' optimizer accepts real parameters to "
                        "complex outputs only when complex neurons appear in the last "
                        "few unparametrized layers."
                    )
                outputs = x
                break
            if i in self._nodes:
                neurons.append(x)
        remainings = jax.vmap(model[i + 1 :]) if i < len(model) - 1 else lambda x, s: x

        def output_fn(x: jax.Array) -> jax.Array:
            x = remainings(x, s=s_symm)
            psi = self.state.symm.symmetrize(x, s)
            return psi / jax.lax.stop_gradient(psi)

        if self.vs_type == VS_TYPE.real_or_holomorphic:
            jac_out = jax.grad(output_fn, holomorphic=self.state.holomorphic)
            delta = jac_out(outputs)
        else:
            jac_real = jax.grad(lambda x: output_fn(x).real)(outputs)
            jac_imag = jax.grad(lambda x: output_fn(x).imag)(outputs)
            delta = jax.lax.complex(jac_real, jac_imag)
        return neurons, delta

    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, None, 0, 0, 0))
    def _layer_backward(
        self, layers: Sequential, neuron: jax.Array, s: jax.Array, delta: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:

        @partial(jax.vmap, in_axes=(None, 0, 0, 0))
        def backward(net, x, s, delta):
            forward = lambda net, x: net(x, s=s)
            f_vjp = filter_vjp(forward, net, x)[1]
            vjp_vals, delta = f_vjp(delta)
            return tree_fully_flatten(vjp_vals), delta

        if self.vs_type == VS_TYPE.real_or_holomorphic:
            grad, delta = backward(layers, neuron, s, delta)
        else:
            grad_real_out, delta_real = backward(layers, neuron, s, delta.real)
            grad_imag_out, delta_imag = backward(layers, neuron, s, delta.imag)
            grad = jax.lax.complex(grad_real_out, grad_imag_out)
            if delta_real is None:
                delta = None
            else:
                delta = jax.lax.complex(delta_real, delta_imag)
        grad = jnp.sum(grad.astype(get_default_dtype()), axis=0)
        return grad, delta

    @eqx.filter_jit
    def _reversed_scan_layers(
        self,
        model: Sequential,
        s: jax.Array,
        fn_on_jac: Callable,
        init_vals: Any,
    ):
        forward_chunk = self.state.forward_chunk
        forward = chunk_map(self._forward, in_axes=(None, 0), chunk_size=forward_chunk)
        neurons, delta = forward(model, s)
        s_symm = jax.vmap(self.state.symm.get_symm_spins)(s)
        end = self._nodes[-1] + 1
        nodes = [0] + [n + 1 for n in self._nodes[:-1]]
        backward_chunk = self.state.backward_chunk
        layer_backward = chunk_map(
            self._layer_backward, in_axes=(None, 0, 0, 0), chunk_size=backward_chunk
        )

        for start, neuron in zip(reversed(nodes), reversed(neurons)):
            layers = model[start:end]
            end = start
            jac, delta = layer_backward(layers, neuron, s_symm, delta)
            if jac.size > 0:
                init_vals = fn_on_jac(jac, init_vals)
        return init_vals

    @partial(jax.jit, static_argnums=0, donate_argnums=1)
    def _get_Obar(self, Omat: jax.Array, reweight: jax.Array) -> jax.Array:
        Omat -= jnp.mean(Omat, axis=0, keepdims=True)
        Omat *= jnp.sqrt(reweight[:, None] / Omat.shape[0])
        if self.vs_type != VS_TYPE.real_or_holomorphic:
            Omat = jnp.concatenate([Omat.real, Omat.imag], axis=0)
        return Omat

    @partial(jax.jit, static_argnums=0, donate_argnums=1)
    def _Tmat_scan_fn(
        self, jac: jax.Array, vals: Tuple[jax.Array, jax.Array]
    ) -> Tuple[jax.Array, jax.Array]:
        Tmat, reweight = vals
        Obar = self._get_Obar(jac, reweight)
        Obar = array_extend(Obar, jax.device_count(), axis=1)
        Obar = to_global_array(Obar.T).T  # sharded in axis=1
        Tmat += Obar @ Obar.conj().T
        return Tmat, reweight

    def get_Tmat(self, samples: Samples) -> jax.Array:
        """Compute the :math:`T` matrix in MinSR"""
        if self.vs_type == VS_TYPE.real_or_holomorphic:
            Ns = samples.nsamples
            dtype = get_default_dtype()
        else:
            Ns = 2 * samples.nsamples
            if get_default_dtype() in (jnp.float64, jnp.complex128):
                dtype = jnp.float64
            else:
                dtype = jnp.float32
        Tmat = jnp.zeros((Ns, Ns), dtype=dtype, device=get_replicate_sharding())

        init_vals = (Tmat, samples.reweight_factor)
        Tmat, reweight = self._reversed_scan_layers(
            self.state.model, samples.spins, self._Tmat_scan_fn, init_vals
        )
        return Tmat

    @partial(jax.jit, static_argnums=0, donate_argnums=1)
    def _Ohvp_scan_fn(
        self, jac: jax.Array, vals: Tuple[List, jax.Array, jax.Array]
    ) -> Tuple[List, jax.Array, jax.Array]:
        outputs, reweight, vec = vals
        Obar = self._get_Obar(jac, reweight)
        Ohvp = Obar.conj().T @ vec
        outputs.append(Ohvp)
        return outputs, reweight, vec

    def Ohvp(self, samples: Samples, vec: jax.Array) -> jax.Array:
        r"""
        Compute :math:`\bar O^† v`. vec @ jac is used instead of vjp for better precision.
        """
        init_vals = ([], samples.reweight_factor, vec)
        outputs, reweight, vec = self._reversed_scan_layers(
            self.state.model, samples.spins, self._Ohvp_scan_fn, init_vals
        )
        outputs = jnp.concatenate(outputs[::-1])
        return outputs

    def solve(self, samples: Samples, Tmat: jax.Array, Ebar: jax.Array) -> jax.Array:
        if self.vs_type != VS_TYPE.real_or_holomorphic:
            Ebar = jnp.concatenate([Ebar.real, Ebar.imag])
        Tinv_E = self._solver(Tmat, Ebar)
        del Tmat, Ebar
        step = self.Ohvp(samples, Tinv_E)

        if self.vs_type == VS_TYPE.non_holomorphic:
            step = step.reshape(2, -1)
            step = step[0] + 1j * step[1]
        step = step.astype(get_default_dtype())
        return step

    def get_step(self, samples: Samples) -> jax.Array:
        """Obtain the MinSR step from given samples"""
        Ebar = self.get_Ebar(samples)
        Tmat = self.get_Tmat(samples)
        step = self.solve(samples, Tmat, Ebar)
        return step
