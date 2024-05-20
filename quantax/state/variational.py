from __future__ import annotations
from typing import Callable, Optional, Tuple, Union, Sequence, BinaryIO
from jaxtyping import PyTree
from pathlib import Path

from warnings import warn
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import jax.flatten_util as jfu
import equinox as eqx

from .state import State
from ..symmetry import Symmetry
from ..nn import NoGradLayer, filter_vjp
from ..utils import (
    is_sharded_array,
    to_array_shard,
    filter_replicate,
    array_extend,
    tree_fully_flatten,
    tree_split_cpl,
    tree_combine_cpl,
)
from ..global_defs import (
    get_params_dtype,
    is_params_cpl,
    get_default_dtype,
)


_Array = Union[np.ndarray, jax.Array]


class VS_TYPE(eqx.Enumeration):
    """
    The enums to distinguish different data types of variational states.
    0: real parameters -> real outputs or (holomorphic complex -> complex)
    1: non-holomorphic complex parameters -> complex outputs
        ∇ψ(θ) = [∇(ψr)(θr) + 1j * ∇(ψi)(θr), ∇(ψr)(θi) + 1j * ∇(ψi)(θi)]
    2: real parameters -> complex outputs
        ∇ψ(θ) = ∇(ψr)(θ) + 1j * ∇(ψi)(θ)
    """

    real_or_holomorphic = 0
    non_holomorphic = 1
    real_to_complex = 2


class Variational(State):
    """
    Variational state.

    Args:
        models: Variational models, either an eqx.Module or a list of them.
            If a list of eqx.Module is given, the `input_fn` and `output_fn` should be
            specified to determine how to combine different variational models.

        param_file: File for loading parameters. Default to not loading parameters.

        symm: Symmetry of the network, default to no symmetry. If `input_fn` and
            `output_fn` are not given, this will generate symmetry projections as
            `input_fn` and `output_fn`.

        max_parallel: The maximum foward pass per device, default to no limit. This is
            important for large batches to avoid memory overflow. For Heisenberg
            hamiltonian, this also helps to improve the efficiency of computing local
            energy by keeping constant amount of forward pass and avoiding re-jit.

        input_fn: Function applied on the input spin before feeding into models.

        output_fn: Function applied on the models output to generate wavefunction.
    """

    def __init__(
        self,
        models: Union[eqx.Module, Sequence[eqx.Module]],
        param_file: Optional[Union[str, Path, BinaryIO]] = None,
        symm: Optional[Symmetry] = None,
        max_parallel: Optional[int] = None,
        input_fn: Optional[Callable] = None,
        output_fn: Optional[Callable] = None,
    ):
        super().__init__(symm)
        if isinstance(models, eqx.Module):
            models = (models,)
        else:
            models = tuple(models)
        self._models = models

        holomorphic = [a.holomorphic for a in models if hasattr(a, "holomorphic")]
        self._holomorphic = len(holomorphic) > 0 and all(holomorphic)

        # load params
        if param_file is not None:
            models = eqx.tree_deserialise_leaves(param_file, models)
        self._models = filter_replicate(models)
        self._max_parallel = max_parallel

        # initialize forward and backward
        self._init_forward(input_fn, output_fn)
        self._init_backward()
        self._maximum = jnp.abs(jnp.zeros((len(self.models),), get_default_dtype()))

        # for params flatten and unflatten
        params, others = self.partition()
        params, self._unravel_fn = jfu.ravel_pytree(params)
        self._nparams = params.size
        if params.dtype == jnp.float16:
            self._params_copy = params.astype(jnp.float32)
        else:
            self._params_copy = None

        batch = jnp.ones((1, self.nsites), jnp.int8)
        outputs = jax.eval_shape(self.forward_vmap, batch)
        is_p_cpl = is_params_cpl()
        is_outputs_cpl = np.issubdtype(outputs, np.complexfloating)
        if (not is_p_cpl) and (not is_outputs_cpl):
            self._vs_type = VS_TYPE.real_or_holomorphic
        elif is_p_cpl and is_outputs_cpl and self.holomorphic:
            self._vs_type = VS_TYPE.real_or_holomorphic
        elif is_p_cpl and is_outputs_cpl:
            self._vs_type = VS_TYPE.non_holomorphic
        elif (not is_p_cpl) and is_outputs_cpl:
            self._vs_type = VS_TYPE.real_to_complex
        else:
            raise RuntimeError("Parameter or output datatype not supported.")

    @property
    def models(self) -> eqx.Module:
        return self._models

    @property
    def holomorphic(self) -> bool:
        return self._holomorphic

    @property
    def max_parallel(self) -> int:
        return self._max_parallel

    @property
    def nparams(self) -> int:
        return self._nparams

    @property
    def vs_type(self) -> VS_TYPE:
        return self._vs_type

    def _init_forward(
        self, input_fn: Optional[Callable] = None, output_fn: Optional[Callable] = None
    ) -> None:
        if len(self.models) > 1 and (input_fn is None or output_fn is None):
            raise ValueError(
                "The default input_fn and output_fn only works for single models."
            )
        if input_fn is None:
            input_fn = lambda s: [self.symm.get_symm_spins(s)]
        if output_fn is None:
            output_fn = lambda x: self.symm.symmetrize(x[0])

        self.input_fn = input_fn
        self.output_fn = output_fn

        def net_forward(net: eqx.Module, x: jax.Array) -> jax.Array:
            batch = x.shape[:-1]
            x = x.reshape(-1, x.shape[-1])
            psi = jax.vmap(net)(x)
            return psi.reshape(batch)

        def forward_fn(
            models: Tuple[eqx.Module], spin: jax.Array, return_max: bool = False
        ) -> jax.Array:
            inputs = input_fn(spin)
            outputs = [net_forward(net, x) for net, x in zip(models, inputs)]
            psi = output_fn(outputs)
            if return_max:
                maximum = jnp.asarray([jnp.max(jnp.abs(out)) for out in outputs])
                return psi, maximum
            else:
                return psi

        self.forward_fn = eqx.filter_jit(forward_fn)
        forward_vmap = eqx.filter_jit(jax.vmap(forward_fn, in_axes=(None, 0, None)))
        self.forward_vmap = lambda spins, return_max=False: forward_vmap(
            self.models, spins, return_max
        )

    def _init_backward(self) -> None:
        """
        Generate functions for computing 1/ψ dψ/dθ. Designed for efficient combination
        of multiple networks.
        """

        def grad_fn(models: Tuple[eqx.Module], spin: jax.Array) -> jax.Array:
            def forward(net, x):
                out = net(x)
                if self.vs_type == VS_TYPE.real_or_holomorphic:
                    out = out.astype(get_default_dtype())
                elif jnp.iscomplexobj(out):
                    out = (out.real, out.imag)
                return out

            inputs = self.input_fn(spin)
            batch = [x.shape[:-1] for x in inputs]
            inputs = [x.reshape(-1, x.shape[-1]) for x in inputs]
            forward_vmap = jax.vmap(forward, in_axes=(None, 0))
            outputs = [forward_vmap(net, x) for net, x in zip(models, inputs)]

            def output_fn(outputs):
                psi = []
                for out, shape in zip(outputs, batch):
                    if isinstance(out, tuple):
                        out = out[0] + 1j * out[1]
                    psi.append(out.reshape(shape))
                psi = self.output_fn(psi)
                return psi / jax.lax.stop_gradient(psi)

            if self.vs_type == VS_TYPE.real_or_holomorphic:
                deltas = jax.grad(output_fn, holomorphic=self.holomorphic)(outputs)
            else:
                output_real = lambda outputs: output_fn(outputs).real
                output_imag = lambda outputs: output_fn(outputs).imag
                deltas_real = jax.grad(output_real)(outputs)
                deltas_imag = jax.grad(output_imag)(outputs)

            if self.vs_type == VS_TYPE.non_holomorphic:
                models_real, models_imag = tree_split_cpl(models)
                models = [(r, i) for r, i in zip(models_real, models_imag)]
                fn = lambda net, x: forward(tree_combine_cpl(net[0], net[1]), x)
            else:
                fn = forward

            @partial(jax.vmap, in_axes=(None, 0, 0))
            def backward(net, x, delta):
                f_vjp = filter_vjp(fn, net, x)[1]
                vjp_vals, _ = f_vjp(delta)
                return tree_fully_flatten(vjp_vals)

            grad = []
            if self.vs_type == VS_TYPE.real_or_holomorphic:
                for net, x, delta in zip(models, inputs, deltas):
                    grad.append(backward(net, x, delta))
            else:
                for net, x, dr, di in zip(models, inputs, deltas_real, deltas_imag):
                    gr = backward(net, x, dr)
                    gi = backward(net, x, di)
                    grad.append(gr + 1j * gi)

            if self.vs_type == VS_TYPE.non_holomorphic:
                grad_real = [g[:, : g.shape[1] // 2] for g in grad]
                grad_imag = [g[:, g.shape[1] // 2 :] for g in grad]
                grad = grad_real + grad_imag
            grad = jnp.concatenate(grad, axis=1).astype(get_default_dtype())
            return jnp.sum(grad, axis=0)

        self.grad_fn = eqx.filter_jit(grad_fn)
        self.grad = lambda s: grad_fn(self.models, s)
        grad_vmap = eqx.filter_jit(jax.vmap(grad_fn, in_axes=(None, 0)))
        self.jacobian = lambda spins: grad_vmap(self.models, spins)

    def __call__(
        self, fock_states: _Array, *, update_maximum: Optional[bool] = None
    ) -> jax.Array:
        ndevices = jax.local_device_count()
        nsamples, nsites = fock_states.shape
        is_input_sharded = is_sharded_array(fock_states)
        if update_maximum is None:
            update_maximum = not is_input_sharded

        if not is_input_sharded:
            fock_states = array_extend(fock_states, ndevices, axis=0, padding_values=1)
        if self._max_parallel is None or nsamples <= ndevices * self._max_parallel:
            fock_states = to_array_shard(fock_states)
            if update_maximum:
                psi, maximum = self.forward_vmap(fock_states, return_max=True)
            else:
                psi = self.forward_vmap(fock_states)
        else:
            fock_states = fock_states.reshape(ndevices, -1, nsites)
            ns_per_device = fock_states.shape[1]
            fock_states = array_extend(fock_states, self._max_parallel, 1, 1)

            nsplits = fock_states.shape[1] // self._max_parallel
            if isinstance(fock_states, jax.Array):
                fock_states = to_array_shard(fock_states)
                fock_states = jnp.split(fock_states, nsplits, axis=1)
            else:
                fock_states = np.split(fock_states, nsplits, axis=1)
            psi, maximum = [], []
            for s in fock_states:
                s = to_array_shard(s.reshape(-1, nsites))
                if update_maximum:
                    new_psi, new_max = self.forward_vmap(s, return_max=True)
                    maximum.append(new_max.reshape(ndevices, -1, len(self.models)))
                else:
                    new_psi = self.forward_vmap(s)
                psi.append(new_psi.reshape(ndevices, -1))

            psi = jnp.concatenate(psi, axis=1)
            psi = psi[:, :ns_per_device].flatten()
            if update_maximum:
                maximum = jnp.concatenate(maximum, axis=1)
                maximum = maximum[:, :ns_per_device, :].reshape(-1, len(self.models))

        if update_maximum:
            maximum = jnp.max(maximum, axis=0)
            self._maximum = jnp.where(maximum > self._maximum, maximum, self._maximum)
        return psi

    def partition(
        self, models: Optional[eqx.Module] = None
    ) -> Tuple[eqx.Module, eqx.Module]:
        if models is None:
            models = self._models
        is_nograd = lambda x: isinstance(x, NoGradLayer)
        return eqx.partition(models, eqx.is_inexact_array, is_leaf=is_nograd)

    def combine(self, params: eqx.Module, others: eqx.Module) -> eqx.Module:
        is_nograd = lambda x: isinstance(x, NoGradLayer)
        return eqx.combine(params, others, is_leaf=is_nograd)

    def get_params_flatten(self) -> jax.Array:
        params, others = self.partition()
        return tree_fully_flatten(params)

    def get_params_unflatten(self, params: jax.Array) -> PyTree:
        return filter_replicate(self._unravel_fn(params))

    def rescale(self) -> None:
        models = []
        for net, maximum in zip(self.models, self._maximum):
            finite_maximum = jnp.isfinite(maximum) & ~jnp.isclose(maximum, 0.)
            if hasattr(net, "rescale") and finite_maximum:
                models.append(net.rescale(maximum))
            else:
                models.append(net)
        self._models = tuple(models)

    def update(self, step: jax.Array, rescale: bool = True) -> None:
        if rescale:
            self.rescale()

        if not jnp.all(jnp.isfinite(step)):
            warn("Got invalid update step. The update is interrupted.")
            return
        if not is_params_cpl():
            step = step.real

        dtype = get_params_dtype()
        if dtype != jnp.float16:
            step = -step.astype(dtype)
            step = self.get_params_unflatten(step)
            self._models = eqx.apply_updates(self._models, step)
        else:
            self._params_copy -= step.astype(jnp.float32)
            new_params = self.get_params_unflatten(self._params_copy)
            params, others = self.partition()
            self._models = self.combine(new_params, others)

    def save(self, file: Union[str, Path, BinaryIO]) -> None:
        eqx.tree_serialise_leaves(file, self._models)

    def __mul__(self, other: Callable) -> Variational:
        """
        Construct a variational state ψ(s) = ψ₁(s) * ψ₂(s). ψ₂ will be taken as a pure
        function without parameters if it's a eqx.Module instead of a variational state.
        """
        if not isinstance(other, Variational):
            if isinstance(other, eqx.Module):
                other = other.__call__
            other = Variational(eqx.nn.Lambda(other))

        models = self.models + other.models  # tuple concatenate
        input_fn = lambda s: [*self.input_fn(s), *other.input_fn(s)]
        sep = len(self.models)
        output_fn = lambda x: self.output_fn(x[:sep]) * other.output_fn(x[sep:])
        if self.max_parallel is None:
            max_parallel = other.max_parallel
        elif other.max_parallel is None:
            max_parallel = self.max_parallel
        else:
            max_parallel = min(self.max_parallel, other.max_parallel)
        return Variational(models, None, self.symm, max_parallel, input_fn, output_fn)

    def __rmul__(self, other: Callable) -> Variational:
        return self * other

    def to_flax_model(self, package="netket", real_outputs: bool = False):
        """
        Convert the state to a flax model compatible with other packages.
        Training the model in other packages may be unstable.
        The supported packages are listed below
            netket (default), input 1/-1, output log(psi)
            jvmc, input 1/0, output log(psi)
        """
        params, others = self.partition()
        params, unravel_fn = jfu.ravel_pytree(params)

        class Model:
            def init(self, *args):
                return {"params": {"params": params}}

            @staticmethod
            def apply(params: dict, inputs: jax.Array, **kwargs) -> jax.Array:
                if package == "jvmc":
                    inputs = 2 * inputs - 1
                params = unravel_fn(params["params"]["params"])
                models = eqx.combine(params, others)
                forward_vmap = jax.vmap(self.forward_fn, in_axes=(None, 0, None))
                psi = forward_vmap(models, inputs, return_max=False)
                if real_outputs:
                    if jnp.iscomplexobj(psi):
                        raise RuntimeError(
                            "The outputs are specified to be real, but got complex"
                        )
                else:
                    psi += 0j
                return jnp.log(psi)

        return Model()
