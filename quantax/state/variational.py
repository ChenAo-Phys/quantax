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
    to_global_array,
    filter_replicate,
    array_extend,
    tree_fully_flatten,
    tree_split_cpl,
    tree_combine_cpl,
)
from ..global_defs import get_default_dtype, is_default_cpl


_Array = Union[np.ndarray, jax.Array]


class VS_TYPE(eqx.Enumeration):
    r"""
    The enums to distinguish different variational states according to their dtypes.

    0: real_or_holomorphic
        Real parameters -> real outputs or (holomorphic complex -> complex)

    1: non_holomorphic
        Non-holomorphic complex parameters -> complex outputs

        .. math::

            ∇_θ ψ = [∇_{θ_r} ψ_r + i ∇_{θ_r} ψ_i, ∇_{θ_i} ψ_r + i ∇_{θ_i} ψ_i]

    2: real_to_complex
        Real parameters -> complex outputs

        .. math::

            ∇_θ ψ = ∇_θ ψ_r + i ∇_θ ψ_i
    """

    real_or_holomorphic = 0
    non_holomorphic = 1
    real_to_complex = 2


class Variational(State):
    """
    Variational state.
    This is a wrapper of a jittable variational ansatz written as an ``equinox.Module``.
    For details of Equinox, see this `documentation <https://docs.kidger.site/equinox/all-of-equinox/>`_.

    .. warning::
        There are many intermediate values stored in the class, so most functions like
        ``__call__`` in this class are non-jittable.

    .. warning::
        Many quantities are only computed once in the initialization.
        Please don't change the private attributes unless an update function is provided.
        One can define a new state if some changes are necessary.
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
        """
        :param models:
            Variational models, either an ``eqx.Module`` or a list of them.
            If a list of eqx.Module is given, the ``input_fn`` and ``output_fn`` should be
            specified to indicate how to combine different variational models.

        :param param_file:
            File for loading parameters which is saved by `~quantax.state.Variational.save`,
            default to not loading parameters.

        :param symm: Symmetry of the network, default to `quantax.symmetry.Identity`.
            If ``input_fn`` and ``output_fn`` are not given, this will generate
            symmetry projections as ``input_fn`` and ``output_fn``.

        :param max_parallel:
            The maximum foward pass allowed per device, default to no limit.
            Specifying a limited value is important for large batches to avoid memory overflow.
            For Heisenberg-like hamiltonian, this also helps to improve the efficiency
            of computing local energy by keeping constant amount of forward pass and
            avoiding re-jitting.

        :param input_fn:
            Function applied on the input spin before feeding into models.

        :param output_fn:
            Function applied on the models' output to generate wavefunction.
        """
        super().__init__(symm)
        if isinstance(models, eqx.Module):
            models = (models,)
        else:
            models = tuple(models)
        if param_file is not None:
            models = eqx.tree_deserialise_leaves(param_file, models)
        self._models = filter_replicate(models)

        holomorphic = [a.holomorphic for a in models if hasattr(a, "holomorphic")]
        self._holomorphic = len(holomorphic) > 0 and all(holomorphic)

        # initialize forward and backward
        self._init_forward(input_fn, output_fn)
        self._init_backward()
        self._maximum = jnp.abs(jnp.zeros((len(self.models),), get_default_dtype()))
        self._max_parallel = max_parallel

        # for params flatten and unflatten
        params, others = self.partition()
        params, self._unravel_fn = jfu.ravel_pytree(params)
        self._nparams = params.size
        self._dtype = params.dtype
        if params.dtype == jnp.float16:
            self._params_copy = params.astype(jnp.float32)
        else:
            self._params_copy = None

        is_params_cpl = np.issubdtype(params.dtype, np.complexfloating)
        is_outputs_cpl = is_default_cpl()
        if (not is_params_cpl) and (not is_outputs_cpl):
            self._vs_type = VS_TYPE.real_or_holomorphic
        elif is_params_cpl and is_outputs_cpl and self.holomorphic:
            self._vs_type = VS_TYPE.real_or_holomorphic
        elif is_params_cpl and is_outputs_cpl:
            self._vs_type = VS_TYPE.non_holomorphic
        elif (not is_params_cpl) and is_outputs_cpl:
            self._vs_type = VS_TYPE.real_to_complex
        else:
            raise RuntimeError(
                f"The parameter dtype is {params.dtype}, while the computation dtype in"
                f"quantax is {get_default_dtype()}. This combination is not supported."
            )

    @property
    def models(self) -> Tuple[eqx.Module, ...]:
        """A tuple containing the variational models used in the variational state."""
        return self._models

    @property
    def holomorphic(self) -> bool:
        """Whether the variational state is holomorphic."""
        return self._holomorphic

    @property
    def max_parallel(self) -> int:
        """The maximum foward pass allowed per device."""
        return self._max_parallel

    @property
    def nparams(self) -> int:
        """Number of total parameters in the variational state."""
        return self._nparams

    @property
    def dtype(self) -> np.dtype:
        """The parameter data type of the variational state."""
        return self._dtype

    @property
    def vs_type(self) -> VS_TYPE:
        """The type of variational state."""
        return self._vs_type

    def _init_forward(
        self, input_fn: Optional[Callable] = None, output_fn: Optional[Callable] = None
    ) -> None:
        if len(self.models) > 1 and (input_fn is None or output_fn is None):
            raise ValueError(
                "The default input_fn and output_fn only works for a single model."
            )
        if input_fn is None:
            input_fn = lambda s: [self.symm.get_symm_spins(s)]
        if output_fn is None:
            output_fn = lambda x, s: self.symm.symmetrize(x[0], s)
        self.input_fn = input_fn
        self.output_fn = output_fn

        def forward_fn(
            models: Tuple[eqx.Module], spin: jax.Array, return_max: bool = False
        ) -> jax.Array:
            inputs = input_fn(spin)
            inputs = [x.reshape(-1, x.shape[-1]) for x in inputs]
            outputs = [jax.vmap(net)(x) for net, x in zip(models, inputs)]
            psi = output_fn(outputs, spin)
            if psi.size == 1:
                psi = psi.flatten()[0]
            else:
                raise RuntimeError("Network output is not a scalar.")
            psi = psi.astype(get_default_dtype())
            if return_max:
                maximum = jnp.asarray([jnp.max(jnp.abs(out)) for out in outputs])
                return psi, maximum
            else:
                return psi

        self._forward_fn = eqx.filter_jit(forward_fn)
        self._forward_vmap = eqx.filter_jit(
            jax.vmap(forward_fn, in_axes=(None, 0, None))
        )

    def forward_fn(
        self, models: Tuple[eqx.Module], spin: jax.Array, return_max: bool = False
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        return self._forward_fn(models, spin, return_max)

    def forward_vmap(
        self, models: Tuple[eqx.Module], spins: jax.Array, return_max: bool = False
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        return self._forward_vmap(models, spins, return_max)

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
            inputs = [x.reshape(-1, x.shape[-1]) for x in inputs]
            forward_vmap = jax.vmap(forward, in_axes=(None, 0))
            outputs = [forward_vmap(net, x) for net, x in zip(models, inputs)]

            def output_fn(outputs):
                psi = []
                for out in outputs:
                    if isinstance(out, tuple):
                        out = out[0] + 1j * out[1]
                    psi.append(out)
                psi = self.output_fn(psi, spin)
                if psi.size == 1:
                    psi = psi.flatten()[0]
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
            grad = [jnp.sum(g.astype(get_default_dtype()), axis=0) for g in grad]
            return jnp.concatenate(grad)

        self._grad_fn = eqx.filter_jit(grad_fn)
        self._grad_vmap = eqx.filter_jit(jax.vmap(grad_fn, in_axes=(None, 0)))

    def grad_fn(self, models: Tuple[eqx.Module], spin: jax.Array) -> jax.Array:
        return self._grad_fn(models, spin)

    def grad_vmap(self, models: Tuple[eqx.Module], spins: jax.Array) -> jax.Array:
        return self._grad_vmap(models, spins)

    def jacobian(self, spins: jax.Array) -> jax.Array:
        """
        Compute the jacobian matrix.

        :param spins:
            The input fock states.

        :return:
            A 2D jacobian matrix with the first dimension for different inputs and
            the second dimension for different parameters. The order of parameters are
            the same as `~quantax.state.Variational.get_params_flatten`.
        """
        return self._grad_vmap(self.models, spins)

    def __call__(
        self, fock_states: _Array, *, update_maximum: bool = False
    ) -> jax.Array:
        r"""
        Compute :math:`\psi(s)` for input states s.

        :param fock_states:
            Input states s with entries :math:`\pm 1`.

        :param update_maximum:
            Whether the maximum wave function stored in the variational state should
            be updated. By default, it's only updated when the input ``fock_states``
            is not distributed on different machines, so that update maximum doesn't
            introduce much additional costs in data communication.

        .. warning::

            This function is not jittable.

        .. note::

            The returned value is :math:`\psi(s)` instead of :math:`\log\psi(s)`.

        .. note::

            The updated maximum wave function can be used later in
            `~quantax.state.Variational.update` and `~quantax.state.Variational.rescale`
            to avoid data overflow.
        """
        models = self.models
        nsamples, nsites = fock_states.shape

        if isinstance(fock_states, jax.Array):
            ndevices = len(fock_states.devices())
        else:
            ndevices = jax.device_count()
            fock_states = array_extend(fock_states, ndevices, axis=0, padding_values=1)
            fock_states = to_global_array(fock_states)

        if self._max_parallel is None or nsamples <= ndevices * self._max_parallel:
            if update_maximum:
                psi, maximum = self.forward_vmap(models, fock_states, return_max=True)
            else:
                psi = self.forward_vmap(models, fock_states)
        else:
            fock_states = fock_states.reshape(ndevices, -1, nsites)
            ns_per_device = fock_states.shape[1]
            fock_states = array_extend(fock_states, self._max_parallel, 1, 1)

            nsplits = fock_states.shape[1] // self._max_parallel
            fock_states = jnp.split(fock_states, nsplits, axis=1)
            psi, maximum = [], []
            for s in fock_states:
                s = s.reshape(-1, nsites)
                if update_maximum:
                    new_psi, new_max = self.forward_vmap(models, s, return_max=True)
                    maximum.append(new_max.reshape(ndevices, -1, len(self.models)))
                else:
                    new_psi = self.forward_vmap(models, s)
                psi.append(new_psi.reshape(ndevices, -1))

            psi = jnp.concatenate(psi, axis=1)[:, :ns_per_device]
            if update_maximum:
                maximum = jnp.concatenate(maximum, axis=1)[:, :ns_per_device, :]

        psi = psi.flatten()[:nsamples]
        if update_maximum and maximum.size > 0:
            maximum = maximum.reshape(-1, len(self.models))[:nsamples, :]
            maximum = jnp.max(maximum, axis=0)
            self._maximum = jnp.where(maximum > self._maximum, maximum, self._maximum)
        return psi

    def partition(
        self, models: Optional[eqx.Module] = None
    ) -> Tuple[eqx.Module, eqx.Module]:
        """
        Split the variational models into two pytrees, one containing all parameters
        and the other containing all other elements, similar to
        `partition <https://docs.kidger.site/equinox/api/manipulation/#equinox.partition>`_
        in Equinox.

        :param models:
            The models to be splitted, default to be the variational models in the
            variational state.
        """
        if models is None:
            models = self._models
        is_nograd = lambda x: isinstance(x, NoGradLayer)
        return eqx.partition(models, eqx.is_inexact_array, is_leaf=is_nograd)

    def combine(self, params: eqx.Module, others: eqx.Module) -> eqx.Module:
        """
        Combine two pytrees, one containing all parameters and the other containing all
        other elements, into one variational model. This is similar to
        `combine <https://docs.kidger.site/equinox/api/manipulation/#equinox.combine>`_
        in Equinox.

        :param params:
            The pytree containing only parameters.

        :param others:
            The pytree containing other elements.
        """
        is_nograd = lambda x: isinstance(x, NoGradLayer)
        return eqx.combine(params, others, is_leaf=is_nograd)

    def get_params_flatten(self) -> jax.Array:
        """
        Obtain a flattened 1D array of all parameters.
        """
        params, others = self.partition()
        return tree_fully_flatten(params)

    def get_params_unflatten(self, params: jax.Array) -> PyTree:
        """
        From a flattened 1D array of all parameters, obtain the parameters pytree.
        """
        return filter_replicate(self._unravel_fn(params))

    def rescale(self) -> None:
        """
        Rescale the variational state according to the maximum wave function stored
        during the forward pass.

        .. note::

            This only works if there is a ``rescale`` function in the given model,
            which exists in most models provided in `quantax.model`.

            Overflow is very likely to happen in the training of variational states
            if there is no ``rescale`` function in the model.
        """
        models = []
        for net, maximum in zip(self.models, self._maximum):
            is_maximum_finite = jnp.isfinite(maximum) & ~jnp.isclose(maximum, 0.0)
            if is_maximum_finite and hasattr(net, "rescale"):
                models.append(net.rescale(maximum))
            else:
                models.append(net)
        self._models = tuple(models)
        self._maximum = jnp.zeros_like(self._maximum)

    def update(self, step: jax.Array, rescale: bool = True) -> None:
        r"""
        Update the variational parameters of the state as
        :math:`\theta' = \theta - \delta\theta`.

        :param step:
            The update step :math:`\delta\theta`.

        :param rescale:
            Whether the `~quantax.state.Variational.rescale` function should be called
            to rescale the variational state, default to ``True``.

        .. note::

            The update direction is :math:`-\delta\theta` instead of :math:`\delta\theta`.
        """
        if rescale:
            self.rescale()

        if not jnp.all(jnp.isfinite(step)):
            warn("Got invalid update step. The update is interrupted.")
            return
        if not np.issubdtype(self.dtype, np.complexfloating):
            step = step.real

        if self.dtype != jnp.float16:
            step = -step.astype(self.dtype)
            step = self.get_params_unflatten(step)
            self._models = eqx.apply_updates(self._models, step)
        else:
            self._params_copy -= step.astype(jnp.float32)
            new_params = self.get_params_unflatten(self._params_copy)
            params, others = self.partition()
            self._models = self.combine(new_params, others)

    def save(self, file: Union[str, Path, BinaryIO]) -> None:
        """
        Save the variational model in the given file. This file can be used be loaded
        when initializing `~quantax.state.Variational`.
        """
        eqx.tree_serialise_leaves(file, self._models)

    # def __mul__(self, other: Callable) -> Variational:
    #     r"""
    #     Construct a variational state :math:`\psi(s) = \psi_1(s) \times \psi_2(s)` by
    #     ``state = state1 * state2``.

    #     .. note::

    #         The input state is taken as a pure function without parameters if it's an
    #         ``eqx.Module`` instead of a `~quantax.state.Variational`.
    #     """
    #     if not isinstance(other, Variational):
    #         if isinstance(other, eqx.Module):
    #             other = other.__call__
    #         other = Variational(eqx.nn.Lambda(other))

    #     models = self.models + other.models  # tuple concatenate
    #     input_fn = lambda s: [*self.input_fn(s), *other.input_fn(s)]
    #     sep = len(self.models)
    #     output_fn = lambda x, s: (
    #         self.output_fn(x[:sep], s) * other.output_fn(x[sep:], s)
    #     )
    #     if self.max_parallel is None:
    #         max_parallel = other.max_parallel
    #     elif other.max_parallel is None:
    #         max_parallel = self.max_parallel
    #     else:
    #         max_parallel = min(self.max_parallel, other.max_parallel)
    #     return Variational(models, None, self.symm, max_parallel, input_fn, output_fn)

    # def __rmul__(self, other: Callable) -> Variational:
    #     return self * other

    def to_flax_model(self, package="netket", make_complex: bool = False):
        """
        Convert the state to a flax model compatible with other packages.
        Training the generated state in other packages is probably unstable,
        but the state can be used to measure observables.

        :param package:
            Convert the current state to the format of the given package.
            The supported packages are

            netket (default)
                input 1/-1, output :math:`\log\psi`

            jvmc
                input 1/0, output :math:`\log\psi`

        :param make_complex:
            Whether the network output should be made complex explicitly.
            This is necessary when :math:`\psi` is real but contains negative values.
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
                psi = self.forward_vmap(models, inputs, return_max=False)
                if make_complex:
                    psi += 0j
                return jnp.log(psi)

        return Model()
