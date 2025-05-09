from __future__ import annotations
from typing import Callable, Optional, Tuple, Union, BinaryIO
from jaxtyping import PyTree, ArrayLike
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
from ..nn import NoGradLayer, filter_vjp, RefModel
from ..utils import (
    chunk_shard_vmap,
    to_global_array,
    filter_replicate,
    array_extend,
    tree_fully_flatten,
    tree_split_cpl,
    tree_combine_cpl,
    apply_updates,
    get_replicate_sharding,
)
from ..global_defs import get_default_dtype, get_real_dtype, is_default_cpl


_Array = Union[np.ndarray, jax.Array]


class VS_TYPE(eqx.Enumeration):
    r"""
    The enums to distinguish different variational states according to their dtypes.

    0: real_or_holomorphic
        Real parameters -> real outputs

        or holomorphic complex -> complex

    1: non_holomorphic
        Non-holomorphic complex parameters -> complex outputs

        .. math::

            ∇_θ ψ = [∇_{θ_r} ψ_r + i ∇_{θ_r} ψ_i, ∇_{θ_i} ψ_r + i ∇_{θ_i} ψ_i]

    2: real_to_complex
        Real parameters -> complex outputs

        .. math::

            ∇_θ ψ = ∇_θ ψ_r + i ∇_θ ψ_i

    Check the table below for the size and data type of gradient matrices with
    :math:`N_s` sampels and :math:`N_p` parameters.

    .. list-table::
        :widths: 25 25 20 20
        :header-rows: 1

        *   - VS_TYPE
            - :math:`O` matrix as Jacobian
            - :math:`S` matrix in SR
            - :math:`T` matrix in MinSR
        *   - 0: real
            - :math:`N_s \times N_p` real
            - :math:`N_p \times N_p` real
            - :math:`N_s \times N_s` real
        *   - 0: holomorphic
            - :math:`N_s \times N_p` complex
            - :math:`N_p \times N_p` complex
            - :math:`N_s \times N_s` complex
        *   - 1: non_holomorphic
            - :math:`N_s \times 2N_p` complex
            - :math:`2N_p \times 2N_p` real
            - :math:`2N_s \times 2N_s` real
        *   - 2: real_to_complex
            - :math:`N_s \times N_p` complex
            - :math:`N_p \times N_p` real
            - :math:`2N_s \times 2N_s` real
    """

    real_or_holomorphic = 0
    non_holomorphic = 1
    real_to_complex = 2


class Variational(State):
    """
    Variational state.
    This is a wrapper of a jittable variational ansatz. The variational model should be
    given as an ``equinox.Module``.
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
        model: eqx.Module,
        param_file: Optional[Union[str, Path, BinaryIO]] = None,
        symm: Optional[Symmetry] = None,
        max_parallel: Union[None, int, Tuple[int, int]] = None,
        factor: Optional[Callable] = None,
    ):
        r"""
        :param model:
            Variational model. Should be an ``equinox.Module``.

        :param param_file:
            File for loading parameters which is saved by `~quantax.state.Variational.save`
            or `equinox.tree_serialise_leaves`, default to not loading parameters.

        :param symm: Symmetry of the network, default to `~quantax.symmetry.Identity`.

        :param max_parallel:
            The maximum foward pass allowed per device, default to no limit.
            Specifying a limited value is important for large batches to avoid memory overflow.
            For Heisenberg-like hamiltonian, this also helps to improve the efficiency
            of computing local energy by keeping constant amount of forward pass and
            avoiding re-jitting.

        :param factor:
            The additional factor multiplied on the network outputs. The parameters in
            this factor won't be updated together with the variational state. This is
            useful for expressing some fixed sign structures.

        Denoting the network output and factor output as :math:`f(s)` and :math:`g(s)`,
        and symmetry elements as :math:`T_i` with characters :math:`\omega_i`,
        the final wave function is given by
        :math:`\psi(s) = \sum_i \omega_i \, f(T_i s) g(T_i s) / n_{symm}`
        """
        super().__init__(symm)
        if param_file is not None:
            model = eqx.tree_deserialise_leaves(param_file, model)
        self._model = filter_replicate(model)
        if hasattr(model, "holomorphic"):
            self._holomorphic = model.holomorphic
        else:
            self._holomorphic = False

        # initialize forward and backward
        self._maximum = jnp.array(
            0.0, dtype=get_real_dtype(), device=get_replicate_sharding()
        )

        if max_parallel is None or isinstance(max_parallel, int):
            self._forward_chunk = max_parallel
            self._backward_chunk = max_parallel
            self._ref_chunk = max_parallel
        elif len(max_parallel) == 2:
            self._forward_chunk, self._backward_chunk = max_parallel
            self._ref_chunk = self._forward_chunk
        else:
            self._forward_chunk, self._backward_chunk, self._ref_chunk = max_parallel
        self._init_forward()
        self._init_backward()

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

        if factor is None:
            self._factor = lambda x: 1
        else:
            self._factor = factor

    @property
    def model(self) -> eqx.Module:
        """The variational model used in the variational state."""
        return self._model

    @property
    def holomorphic(self) -> bool:
        """Whether the variational state is holomorphic."""
        return self._holomorphic

    @property
    def forward_chunk(self) -> int:
        """The maximum foward pass allowed per device."""
        return self._forward_chunk

    @property
    def backward_chunk(self) -> int:
        """The maximum backward pass allowed per device."""
        return self._backward_chunk

    @property
    def ref_chunk(self) -> int:
        """The maximum reference forward with updates allowed per device."""
        return self._ref_chunk

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

    def _init_forward(self) -> None:
        def direct_forward(model: eqx.Module, s: jax.Array) -> jax.Array:
            s_symm = self.symm.get_symm_spins(s)
            psi = jax.vmap(model)(s_symm) * jax.vmap(self._factor)(s_symm)
            psi = self.symm.symmetrize(psi, s)
            return psi.astype(get_default_dtype())

        self._direct_forward = chunk_shard_vmap(
            direct_forward, in_axes=(None, 0), out_axes=0, chunk_size=self.forward_chunk
        )

        def init_internal(model, s):
            s_symm = self.symm.get_symm_spins(s)
            return jax.vmap(model.init_internal)(s_symm)

        self._init_internal = chunk_shard_vmap(
            init_internal, in_axes=(None, 0), out_axes=0, chunk_size=self.ref_chunk
        )

        def ref_forward_with_updates(model, s, s_old, nflips, internal):
            s_symm = self.symm.get_symm_spins(s)
            s_old_symm = self.symm.get_symm_spins(s_old)
            forward = eqx.filter_vmap(
                model.ref_forward_with_updates, in_axes=(0, 0, None, 0)
            )
            psi, internal = forward(s_symm, s_old_symm, nflips, internal)
            psi *= jax.vmap(self._factor)(s_symm)
            psi = self.symm.symmetrize(psi, s)
            return psi.astype(get_default_dtype()), internal

        self._ref_forward_with_updates = chunk_shard_vmap(
            ref_forward_with_updates,
            in_axes=(None, 0, 0, None, 0),
            out_axes=(0, 0),
            chunk_size=self.ref_chunk,
        )

        def ref_forward(model, s, s_old, nflips, idx_segment, internal):
            s_symm = self.symm.get_symm_spins(s)
            s_old_symm = jax.vmap(self.symm.get_symm_spins)(s_old)
            forward = eqx.filter_vmap(model.ref_forward, in_axes=(0, 1, None, None, 1))
            psi = forward(s_symm, s_old_symm, nflips, idx_segment, internal)
            psi *= jax.vmap(self._factor)(s_symm)
            psi = self.symm.symmetrize(psi, s)
            return psi.astype(get_default_dtype())

        self._ref_forward = chunk_shard_vmap(
            ref_forward,
            in_axes=(None, 0, None, None, 0, None),
            shard_axes=(None, 0, 0, None, 0, 0),
            out_axes=0,
            chunk_size=self.forward_chunk,
        )

    def __call__(self, s: _Array) -> jax.Array:
        r"""
        Compute :math:`\psi(s)` for input states s.

        :param s:
            Input states s with entries :math:`\pm 1`.

        .. warning::

            This function is not jittable.

        .. note::

            The returned value is :math:`\psi(s)` instead of :math:`\log\psi(s)`.
        """
        nsamples = s.shape[0]
        ndevices = jax.device_count()
        s = to_global_array(array_extend(s, ndevices))

        psi = self._direct_forward(self.model, s)

        psi = psi[:nsamples]
        return psi

    def init_internal(self, x: jax.Array) -> PyTree:
        if isinstance(self.model, RefModel):
            return self._init_internal(self.model, x)
        else:
            return None

    def ref_forward_with_updates(
        self, s: _Array, s_old: jax.Array, nflips: int, internal: PyTree
    ) -> Tuple[jax.Array, PyTree]:
        if not isinstance(self.model, RefModel):
            return self(s), None

        return self._ref_forward_with_updates(self.model, s, s_old, nflips, internal)

    def ref_forward(
        self,
        s: _Array,
        s_old: jax.Array,
        nflips: int,
        idx_segment: jax.Array,
        internal: PyTree,
    ) -> jax.Array:
        if not isinstance(self.model, RefModel):
            return self(s)

        return self._ref_forward(self.model, s, s_old, nflips, idx_segment, internal)

    def _init_backward(self) -> None:
        """
        Generate functions for computing 1/ψ dψ/dθ.
        """

        def grad_fn(model: eqx.Module, s: jax.Array) -> jax.Array:
            def forward(model, x):
                psi = model(x) * self._factor(x)
                if self.vs_type == VS_TYPE.real_or_holomorphic:
                    psi = psi.astype(get_default_dtype())
                elif jnp.iscomplexobj(psi):
                    psi = (psi.real, psi.imag)
                return psi

            s_symm = self.symm.get_symm_spins(s)
            forward_vmap = jax.vmap(forward, in_axes=(None, 0))
            psi = forward_vmap(model, s_symm)

            def output_fn(psi):
                if isinstance(psi, tuple):
                    psi = psi[0] + 1j * psi[1]
                psi = self.symm.symmetrize(psi, s)
                return psi / jax.lax.stop_gradient(psi)

            if self.vs_type == VS_TYPE.real_or_holomorphic:
                delta = jax.grad(output_fn, holomorphic=self.holomorphic)(psi)
            else:
                output_real = lambda outputs: output_fn(outputs).real
                output_imag = lambda outputs: output_fn(outputs).imag
                delta_real = jax.grad(output_real)(psi)
                delta_imag = jax.grad(output_imag)(psi)

            if self.vs_type == VS_TYPE.non_holomorphic:
                model = tree_split_cpl(model)
                fn = lambda net, x: forward(tree_combine_cpl(net[0], net[1]), x)
            else:
                fn = forward

            @partial(jax.vmap, in_axes=(None, 0, 0))
            def backward(net, s, delta):
                f_vjp = filter_vjp(fn, net, s)[1]
                vjp_vals, _ = f_vjp(delta)
                return tree_fully_flatten(vjp_vals)

            if self.vs_type == VS_TYPE.real_or_holomorphic:
                grad = backward(model, s_symm, delta)
            else:
                grad_real_out = backward(model, s_symm, delta_real)
                grad_imag_out = backward(model, s_symm, delta_imag)
                grad = jax.lax.complex(grad_real_out, grad_imag_out)

            if self.vs_type == VS_TYPE.non_holomorphic:
                grad_real_param = grad[:, : grad.shape[1] // 2]
                grad_imag_param = grad[:, grad.shape[1] // 2 :]
                grad = jnp.concatenate([grad_real_param, grad_imag_param], axis=1)
            return jnp.sum(grad.astype(get_default_dtype()), axis=0)

        self._grad_vmap = chunk_shard_vmap(
            grad_fn, in_axes=(None, 0), out_axes=0, chunk_size=self.backward_chunk
        )

    def jacobian(self, fock_states: jax.Array) -> jax.Array:
        r"""
        Compute the jacobian matrix :math:`\frac{1}{\psi} \frac{\partial \psi}{\partial \theta}`.
        See `~quantax.state.VS_TYPE` for the definition of jacobian for different kinds
        of networks.

        :param fock_states:
            The input fock states.

        :return:
            A 2D jacobian matrix with the first dimension for different inputs and
            the second dimension for different parameters. The order of parameters are
            the same as `~quantax.state.Variational.get_params_flatten`.
        """
        return self._grad_vmap(self.model, fock_states)

    def partition(
        self, model: Optional[eqx.Module] = None
    ) -> Tuple[eqx.Module, eqx.Module]:
        """
        Split the variational model into two pytrees, one containing all parameters
        and the other containing all other elements, similar to
        `partition <https://docs.kidger.site/equinox/api/manipulation/#equinox.partition>`_
        in Equinox.

        :param model:
            The model to be splitted, default to be the variational model in the
            variational state.
        """
        if model is None:
            model = self._model
        is_nograd = lambda x: isinstance(x, NoGradLayer)
        return eqx.partition(model, eqx.is_inexact_array, is_leaf=is_nograd)

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
        Obtain the parameters pytree from a flattened 1D array of all parameters.
        """
        return filter_replicate(self._unravel_fn(params))

    def rescale(self, factor: Optional[ArrayLike] = None) -> None:
        """
        Rescale the variational state according to the maximum wave function stored
        during the forward pass.

        .. note::

            This only works if there is a ``rescale`` function in the given model,
            which exists in most models provided in `quantax.model`.

            Overflow is very likely to happen in the training of variational states
            if there is no ``rescale`` function in the model.
        """
        if factor is None:
            factor = self._maximum
        if jnp.isfinite(factor) & ~jnp.isclose(factor, 0.0):
            if hasattr(self.model, "rescale"):
                self._model = self._model.rescale(factor)
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
        if not jnp.all(jnp.isfinite(step)):
            warn("Got invalid update step. The update is interrupted.")
            return
        if not np.issubdtype(self.dtype, np.complexfloating):
            step = step.real

        if self.dtype != jnp.float16:
            step = -step.astype(self.dtype)
            step = self.get_params_unflatten(step)
            self._model = apply_updates(self._model, step)
        else:
            self._params_copy -= step.astype(jnp.float32)
            new_params = self.get_params_unflatten(self._params_copy)
            params, others = self.partition()
            self._model = self.combine(new_params, others)

        if rescale:
            self.rescale()

    def save(self, file: Union[str, Path, BinaryIO]) -> None:
        """
        Save the variational model in the given file. This file can be used be loaded
        when initializing `~quantax.state.Variational`.
        """
        if jax.process_index() == 0:
            eqx.tree_serialise_leaves(file, self._model)

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
                model = eqx.combine(params, others)
                psi = self._direct_forward(model, inputs)
                if make_complex:
                    psi += 0j
                return jnp.log(psi)

        return Model()
