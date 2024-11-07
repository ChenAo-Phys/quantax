from __future__ import annotations
from typing import Optional, Sequence, Tuple, Union, Callable, Any
from jaxtyping import PyTree
import jax
import jax.tree_util as jtu
import equinox as eqx


class Sequential(eqx.nn.Sequential):
    """
    A sequence of ``equinox.Module`` applied in order similar to
    `Sequential <https://docs.kidger.site/equinox/api/nn/sequential/>`_ in Equinox.

    .. note::

        Functions can be added as a layer by wrapping them in
        `equinox.nn.Lambda <https://docs.kidger.site/equinox/api/nn/sequential/#equinox.nn.Lambda>`.
    """

    layers: tuple
    holomorphic: bool = eqx.field(static=True)

    def __init__(self, layers: Sequence[Callable], holomorphic: bool = False):
        """
        :param layers:
            A sequence of ``equinox.Module``.

        :param holomorphic:
            Whether the whole network is a complex holomorphic function, default to ``False``.

        .. note::

            The users are responsible to ensure the given ``holomorphic`` argument is
            correct.
        """
        super().__init__(layers)
        self.holomorphic = holomorphic

    def __call__(self, x: jax.Array, *, s: jax.Array = None) -> jax.Array:
        """**Arguments:**

        - `x`: passed to the first member of the sequence.
        - `state`: If provided, then it is passed to, and updated from, any layer
            which subclasses [`equinox.nn.StatefulLayer`][].
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**
        The output of the last member of the sequence.

        If `state` is passed, then a 2-tuple of `(output, state)` is returned.
        If `state` is not passed, then just the output is returned.
        """
        if s is None:
            s = x
        for layer in self.layers:
            if isinstance(layer, RawInputLayer):
                x = layer(x, s)
            else:
                x = layer(x)
        return x

    def __getitem__(self, i: Union[int, slice]) -> Callable:
        if isinstance(i, int):
            return self.layers[i]
        elif isinstance(i, slice):
            return Sequential(self.layers[i])
        else:
            raise TypeError(f"Indexing with type {type(i)} is not supported")

    def rescale(self, maximum: jax.Array) -> Sequential:
        r"""
        Generate a new network in which all layers are rescaled.
        This is often used when a `~quantax.nn.Theta0Layer` is included as a layer.

        :param maximum:
            The maximum output obtained from this network.

        :return:
            The rescaled network

        .. note::

            This method generates a new network while doesn't modify the existing network.
        """
        layers = tuple(
            l.rescale(maximum) if hasattr(l, "rescale") else l for l in self.layers
        )
        return eqx.tree_at(lambda tree: tree.layers, self, layers)


class RawInputLayer(eqx.Module):
    """
    The layer that takes not only the output of the previous layer, but also the raw input
    fock state of the whole network.
    """

    def __call__(self, x: jax.Array, s: jax.Array) -> jax.Array:
        """
        The forward pass that takes two arguments, output of the previous layer and
        the raw input of the whole network.
        """


class RefModel(eqx.Module):
    """
    The model that allows accelerated forward pass through local updates and
    internal quantities.
    """

    def init_internal(self, x: jax.Array) -> PyTree:
        """
        Return initial internal values for the given configuration.
        """

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Usual forward pass without internal quantities.
        """

    def ref_forward_with_updates(
        self,
        x: jax.Array,
        x_old: jax.Array,
        nflips: int,
        internal: PyTree,
    ) -> Tuple[jax.Array, PyTree]:
        """
        Accelerated forward pass through local updates and internal quantities.
        This function is designed for sampling.

        :return:
            The evaluated wave function and the updated internal values.
        """

    def ref_forward(
        self,
        x: jax.Array,
        x_old: jax.Array,
        nflips: int,
        idx_segment: jax.Array,
        internal: PyTree,
    ) -> jax.Array:
        """
        Accelerated forward pass through local updates and internal quantities.
        This function is designed for local observables.
        """


class NoGradLayer(eqx.Module):
    """
    The layer in which the pytree leaves are not considered as differentiable parameters
    in Quantax computations.
    """


def filter_grad(
    fun: Callable, *, has_aux: bool = False, **gradkwargs
) -> Union[Callable, Tuple[Callable, Any]]:
    """
    Creates a function that computes the gradient of ``fun`` similar to
    `equinox.filter_grad <https://docs.kidger.site/equinox/api/transformations/#equinox.filter_grad>`_.

    The leaves in `~quantax.nn.NoGradLayer` are not considered as differentiable
    parameters.
    """
    grad_fn = eqx.filter_grad(fun, has_aux=has_aux, **gradkwargs)
    if has_aux:
        grad_fn, aux = grad_fn

    def filter_grad_fn(*args):
        grad = grad_fn(*args)

        is_nograd = lambda x: isinstance(x, NoGradLayer)
        set_none = lambda x: jtu.tree_map(lambda y: None, x) if is_nograd(x) else x
        grad = jtu.tree_map(set_none, grad, is_leaf=is_nograd)
        return grad

    if has_aux:
        return filter_grad_fn, aux
    else:
        return filter_grad_fn


def filter_vjp(
    fun: Callable, *primals, has_aux: bool = False, **vjpkwargs
) -> Union[Tuple[Any, Callable], Tuple[Any, Callable, Any]]:
    """
    Like
    `equinox.filter_vjp <https://docs.kidger.site/equinox/api/transformations/#equinox.filter_vjp>`_.

    The leaves in `~quantax.nn.NoGradLayer` are not considered as differentiable
    parameters.
    """
    outs = eqx.filter_vjp(fun, *primals, has_aux=has_aux, **vjpkwargs)
    if has_aux:
        out, vjp_fn, aux = outs
    else:
        out, vjp_fn = outs

    def filter_vjp_fn(*args):
        vjp = vjp_fn(*args)

        is_nograd = lambda x: isinstance(x, NoGradLayer)
        set_none = lambda x: jtu.tree_map(lambda y: None, x) if is_nograd(x) else x
        vjp = jtu.tree_map(set_none, vjp, is_leaf=is_nograd)
        return vjp

    if has_aux:
        return out, filter_vjp_fn, aux
    else:
        return out, filter_vjp_fn
