from __future__ import annotations
from typing import Sequence, Tuple, Union, Callable, Any
import jax
import jax.tree_util as jtu
import equinox as eqx


class Sequential(eqx.nn.Sequential):
    """A sequence of [`equinox.Module`][]s applied in order.
    !!! note
        Activation functions can be added by wrapping them in [`equinox.nn.Lambda`][].
    """
    layers: tuple
    holomorphic: bool = eqx.field(static=True)

    def __init__(self, layers: Sequence[Callable], holomorphic: bool = False):
        super().__init__(layers)
        self.holomorphic = holomorphic

    def rescale(self, maximum: jax.Array) -> Sequential:
        layers = []
        for l in self.layers:
            layers.append(l.rescale(maximum) if hasattr(l, "rescale") else l)
        return Sequential(layers, self.holomorphic)


class NoGradLayer(eqx.Module):
    """
    For layers in which the leaves are not considered as differentiable parameters.
    """


def filter_grad(
    fun: Callable, *, has_aux: bool = False, **gradkwargs
) -> Union[Callable, Tuple[Callable, Any]]:
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
