from __future__ import annotations
from typing import Sequence, Tuple, Union, Callable
from jaxtyping import PyTree
import jax
import equinox as eqx


class Sequential(eqx.nn.Sequential):
    """
    A sequence of ``equinox.Module`` applied in order similar to
    `Sequential <https://docs.kidger.site/equinox/api/nn/sequential/>`_ in Equinox.

    .. note::

        Functions can be added as a layer by wrapping them in
        `equinox.nn.Lambda <https://docs.kidger.site/equinox/api/nn/sequential/#equinox.nn.Lambda>`.
    """

    layers: Tuple[Callable, ...]
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

    def ref_forward(
        self,
        s: jax.Array,
        s_old: jax.Array,
        nflips: int,
        internal: PyTree,
        return_update: bool = False,
    ) -> Union[jax.Array, Tuple[jax.Array, PyTree]]:
        """
        Accelerated forward pass through local updates and internal quantities.
        """
