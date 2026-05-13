from __future__ import annotations
from typing import Sequence, Callable, Any, overload, Literal
from jaxtyping import PyTree
import jax
import equinox as eqx
from ..utils import PsiArray


class Sequential(eqx.Module):
    """
    A sequence of ``equinox.Module`` applied in order similar to
    `Sequential <https://docs.kidger.site/equinox/api/nn/sequential/>`_ in Equinox.
    """

    layers: tuple[Callable, ...]
    holomorphic: bool

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
        self.layers = tuple(layers)
        self.holomorphic = holomorphic

    def __call__(self, x: Any, *, s: jax.Array | None = None) -> PsiArray:
        """
        ...
        """
        if s is None:
            s = x
        for layer in self.layers:
            if isinstance(layer, RawInputLayer):
                if s is None:
                    raise ValueError("`RawInputLayer` requires an additional input s.")
                x = layer(x, s)
            else:
                x = layer(x)
        return x

    def __getitem__(self, i: int | slice) -> Callable:
        if isinstance(i, int):
            return self.layers[i]
        elif isinstance(i, slice):
            return Sequential(self.layers[i], holomorphic=self.holomorphic)
        else:
            raise TypeError(f"Indexing with type {type(i)} is not supported")

    def __iter__(self):
        yield from self.layers

    def __len__(self) -> int:
        return len(self.layers)


class RawInputLayer(eqx.Module):
    """
    The layer that takes not only the output of the previous layer, but also the raw input
    basis state.
    """

    def __call__(self, x: Any, s: jax.Array) -> Any:
        """
        The forward pass.

        :param x:
            The output of the previous layer.

        :param s:
            The raw input basis state.
        """
        raise NotImplementedError


class RefModel(eqx.Module):
    """
    The model that allows accelerated forward pass through local updates and
    internal quantities.
    """

    @property
    def use_ref(self) -> bool:
        """
        Whether to use reference implementation for local updates. Default to True.
        """
        return True

    def init_internal(self, s: jax.Array) -> tuple[PsiArray, PyTree]:
        """
        Return initial wavefunction and internal values for the given configuration.

        :returns:
            A tuple of (initial wavefunction, internal quantities).
        """
        raise NotImplementedError

    def __call__(self, s: jax.Array) -> PsiArray:
        """
        Usual forward pass without internal quantities.
        """
        raise NotImplementedError

    @property
    def required_update_modes(self) -> tuple[str, ...]:
        """
        The required update modes for accelerated ref_forward pass.
        """
        return ()

    @overload
    def ref_forward(
        self,
        s: jax.Array,
        s_old: jax.Array,
        update_mode: dict[str, Any],
        internal: PyTree,
        return_update: Literal[False] = False,
    ) -> PsiArray: ...

    @overload
    def ref_forward(
        self,
        s: jax.Array,
        s_old: jax.Array,
        update_mode: dict[str, Any],
        internal: PyTree,
        return_update: Literal[True],
    ) -> tuple[PsiArray, PyTree]: ...

    def ref_forward(
        self,
        s: jax.Array,
        s_old: jax.Array,
        update_mode: dict[str, Any],
        internal: PyTree,
        return_update: bool = False,
    ) -> PsiArray | tuple[PsiArray, PyTree]:
        """
        Accelerated forward pass through local updates and internal quantities.

        :param s:
            The new configuration.

        :param s_old:
            The old configuration.

        :param update_mode:
            A dictionary specifying the update mode.
            For instance, ``{"nflips": 2}`` indicates that there are 2 local updates.

        :param internal:
            The internal quantities.

        :param return_update:
            Whether to return the updated internal quantities.
        """
        raise NotImplementedError
