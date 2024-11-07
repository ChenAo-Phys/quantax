from typing import Optional
from jaxtyping import Key
import jax
import jax.numpy as jnp
import equinox as eqx
from .modules import NoGradLayer, RawInputLayer
from ..symmetry import Symmetry, TransND, Identity
from ..global_defs import get_lattice


class ReshapeConv(NoGradLayer):
    """
    Reshape the input to the shape suitable for convolutional layers.

    A fock state in Quantax is usually givne by a 1D array with entries +1/-1.
    This layer reshape it to `~quantax.sites.Lattice.shape`.
    """

    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(self, dtype: jnp.dtype = jnp.float32):
        """
        :param dtype:
            Convert the input to the given data type, by default ``float32``.
        """
        super().__init__()
        self.dtype = dtype

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        lattice = get_lattice()
        shape = lattice.shape
        if lattice.is_fermion:
            shape = (shape[0] * 2,) + shape[1:]
        x = x.reshape(shape)
        x = x.astype(self.dtype)
        return x


class ConvSymmetrize(NoGradLayer, RawInputLayer):
    """
    Symmetrize the output of a convolutional network according to the given symmetry.
    """

    symm: Symmetry = eqx.field(static=True)

    def __init__(self, symm: Optional[Symmetry] = None):
        """
        :param symm:
            The symmetry used for symmetrization, by default
            `~quantax.symmetry.TransND` with sectors 0.
            If `~quantax.symmetry.Identity` is given, the layer won't symmetrize its
            output.
        """
        super().__init__()
        if symm is None:
            symm = TransND()
        self.symm = symm

    def __call__(self, x: jax.Array, s: jax.Array) -> jax.Array:
        if self.symm is Identity():
            return x

        x = x.reshape(-1, self.symm.nsymm).mean(axis=0)
        x = self.symm.symmetrize(x, s)
        return x
