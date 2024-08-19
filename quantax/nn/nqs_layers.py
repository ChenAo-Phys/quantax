from typing import Optional
from jaxtyping import Key
import jax
import jax.numpy as jnp
import equinox as eqx
from .modules import NoGradLayer
from ..symmetry import Symmetry, TransND
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
        x = x.reshape(get_lattice().shape)
        x = x.astype(self.dtype)
        return x


class ConvSymmetrize(NoGradLayer):
    """
    Symmetrize the output of a convolutional network according to the given
    translational symmetry.
    """
    eigval: jax.Array

    def __init__(self, trans_symm: Optional[Symmetry] = None):
        """
        :param trans_symm:
            The translational symmetry used for symmetrization, by default
            `~quantax.symmetry.TransND` with sectors 0.
        """
        super().__init__()
        if trans_symm is None:
            trans_symm = TransND()
        self.eigval = trans_symm.eigval

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        x = x.reshape(-1, self.eigval.size)  # check for unit cells with multiple atoms
        return jnp.mean(x * self.eigval[None])
