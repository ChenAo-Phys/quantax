from typing import Optional
from jaxtyping import Key
import jax
import jax.numpy as jnp
from .modules import NoGradLayer
from ..symmetry import Symmetry, TransND
from ..global_defs import get_lattice, get_params_dtype


class ReshapeConv(NoGradLayer):
    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        x = x.reshape(get_lattice().shape)
        x = jnp.moveaxis(x, -1, 0)
        x = x.astype(get_params_dtype())
        # x = jnp.concatenate([(x+1)/2, (-x+1)/2], axis=0)
        return x


class ConvSymmetrize(NoGradLayer):
    # stored as static will trigger jit error
    eigval: jax.Array

    def __init__(self, trans_symm: Optional[Symmetry] = None):
        super().__init__()
        if trans_symm is None:
            trans_symm = TransND()
        self.eigval = trans_symm.eigval

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        x = x.reshape(-1, self.eigval.size)  # check for unit cells with multiple atoms
        return jnp.mean(x * self.eigval[None])