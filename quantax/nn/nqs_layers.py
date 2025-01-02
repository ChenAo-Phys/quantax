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

class SquareGconv(eqx.Module):

    weight: jax.Array
    idxarray: jax.Array

    def __init__(self, out_features, in_features, idxarray, kernel_len, npoint, layer0, key, dtype: jnp.dtype = jnp.float32):

        if layer0 == True:
            nelems = 2*kernel_len**2
            idxarray = idxarray[:,:2] % nelems
        else:
            nelems = npoint*kernel_len**2
        
        self.weight = jax.random.normal(key, [out_features,in_features,nelems],dtype=dtype)/(in_features*nelems/2)**0.5
        self.idxarray = idxarray 

        super().__init__() 

    def __call__(self,x):
        
        lattice = get_lattice()

        x = x.reshape(1,-1,lattice.shape[1],lattice.shape[2])

        padx = self.idxarray.shape[-2]//2
        pady = self.idxarray.shape[-1]//2

        x = jnp.concatenate((x[:,:,-padx:],x,x[:,:,:padx]),axis=-2)
        x = jnp.concatenate((x[:,:,:,-pady:],x,x[:,:,:,:pady]),axis=-1)

        weight = self.weight[...,self.idxarray]

        weight = weight.transpose(0,2,1,3,4,5)
        weight = weight.reshape(weight.shape[0]*weight.shape[1],-1,weight.shape[4],weight.shape[5])

        x = x.astype(weight)
        
        return jax.lax.conv(x,weight,(1,1),'Valid')
