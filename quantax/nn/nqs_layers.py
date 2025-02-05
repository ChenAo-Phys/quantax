from typing import Optional
from jaxtyping import Key
import jax
import jax.numpy as jnp
import equinox as eqx
from .modules import NoGradLayer, RawInputLayer
from ..symmetry import Symmetry, TransND, Identity
from ..global_defs import get_lattice
from ..utils import _triangularb_circularpad
from ..sites import TriangularB

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

class Gconv(eqx.Module):

    weight: jax.Array
    idxarray: jax.Array

    def __init__(self, out_features, in_features, idxarray, npoint, layer0, key, dtype: jnp.dtype = jnp.float32):

        if layer0 == True:
            nelems = 2*idxarray.shape[-1]
            idxarray = idxarray[:,:2] % nelems
            scale = (1/(in_features*nelems))**0.5
        else:
            nelems = npoint*idxarray.shape[-1]
            scale = (2/(in_features*nelems))**0.5

        self.weight = jax.random.normal(key, [out_features,in_features,nelems],dtype=dtype)*scale
        self.idxarray = idxarray 

        super().__init__() 

    def __call__(self,x):
        
        lattice = get_lattice()

        x = x.reshape(1,-1,lattice.shape[1],lattice.shape[2])

        if isinstance(lattice,TriangularB):
            x = jax.vmap(_triangularb_circularpad)(x)
        else:
            x = jnp.concatenate((x[:,:,-1:],x,x[:,:,:1]),axis=-2)
            x = jnp.concatenate((x[:,:,:,-1:],x,x[:,:,:,:1]),axis=-1)

        weight = self.weight[...,self.idxarray]

        if weight.shape[-1] == 9:
            weight = weight.reshape(*weight.shape[:-1],3,3)
        else:
            zeros = jnp.zeros_like(weight[...,:1])
            weight = jnp.concatenate((zeros,weight,zeros),-1)
            weight = weight.reshape(*weight.shape[:-1],3,3)

        weight = weight.transpose(0,2,1,3,4,5)
        weight = weight.reshape(weight.shape[0]*weight.shape[1],-1,weight.shape[4],weight.shape[5])

        x = x.astype(weight)
        
        return jax.lax.conv(x,weight,(1,1),'Valid')
