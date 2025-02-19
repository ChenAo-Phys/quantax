from typing import Optional, Union, Sequence, Callable
from jaxtyping import Key
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from equinox.nn import Conv
from ..nn import (
    Sequential,
    apply_he_normal,
    Exp,
    Scale,
    pair_cpl,
    ReshapeConv,
    ConvSymmetrize,
    Gconv,
)
from ..symmetry import Symmetry, Trans2D
from ..symmetry.symmetry import _reordering_perm
from ..global_defs import get_lattice, is_default_cpl, get_subkeys
from functools import partial
from quantax.sites import Grid, Triangular, TriangularB
from ..utils import Reshape_TriangularB, ReshapeTo_TriangularB

class _ResBlock(eqx.Module):
    """Residual block"""

    conv1: Conv
    conv2: Conv
    nblock: int = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        nblock: int,
        total_blocks: int,
        dtype: jnp.dtype = jnp.float32,
    ):
        lattice = get_lattice()

        def new_layer(is_first_layer: bool, is_last_layer: bool) -> Conv:
            if is_first_layer:
                in_channels = lattice.shape[0]
                if lattice.is_fermion:
                    in_channels *= 2
            else:
                in_channels = channels
            key = get_subkeys()
            conv = Conv(
                num_spatial_dims=lattice.ndim,
                in_channels=in_channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding="SAME",
                use_bias=not is_last_layer,
                padding_mode="CIRCULAR",
                dtype=dtype,
                key=key,
            )
            conv = apply_he_normal(key, conv)
            return conv

        self.conv1 = new_layer(nblock == 0, False)
        self.conv2 = new_layer(False, nblock == total_blocks - 1)
        self.nblock = nblock

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        residual = x.copy()
        x /= np.sqrt(self.nblock + 1, dtype=x.dtype)

        if self.nblock == 0:
            x /= np.sqrt(2, dtype=x.dtype)
        else:
            x = jax.nn.gelu(x)
        x = self.conv1(x)
        x = jax.nn.gelu(x)
        x = self.conv2(x)

        if x.shape[0] > residual.shape[0]:
            residual = jnp.repeat(residual, x.shape[0] // residual.shape[0], axis=0)
        return x + residual


def ResSum(
    nblocks: int,
    channels: int,
    kernel_size: Union[int, Sequence[int]],
    final_activation: Optional[Callable] = None,
    trans_symm: Optional[Symmetry] = None,
    dtype: jnp.dtype = jnp.float32,
):
    """
    The convolutional residual network with a summation in the end.

    :param nblocks:
        The number of residual blocks. Each block contains two convolutional layers.

    :param channels:
        The number of channels. Each layer has the same amount of channels.

    :param kernel_size:
        The kernel size. Each layer has the same kernel size.

    :param final_activation:
        The activation function in the last layer.
        By default, `~quantax.nn.Exp` is used.

    :param trans_symm:
        The translation symmetry to be applied in the last layer, see `~quantax.nn.ConvSymmetrize`.

    :param dtype:
        The data type of the parameters.

    .. tip::
        This is the recommended architecture for deep neural quantum states.
    """
    if np.issubdtype(dtype, np.complexfloating):
        raise ValueError("`ResSum` doesn't support complex dtypes.")

    blocks = [
        _ResBlock(channels, kernel_size, i, nblocks, dtype) for i in range(nblocks)
    ]

    scale = Scale(1 / np.sqrt(nblocks + 1))
    layers = [ReshapeConv(dtype), *blocks, scale]

    if is_default_cpl():
        cpl_layer = eqx.nn.Lambda(lambda x: pair_cpl(x))
        layers.append(cpl_layer)

    if final_activation is None:
        final_activation = Exp()
    elif not isinstance(final_activation, eqx.Module):
        final_activation = eqx.nn.Lambda(final_activation)

    layers.append(final_activation)
    layers.append(ConvSymmetrize(trans_symm))

    return Sequential(layers, holomorphic=False)


class _ResBlockGconv(eqx.Module):
    """Residual block"""

    conv1: Conv
    conv2: Conv
    nblock: int = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        idxarray: jax.Array,
        npoint: int,
        nblock: int,
        dtype: jnp.dtype = jnp.float32,
    ):

        def new_layer() -> Conv:
            key = get_subkeys()
            conv = Gconv(
                channels,
                channels,
                idxarray,
                npoint,
                False,
                key,
                dtype,
            )
            return conv

        self.conv1 = new_layer()
        self.conv2 = new_layer()
        self.nblock = nblock

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        residual = x.copy()

        x /= (self.nblock + 1) ** 0.5

        x = jax.nn.gelu(x)
        x = self.conv1(x)
        x = jax.nn.gelu(x)
        x = self.conv2(x)

        return x + residual

def ResSumGconv(
    nblocks: int,
    channels: int,
    pg_symm: Symmetry,
    final_activation: Optional[Callable] = None,
    project: bool = True,
    dtype: jnp.dtype = jnp.float32,
):
    """
    The convolutional residual network with a summation in the end.

    :param nblocks:
        The number of residual blocks. Each block contains two convolutional layers.

    :param channels:
        The number of channels. Each layer has the same amount of channels.

    :param kernel_size:
        The kernel size. Each layer has the same kernel size.

    :param final_activation:
        The activation function in the last layer.
        By default, `~quantax.nn.Exp` is used.

    :param trans_symm:
        The translation symmetry to be applied in the last layer, see `~quantax.nn.ConvSymmetrize`.

    :param dtype:
        The data type of the parameters.

    .. tip::
        This is the recommended architecture for deep neural quantum states.
    """
    if np.issubdtype(dtype, np.complexfloating):
        raise ValueError("`ResSum` doesn't support complex dtypes.")
    
    trans_symm = Trans2D()

    lattice = get_lattice()
    if isinstance(lattice, TriangularB):
        reshape = Reshape_TriangularB(dtype)
    else:
        reshape = ReshapeConv(dtype)

    idxarray, npoint = compute_idxarray(pg_symm, trans_symm)

    embedding = Gconv(channels,1,idxarray,npoint,True,get_subkeys(),dtype)

    blocks = [
        _ResBlockGconv(channels, idxarray, npoint, i, dtype)
        for i in range(nblocks)
    ]

    scale = Scale(1 / np.sqrt(nblocks + 1))
    layers = [reshape,embedding,*blocks, scale]

    layers.append(eqx.nn.Lambda(lambda x: jnp.squeeze(x)))
    if isinstance(lattice, TriangularB):
        layers.append(ReshapeTo_TriangularB(dtype))

    if is_default_cpl():
        cpl_layer = eqx.nn.Lambda(lambda x: pair_cpl(x))
        layers.append(cpl_layer)

    if final_activation is None:
        final_activation = Exp()
    elif not isinstance(final_activation, eqx.Module):
        final_activation = eqx.nn.Lambda(final_activation)

    layers.append(final_activation)
    output_reshape = eqx.nn.Lambda(lambda x: x.reshape(channels,-1))
    layers.append(output_reshape)

    if project == True:
        output_transpose= eqx.nn.Lambda(lambda x: x.reshape(channels,npoint,-1).swapaxes(1,2))
        layers.append(output_transpose)
        layers.append(ConvSymmetrize(trans_symm + pg_symm))
    else:
        perm = _reordering_perm(pg_symm, trans_symm)
        reordering_layer = eqx.nn.Lambda(lambda x: x[:,perm].reshape(channels,npoint,-1))
        layers.append(reordering_layer)

    return Sequential(layers, holomorphic=False)

def compute_idxarray(pg_symm, trans_symm):
    
    lattice = get_lattice()

    pg_perms = jnp.argsort(pg_symm._perm)
    trans_perms = trans_symm._perm
    
    @partial(jax.vmap,in_axes=(0,None))
    @partial(jax.vmap,in_axes=(None,0))
    def take(x,y):
        return x[y]

    perms = take(pg_perms,trans_perms)
    perms = perms.reshape(-1,perms.shape[-1])
        
    npoint = len(perms)//(lattice.shape[1]*lattice.shape[2])        

    perms = perms.reshape(npoint, lattice.shape[1],lattice.shape[2],-1)
    inv_perms = jnp.argsort(perms[:,0,0],-1)

    lattice = get_lattice()
    if isinstance(lattice,Grid) and lattice.ndim == 2:
        mask1 = jnp.asarray([-1,-1,-1,0,0,0,1,1,1])
        mask2 = jnp.asarray([-1,0,1,-1,0,1,-1,0,1])        
    elif isinstance(lattice,Triangular):
        mask1 = jnp.asarray([-1,-1,0,0,0,1,1])
        mask2 = jnp.asarray([0,1,-1,0,1,-1,0])
    elif isinstance(lattice,TriangularB):
        mask1 = jnp.asarray([-1,-2,1,0,-1,2,1])
        mask2 = jnp.asarray([0,1,-1,0,1,-1,0])
    else:
        raise ValueError('No GCNN defined for this lattice type')

    perms = perms[:,mask1,mask2]
    
    perms = perms.reshape(-1,perms.shape[-1])

    idxarray = jnp.zeros([npoint,npoint*len(mask1)],dtype=jnp.int16)

    for i, inv_perm in enumerate(inv_perms):
        for j, perm in enumerate(perms):
            comp_perm = perm[inv_perm][None]
            
            k = jnp.argmin(jnp.sum(jnp.abs(comp_perm - perms),-1))
            if jnp.amin(jnp.sum(jnp.abs(comp_perm - perms),-1)) != 0:
                print('false',flush=True)

            idxarray = idxarray.at[i,j].set(k.astype(jnp.int16))

    idxarray = idxarray.reshape(npoint,npoint,len(mask1))

    return idxarray, npoint
