from typing import Optional, Callable
from jaxtyping import Key
import numpy as np
import jax
import jax.numpy as jnp
from jax.nn import initializers
import jax.random as jr
from jax import lax
import equinox as eqx
from ..sites import Triangular, TriangularB
from ..symmetry import Symmetry, Identity
from ..nn import (
    lecun_normal,
    he_normal,
    SinhShift,
    Exp,
    Scale,
    pair_cpl,
    ReshapeConv,
    ConvSymmetrize,
    Sequential,
)
from ..global_defs import (
    get_lattice,
    get_params_dtype,
    is_default_cpl,
    is_params_cpl,
    get_subkeys,
)


class Reshape_TriangularB(eqx.Module):
    """
    Reshape the TriangularB spins into the arrangement of Triangular for more efficient
    convolutions.
    """

    permutation: np.ndarray

    def __init__(self):
        lattice = get_lattice()
        if not isinstance(lattice, TriangularB):
            raise ValueError("The current lattice is not `TriangularB`.")

        permutation = np.arange(lattice.nsites, dtype=np.uint16)
        permutation = permutation.reshape(lattice.shape[1:])
        for i in range(permutation.shape[1]):
            permutation[:, i] = np.roll(permutation[:, i], shift=i)

        self.permutation = permutation

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        x = x[self.permutation]
        x = x.reshape(get_lattice().shape)
        x = x.astype(get_params_dtype())
        return x


class ReshapeTo_TriangularB(eqx.Module):
    """
    Reshape the Triangular spins back into the arrangement of TriangularB.
    """

    permutation: np.ndarray

    def __init__(self):
        lattice = get_lattice()
        if not isinstance(lattice, TriangularB):
            raise ValueError("The current lattice is not `TriangularB`.")

        permutation = np.arange(lattice.nsites, dtype=np.uint16)
        permutation = permutation.reshape(lattice.shape[1:])
        for i in range(permutation.shape[1]):
            permutation[:, i] = np.roll(permutation[:, i], shift=-i)

        self.permutation = permutation

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        x = x.reshape(x.shape[0], -1)
        x = x[:, self.permutation]
        x = x.reshape(x.shape[0], *get_lattice().shape)
        return x


def _triangularb_circularpad(x: jax.Array) -> jax.Array:
    pad_lower = jnp.roll(x[:, :, -1:], shift=-x.shape[2], axis=1)
    pad_upper = jnp.roll(x[:, :, :1], shift=x.shape[2], axis=1)
    x = jnp.concatenate([pad_lower, x, pad_upper], axis=2)
    x = jnp.pad(x, [(0, 0), (1, 1), (0, 0)], mode="wrap")
    return x


class Triangular_Neighbor_Conv(eqx.Module):
    """Nearest neighbor convolution for the triangular lattice."""

    weight: jax.Array
    bias: Optional[jax.Array]
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    use_mask: bool = eqx.field(static=True)
    is_triangularB: bool = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_bias: bool = True,
        kernel_init: Callable = lecun_normal,
        bias_init: Callable = initializers.zeros,
        use_mask: bool = False,
        *,
        key: Key,
        **kwargs,
    ):
        lattice = get_lattice()
        if isinstance(lattice, Triangular):
            self.is_triangularB = False
        elif isinstance(lattice, TriangularB):
            self.is_triangularB = True
        else:
            raise ValueError("The current lattice is not triangular.")

        super().__init__(**kwargs)
        wkey, bkey = jr.split(key, 2)
        dtype = get_params_dtype()
        if use_mask:
            kernel_shape = (out_channels, in_channels, 7)
        else:
            kernel_shape = (out_channels, in_channels, 3, 3)
        self.weight = kernel_init(wkey, kernel_shape, dtype)
        if use_bias:
            bias_shape = (out_channels, 1, 1)
            self.bias = bias_init(bkey, bias_shape, dtype)
        else:
            self.bias = None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = use_bias
        self.use_mask = use_mask

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        if x.ndim != 3:
            raise ValueError(f"Input needs to have rank 3, but has shape {x.shape}.")

        x = x.astype(self.weight.dtype)
        if self.is_triangularB:
            x = _triangularb_circularpad(x)
        else:
            x = jnp.pad(x, [(0, 0), (1, 1), (1, 1)], mode="wrap")
        x = jnp.expand_dims(x, axis=0)

        if self.use_mask:
            weight = jnp.pad(self.weight, [(0, 0), (0, 0), (1, 1)])
            weight = weight.reshape(self.out_channels, self.in_channels, 3, 3)
        else:
            weight = self.weight

        x = lax.conv_general_dilated(
            lhs=x, rhs=weight, window_strides=(1, 1), padding="VALID"
        )
        x = jnp.squeeze(x, axis=0)
        if self.use_bias:
            x = x + self.bias
        return x


class _ResBlock(eqx.Module):
    """Residual block"""

    conv1: Triangular_Neighbor_Conv
    conv2: Triangular_Neighbor_Conv
    nblock: int = eqx.field(static=True)

    def __init__(self, channels: int, nblock: int, total_blocks: int):
        def new_layer(is_first: bool, is_last: bool) -> Triangular_Neighbor_Conv:
            lattice = get_lattice()
            in_channels = lattice.shape[0] if is_first else channels
            return Triangular_Neighbor_Conv(
                in_channels=in_channels,
                out_channels=channels,
                use_bias=not is_last,
                kernel_init=he_normal,
                key=get_subkeys(),
            )

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
        return x + residual


def Triangular_ResSum(
    depth: int,
    channels: int,
    use_sinh: bool = False,
    trans_symm: Optional[Symmetry] = None,
):
    lattice = get_lattice()
    if isinstance(lattice, Triangular):
        is_triangularB = False
    elif isinstance(lattice, TriangularB):
        is_triangularB = True
    else:
        raise ValueError("The current lattice is not triangular.")

    if is_params_cpl():
        raise ValueError("'ResSum' only supports real parameters.")
    if depth % 2:
        raise ValueError(f"'depth' should be a multiple of 2, got {depth}")

    num_blocks = depth // 2
    blocks = [_ResBlock(channels, i, num_blocks) for i in range(num_blocks)]

    reshape = Reshape_TriangularB() if is_triangularB else ReshapeConv()
    scale = Scale(1 / np.sqrt(num_blocks))
    layers = [reshape, *blocks, scale]

    if is_default_cpl():
        layers.append(eqx.nn.Lambda(lambda x: pair_cpl(x)))
    layers.append(SinhShift() if use_sinh else Exp())
    if is_triangularB:
        layers.append(ReshapeTo_TriangularB())
    if trans_symm is not Identity():
        layers.append(ConvSymmetrize(trans_symm))

    return Sequential(layers, holomorphic=False)
