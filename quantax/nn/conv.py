from typing import Union, Sequence, Optional
from jaxtyping import Key
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from .initializers import apply_lecun_normal


class Depthwise_Separable_Conv(eqx.Module, strict=True):
    """
    Depth-wise separable convolution, expressed as the composition of a depth-wise
    convolution and a point-wise convolution. See https://www.youtube.com/watch?v=vVaRhZXovbw
    The kernel_size, stride, padding, dilation, and padding_mode arguments only apply to conv1. 
    The groups argument only applies to conv2.
    """

    conv1: eqx.nn.Conv
    conv2: eqx.nn.Conv

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[str, int, Sequence[int], Sequence[tuple[int, int]]] = 0,
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        use_bias: bool = True,
        padding_mode: str = "ZEROS",
        dtype=None,
        *,
        key: Key,
    ):
        key1, key2 = jr.split(key, 2)
        # depth-wise convolution
        conv1 = eqx.nn.Conv(
            num_spatial_dims,
            in_channels=in_channels,
            out_channels=groups * in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            use_bias=use_bias,
            padding_mode=padding_mode,
            dtype=dtype,
            key=key1,
        )
        self.conv1 = apply_lecun_normal(key1, conv1)

        # point-wise convolution
        conv2 = eqx.nn.Conv(
            num_spatial_dims,
            in_channels=groups * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            groups=groups,
            use_bias=use_bias,
            dtype=dtype,
            key=key2,
        )
        self.conv2 = apply_lecun_normal(key2, conv2)

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        x = self.conv1(x)
        x = x.reshape(-1, self.conv2.groups, *x.shape[1:])
        x = jnp.swapaxes(x, 0, 1).reshape(-1, *x.shape[2:])
        x = self.conv2(x)
        return x
