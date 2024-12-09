from typing import Optional, Callable
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from equinox.nn import Conv, Lambda
import matplotlib.pyplot as plt
from ..sites import TriangularB
from ..nn import Sequential, ReshapeConv
from ..utils import neel, stripe
from ..global_defs import get_lattice, get_subkeys


def sign(neg: bool = False) -> Callable:
    if neg:
        fn = lambda x: -jnp.sign(jnp.sum(jnp.cos(x)))
    else:
        fn = lambda x: jnp.sign(jnp.sum(jnp.cos(x)))
    return Lambda(fn)


def phase(neg: bool = False) -> Callable:
    def fn(x):
        x = jnp.sum(jnp.exp(1j * x))
        x /= jnp.abs(x)
        if neg:
            x *= -1
        return x

    return Lambda(fn)


def cos(neg: bool = False) -> Callable:
    if neg:
        fn = lambda x: -jnp.mean(jnp.cos(x))
    else:
        fn = lambda x: jnp.mean(jnp.cos(x))
    return Lambda(fn)


class SgnNet(Sequential):
    """
    The network used for expressing simple sign structures.
    """

    def __init__(
        self,
        kernel: Optional[jax.Array] = None,
        output: str = "sign",
        neg: bool = False,
        dtype: jnp.dtype = jnp.float32,
    ):
        """
        :param kernel:
            The convolutional kernel used for generating the sign structure, default
            to zeros.

        :param output:
            The form of the network output, chosen among "sign", "phase", and "cos".

        :param neg:
            Whether an additional negative sign should be applied to the sign structure.

        :param dtype:
            The data type of the parameters.
        """
        if output == "sign":
            actfn = sign(neg)
        elif output == "phase":
            actfn = phase(neg)
        elif output == "cos":
            actfn = cos(neg)
        else:
            raise ValueError

        lattice = get_lattice()
        key = get_subkeys()
        conv = Conv(
            num_spatial_dims=lattice.ndim,
            in_channels=lattice.shape[0],
            out_channels=1,
            kernel_size=lattice.shape[1:],
            padding="SAME",
            padding_mode="CIRCULAR",
            use_bias=False,
            dtype=dtype,
            key=key,
        )
        if kernel is None:
            if output == "sign":
                kernel = jnp.zeros_like(conv.weight)
            else:
                kernel = jr.normal(key, conv.weight.shape, dtype) * jnp.pi / 4
        else:
            kernel = kernel.reshape(conv.weight.shape).astype(dtype)
        conv = eqx.tree_at(lambda tree: tree.weight, conv, kernel)
        layers = [ReshapeConv(dtype), conv, actfn]
        super().__init__(layers, holomorphic=False)

    def plot(self):
        """
        Plot the sign kernel.
        """
        params = jax.tree_util.tree_flatten(self.layers[1])[0][0]
        params = params[0, 0]
        params = params - jnp.round(params / jnp.pi) * jnp.pi
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        pic = ax.imshow(
            params,
            cmap=plt.get_cmap("twilight"),
            interpolation="nearest",
            vmin=-jnp.pi / 2,
            vmax=jnp.pi / 2,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(pic, fraction=0.046, pad=0.04)
        return fig


def MarshallSign(output: str = "sign") -> SgnNet:
    L = get_lattice().N
    neg = (L // 4) % 2 == 1
    net = SgnNet(jnp.pi / 4 * neel(), output, neg)
    return net


def StripeSign(output: str = "sign", alternate_dim: int = 1) -> SgnNet:
    L = get_lattice().N
    neg = (L // 4) % 2 == 1
    net = SgnNet(jnp.pi / 4 * stripe(alternate_dim), output, neg)
    return net


def Neel120(output: str = "phase") -> SgnNet:
    lattice = get_lattice()
    Lx, Ly = lattice.shape[1:]
    x = 2 * jnp.arange(Lx)
    if isinstance(lattice, TriangularB):
        y = jnp.zeros(Ly, dtype=x.dtype)
    else:
        y = jnp.arange(Ly)
    kernel = (x[:, None] + y[None, :]) % 3
    kernel = jnp.pi / 3 * kernel - jnp.pi / 6
    net = SgnNet(kernel, output)
    return net
