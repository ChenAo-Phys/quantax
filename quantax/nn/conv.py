from jaxtyping import Key
import numpy as np
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
import equinox as eqx
from .modules import RawInputLayer
from ..symmetry import Symmetry, TransND, Identity
from ..global_defs import PARTICLE_TYPE, get_lattice
from ..sites import TriangularB


class ReshapeConv(eqx.Module):
    """
    Reshape the input to the shape suitable for convolutional layers.

    A Fock state in Quantax is usually given by a 1D array with entries +1/-1.
    This layer reshapes it to `~quantax.sites.Lattice.shape`.
    """

    dtype: DTypeLike = eqx.field(static=True)

    def __init__(self, dtype: DTypeLike = jnp.float32):
        """
        :param dtype:
            Convert the input to the given data type, by default ``float32``.
        """
        super().__init__()
        self.dtype = dtype

    def __call__(self, x: jax.Array) -> jax.Array:
        lattice = get_lattice()
        shape = lattice.shape
        if lattice.particle_type == PARTICLE_TYPE.spinful_fermion:
            shape = (shape[0] * 2,) + shape[1:]
        x = x.reshape(shape)
        x = x.astype(self.dtype)
        return x


class ConvSymmetrize(RawInputLayer):
    """
    Symmetrize the output of a convolutional network according to the given symmetry.
    """

    symm: Symmetry = eqx.field(static=True)

    def __init__(self, symm: Symmetry | None = None):
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


class Reshape_TriangularB(eqx.Module):
    """
    Reshape the TriangularB spins into the arrangement of Triangular for more efficient
    convolutions.
    """

    dtype: DTypeLike = eqx.field(static=True)
    permutation: np.ndarray

    def __init__(self, dtype: DTypeLike = jnp.float32):
        self.dtype = dtype
        lattice = get_lattice()
        if not isinstance(lattice, TriangularB):
            raise ValueError("The current lattice is not `TriangularB`.")

        permutation = np.arange(lattice.Nsites, dtype=np.uint16)
        permutation = permutation.reshape(lattice.shape[1:])
        for i in range(permutation.shape[1]):
            permutation[:, i] = np.roll(permutation[:, i], shift=i)

        self.permutation = permutation

    def __call__(self, x: jax.Array) -> jax.Array:
        lattice = get_lattice()
        shape = lattice.shape
        if lattice.particle_type == PARTICLE_TYPE.spinful_fermion:
            shape = (shape[0] * 2,) + shape[1:]
            x = x.reshape(2, -1)

        x = x[..., self.permutation]
        x = x.reshape(shape).astype(self.dtype)
        return x


class ReshapeTo_TriangularB(eqx.Module):
    """
    Reshape the Triangular spins back into the arrangement of TriangularB.
    """

    dtype: DTypeLike = eqx.field(static=True)
    permutation: np.ndarray

    def __init__(self, dtype: DTypeLike = jnp.float32):
        self.dtype = dtype
        lattice = get_lattice()
        if not isinstance(lattice, TriangularB):
            raise ValueError("The current lattice is not `TriangularB`.")

        permutation = np.arange(lattice.Nsites, dtype=np.uint16)
        permutation = permutation.reshape(lattice.shape[1:])
        for i in range(permutation.shape[1]):
            permutation[:, i] = np.roll(permutation[:, i], shift=-i)

        self.permutation = permutation

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x.reshape(x.shape[0], -1)
        x = x[:, self.permutation]
        x = x.reshape(x.shape[0], *get_lattice().shape)
        return x


def triangularb_circularpad(x: jax.Array) -> jax.Array:
    """
    One-cell circular padding for a ``TriangularB`` feature map of shape
    ``(channels, L1, L2)``. In addition to the usual periodic wrap, the columns
    added on the left/right are rolled along the first spatial axis to match the
    skewed periodicity of the ``TriangularB`` arrangement.
    """
    pad_lower = jnp.roll(x[:, :, -1:], shift=-x.shape[2], axis=1)
    pad_upper = jnp.roll(x[:, :, :1], shift=x.shape[2], axis=1)
    x = jnp.concatenate([pad_lower, x, pad_upper], axis=2)
    x = jnp.pad(x, [(0, 0), (1, 1), (0, 0)], mode="wrap")
    return x


class GConv(eqx.Module):
    """
    Group-equivariant convolution layer for 2D square and triangular lattices.

    The trainable weights are stored as a flat per-channel array and gathered
    onto the spatial kernel through ``idxarray``, which encodes the action of
    the point-group symmetry on the kernel positions. This is the building
    block of `~quantax.model.ResGConv`.
    """

    weight: jax.Array
    idxarray: jax.Array

    def __init__(
        self,
        out_features: int,
        in_features: int,
        idxarray: jax.Array,
        npoint: int,
        layer0: bool,
        key: Key,
        dtype: DTypeLike = jnp.float32,
    ):
        """
        :param out_features:
            Number of output channels.

        :param in_features:
            Number of input channels. For the lifting layer (``layer0=True``)
            this is the number of lattice input channels and must not exceed
            ``npoint``.

        :param idxarray:
            Index array mapping the stored weights onto the
            ``(npoint, kernel)`` positions, generated together with ``npoint``
            by the network builder.

        :param npoint:
            Number of point-group elements.

        :param layer0:
            Whether this is the first (lifting) layer that maps the lattice
            input into the group dimension.

        :param key:
            The PRNG key for weight initialization.

        :param dtype:
            The data type of the parameters, by default ``float32``.
        """
        if layer0:
            # Lifting layer: each of the `in_features` input channels is gathered
            # onto the group dimension as a group-transformed kernel. The stored
            # weights are a flat `in_features * kernel` table indexed per channel.
            if in_features > npoint:
                raise ValueError(
                    f"The lifting `GConv` supports at most {npoint} input channels "
                    f"(the point-group size), but got `in_features={in_features}`."
                )
            nelems = in_features * idxarray.shape[-1]
            idxarray = idxarray[:, :in_features] % nelems
            scale = (1 / nelems) ** 0.5
            weight_in = 1
        else:
            nelems = npoint * idxarray.shape[-1]
            scale = (2 / (in_features * nelems)) ** 0.5
            weight_in = in_features

        self.weight = (
            jax.random.normal(key, [out_features, weight_in, nelems], dtype=dtype)
            * scale
        )
        self.idxarray = idxarray

        super().__init__()

    def __call__(self, x: jax.Array) -> jax.Array:

        lattice = get_lattice()

        x = x.reshape(1, -1, lattice.shape[1], lattice.shape[2])

        if isinstance(lattice, TriangularB):
            x = jax.vmap(triangularb_circularpad)(x)
        else:
            x = jnp.concatenate((x[:, :, -1:], x, x[:, :, :1]), axis=-2)
            x = jnp.concatenate((x[:, :, :, -1:], x, x[:, :, :, :1]), axis=-1)

        weight = self.weight[..., self.idxarray]

        if weight.shape[-1] == 9:
            # Square lattice: the full 3x3 kernel has 9 weights.
            weight = weight.reshape(*weight.shape[:-1], 3, 3)
        else:
            # Triangular lattice: the 7-weight kernel is padded with zeros at
            # the two missing corners to form a 3x3 kernel.
            zeros = jnp.zeros_like(weight[..., :1])
            weight = jnp.concatenate((zeros, weight, zeros), -1)
            weight = weight.reshape(*weight.shape[:-1], 3, 3)

        weight = weight.transpose(0, 2, 1, 3, 4, 5)
        weight = weight.reshape(
            weight.shape[0] * weight.shape[1], -1, weight.shape[4], weight.shape[5]
        )

        x = x.astype(weight.dtype)

        return jax.lax.conv(x, weight, (1, 1), "Valid")
