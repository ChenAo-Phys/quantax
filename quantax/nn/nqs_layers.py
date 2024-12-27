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

    def __init__(
        self,
        out_features,
        in_features,
        symm,
        kernel_size,
        layer0,
        key,
        dtype: jnp.dtype = jnp.float32,
    ):

        lattice = get_lattice()

        perms = symm._perm

        npoint = len(perms) // (lattice.shape[1] * lattice.shape[2])

        perms = perms.reshape(npoint, lattice.shape[1], lattice.shape[2], -1)
        inv_perms = jnp.argsort(perms[:, 0, 0], -1)

        perms = jnp.roll(perms, (kernel_size[0] // 2, kernel_size[1] // 2), axis=(1, 2))
        perms = perms[:, : kernel_size[0], : kernel_size[1]]
        perms = perms.reshape(-1, perms.shape[-1])

        idxarray = jnp.zeros(
            [npoint, npoint * kernel_size[0] * kernel_size[1]], dtype=jnp.int16
        )

        for i, inv_perm in enumerate(inv_perms):
            for j, perm in enumerate(perms):
                comp_perm = perm[inv_perm][None]

                k = jnp.argmin(jnp.sum(jnp.abs(comp_perm - perms), -1))

                idxarray = idxarray.at[i, j].set(k.astype(jnp.int16))

        idxarray = idxarray.reshape(npoint, npoint, kernel_size[0], kernel_size[1])

        if layer0 == True:
            nelems = 2 * kernel_size[0] * kernel_size[1]
            idxarray = idxarray[:, :2] % nelems
        else:
            nelems = npoint * kernel_size[0] * kernel_size[1]

        self.weight = (
            jax.random.normal(key, [out_features, in_features, nelems], dtype=dtype)
            / (in_features * nelems / 4) ** 0.5
        )
        self.idxarray = idxarray

        super().__init__()

    def __call__(self, x):

        lattice = get_lattice()

        x = x.reshape(1, -1, lattice.shape[1], lattice.shape[2])

        padx = self.idxarray.shape[-2] // 2
        pady = self.idxarray.shape[-1] // 2

        x = jnp.concatenate((x[:, :, -padx:], x, x[:, :, :padx]), axis=-2)
        x = jnp.concatenate((x[:, :, :, -pady:], x, x[:, :, :, :pady]), axis=-1)

        weight = self.weight[..., self.idxarray]

        weight = weight.transpose(0, 2, 1, 3, 4, 5)
        weight = weight.reshape(
            weight.shape[0] * weight.shape[1], -1, weight.shape[4], weight.shape[5]
        )

        x = x.astype(weight)

        return jax.lax.conv(x, weight, (1, 1), "Valid")
