from typing import Sequence
from jaxtyping import Key
import numpy as np
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
import equinox as eqx
from .modules import RawInputLayer
from .initializers import lecun_normal
from ..symmetry import Symmetry, Translation, Identity
from ..global_defs import PARTICLE_TYPE, get_lattice


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
        """
        Reshape a configuration to the lattice shape and cast it to ``dtype``.
        """
        lattice = get_lattice()
        shape = lattice.shape
        if lattice.particle_type == PARTICLE_TYPE.spinful_fermion:
            shape = (shape[0] * 2,) + shape[1:]
        x = x.reshape(shape)
        x = x.astype(self.dtype)
        return x


class ConvSymmetrize(RawInputLayer):
    r"""
    Symmetrize the output of a convolutional network by translations.

    The input should be a translation-covariant feature map in the original site
    ordering, reshapeable to ``(channels, *lattice.shape[1:])``.
    The layer computes the symmetrized wavefunction by

    - averaging over the channels;
    - averaging over positions along the axes without translation symmetry,
      e.g. open boundaries;
    - keeping one value per translation group element along the translated axes,
      where positions between two group elements are averaged into one value;
    - combining the remaining values with the characters of the symmetry group
      through `~quantax.symmetry.Symmetry.symmetrize`.
    """

    symm: Symmetry
    trans_step: tuple[int, ...] | None

    def __init__(self, symm: Symmetry | None = None):
        """
        :param symm:
            The symmetry used for symmetrization. It can be

            - ``None`` (default): Unit translations are imposed along all periodic
              axes, and outputs are averaged along all open axes. For fully open
              boundaries, this reduces to the average over all positions.

            - `~quantax.symmetry.Identity`: The layer leaves inputs unchanged.

            - `~quantax.symmetry.Translation`: Translations given by axis-aligned
              vectors. An entry :math:`v` on an axis keeps one output every
              :math:`v` positions as a translation group element. Axes not covered
              by any vector are averaged.
        """
        super().__init__()
        lattice = get_lattice()

        if symm is None:
            is_axis_trans = np.asarray(lattice.boundary) != 0
            trans_step = tuple(int(i) for i in is_axis_trans)
            if np.any(is_axis_trans):
                trans_vectors = np.identity(lattice.ndim, dtype=np.int64)[is_axis_trans]
                symm = Translation(trans_vectors)
            else:
                # full OBC: no translations, only the average over all positions
                symm = Identity()
        elif symm is Identity():
            trans_step = None
        elif isinstance(symm, Translation):
            trans_step = np.zeros(lattice.ndim, dtype=np.int64)
            for vec in symm.vectors:
                if np.sum(vec != 0) > 1:
                    raise NotImplementedError(
                        "ConvSymmetrize doesn't support tilted translations."
                    )
                trans_step = np.where(vec != 0, np.abs(vec), trans_step)
            trans_step = tuple(int(n) for n in trans_step)
        else:
            raise ValueError(
                "`ConvSymmetrize` only supports `None`, `Identity`, or `Translation` "
                f"symmetries, but got {type(symm).__name__}."
            )
        self.symm = symm
        self.trans_step = trans_step

    def __call__(self, x: jax.Array, s: jax.Array) -> jax.Array:
        if self.trans_step is None:
            return x

        lattice = get_lattice()
        x = x.reshape(-1, *lattice.shape[1:]).mean(axis=0)
        for axis, step in enumerate(self.trans_step):
            if step == 0:
                # no translation along this axis, average over positions
                x = x.mean(axis=axis, keepdims=True)
            else:
                # split the axis into (translation, step) and average within each
                # step, keeping one value per translation group element
                x = x.reshape(*x.shape[:axis], -1, step, *x.shape[axis + 1 :])
                x = x.mean(axis=axis + 1)

        x = self.symm.symmetrize(x.flatten(), s)

        return x


def _normalize_padding(
    padding: int | Sequence[int] | Sequence[tuple[int, int]], ndim: int
) -> list[tuple[int, int]]:
    """Normalize an ``eqx.nn.Conv``-style ``padding`` to per-dimension
    ``(low, high)`` pairs of length ``ndim``."""
    if isinstance(padding, int):
        return [(padding, padding)] * ndim

    if len(padding) != ndim:
        raise ValueError(
            f"`padding` sequence has length {len(padding)}, expected {ndim} "
            "(one entry per physical dimension)."
        )

    pads = []
    for p in padding:
        if isinstance(p, int):
            pads.append((p, p))
        else:
            low, high = p
            pads.append((int(low), int(high)))
    return pads


def circular_pad(
    x: jax.Array, padding: int | Sequence[int] | Sequence[tuple[int, int]]
) -> jax.Array:
    r"""
    Circular padding on all physical dimensions of a feature map.

    The input ``x`` is given in the *neighbor representation* with shape
    ``(channels, Lx, Ly, ...)``, where the trailing axes must match the lattice
    shape ``get_lattice().shape[1:]`` and the leading axes are channels. Each
    physical dimension is padded according to ``padding``, wrapping around the
    boundary for periodic or anti-periodic boundary conditions.

    The wrap obeys the lattice periodicity in the *original representation*, not a
    naive per-axis wrap of the neighbor grid: for skewed arrangements such as
    `~quantax.sites.TriangularB` a plain wrap is only covariant under a subgroup
    of the translations. The correct (possibly twisted) wrap is derived
    automatically from ``to_neighbor_repr``, assuming one step on the neighbor
    grid translates all sites by the same lattice vector. It reduces to a plain
    wrap when the neighbor grid is an unskewed torus (e.g. square lattices, whose
    ``to_neighbor_repr`` is the identity).

    Implemented as a single gather with a statically-built index, which is
    memory-bandwidth optimal and noticeably faster than ``jnp.pad(mode="wrap")``.

    :param x:
        The feature map of shape ``(channels, Lx, Ly, ...)`` in neighbor
        representation, with spatial shape equal to ``get_lattice().shape[1:]``.

    :param padding:
        The padding on every physical dimension, following the convention of
        `equinox.nn.Conv <https://docs.kidger.site/equinox/api/nn/conv/#equinox.nn.Conv>`__
        (string inputs are not supported). It can be

        - an ``int``: the same padding on both sides of every dimension;
        - a sequence of ``int``: one symmetric padding per dimension;
        - a sequence of ``(low, high)`` pairs: asymmetric padding per dimension.
    """
    lattice = get_lattice()
    ndim = lattice.ndim
    spatial = tuple(int(s) for s in lattice.shape[1:])
    if tuple(x.shape[-ndim:]) != spatial:
        raise ValueError(
            f"`circular_pad` requires the spatial shape of the input to match the "
            f"lattice shape {spatial}, but got {tuple(x.shape[-ndim:])}."
        )
    lead = x.shape[:-ndim]
    pads = _normalize_padding(padding, ndim)
    N = int(np.prod(spatial))

    # original coordinate sitting at each neighbor position, and the neighbor
    # position holding each original site
    coords = lattice.to_neighbor_repr(np.indices(spatial))
    pos_of_orig = lattice.to_original_repr(np.arange(N)).reshape(spatial)

    padded = tuple(spatial[d] + pads[d][0] + pads[d][1] for d in range(ndim))
    # neighbor coords over the padded grid: index 0 maps to neighbor coord -low
    grid = np.indices(padded)
    nbr_coord = [grid[d] - pads[d][0] for d in range(ndim)]

    # One step on the neighbor grid shifts all original coordinates by a constant
    # vector, so the source site of any padded position follows from the origin
    # site and the per-axis steps, wrapped on the original torus.
    zero = (0,) * ndim
    src_coord = []
    for e in range(ndim):
        c = coords[e]
        step = [
            int(c[tuple(1 if k == d else 0 for k in range(ndim))]) - int(c[zero])
            for d in range(ndim)
        ]
        raw = int(c[zero]) + sum(step[d] * nbr_coord[d] for d in range(ndim))
        src_coord.append(raw % spatial[e])
    src = pos_of_orig[tuple(src_coord)].reshape(-1)

    out = x.reshape(*lead, -1)[..., src]
    return out.reshape(*lead, *padded)


def _ntuple(x: int | Sequence[int], n: int) -> tuple[int, ...]:
    if isinstance(x, int):
        return (x,) * n
    x = tuple(x)
    if len(x) != n:
        raise ValueError(f"Expected {n} values, got {len(x)}.")
    return x


class Conv(eqx.Module):
    r"""
    Convolution layer whose padding is determined by the lattice boundary.

    This is analogous to `equinox.nn.Conv <https://docs.kidger.site/equinox/api/nn/conv/#equinox.nn.Conv>`__,
    but it has no ``padding`` or
    ``padding_mode`` arguments: the padding is fixed by the boundary conditions of
    the current lattice (`~quantax.get_lattice`), and the number of spatial
    dimensions is taken from ``lattice.ndim``. For each physical dimension,

    - periodic and anti-periodic dimensions (``boundary != 0``) are wrapped with
      `~quantax.nn.circular_pad` (a "SAME"-sized wrap, no anti-periodic sign);
    - open dimensions (``boundary == 0``) are zero-padded ("SAME") inside
      `jax.lax.conv_with_general_padding`.

    The output therefore has the same spatial shape as the input (for unit
    stride). The input/output are given in the *neighbor representation*.
    """

    weight: jax.Array
    bias: jax.Array | None
    kernel_size: tuple[int, ...] = eqx.field(static=True)
    stride: tuple[int, ...] = eqx.field(static=True)
    dilation: tuple[int, ...] = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        dilation: int | Sequence[int] = 1,
        use_bias: bool = True,
        dtype: DTypeLike = jnp.float32,
        *,
        key: Key,
    ):
        """
        :param in_channels:
            The number of input channels.

        :param out_channels:
            The number of output channels.

        :param kernel_size:
            The size of the convolutional kernel, an int or one int per physical
            dimension.

        :param stride:
            The stride of the convolution, by default 1.

        :param dilation:
            The dilation of the convolution, by default 1.

        :param use_bias:
            Whether to add a bias after the convolution, by default ``True``.

        :param dtype:
            The data type of the parameters, by default ``float32``.

        :param key:
            The PRNG key for weight initialization.
        """
        ndim = get_lattice().ndim
        self.kernel_size = _ntuple(kernel_size, ndim)
        self.stride = _ntuple(stride, ndim)
        self.dilation = _ntuple(dilation, ndim)
        self.use_bias = use_bias

        wshape = (out_channels, in_channels) + self.kernel_size
        self.weight = lecun_normal(key, wshape, dtype)
        if use_bias:
            self.bias = jnp.zeros((out_channels,) + (1,) * ndim, dtype)
        else:
            self.bias = None

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        :param x:
            The input of shape ``(in_channels, *spatial)`` in neighbor
            representation.
        """
        lattice = get_lattice()
        ndim = lattice.ndim
        boundary = lattice.boundary

        # padding that keeps the spatial shape ("SAME"), split per dimension into
        # a periodic wrap (circular_pad) and a zero pad (in the conv itself).
        rhs_shape = tuple(
            d * (k - 1) + 1 for k, d in zip(self.kernel_size, self.dilation)
        )
        same_pads = jax.lax.padtype_to_pads(x.shape[1:], rhs_shape, self.stride, "SAME")
        circ_pads = [same_pads[i] if boundary[i] != 0 else (0, 0) for i in range(ndim)]
        conv_pads = [(0, 0) if boundary[i] != 0 else same_pads[i] for i in range(ndim)]

        if any(p != (0, 0) for p in circ_pads):
            x = circular_pad(x, circ_pads)

        x = jnp.expand_dims(x, axis=0)
        x = jax.lax.conv_with_general_padding(
            x, self.weight, self.stride, conv_pads, None, self.dilation
        )
        x = jnp.squeeze(x, axis=0)

        if self.bias is not None:
            x = x + self.bias
        return x
