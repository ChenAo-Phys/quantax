from typing import TYPE_CHECKING, Sequence, overload
from numpy.typing import ArrayLike, NDArray
import numpy as np
import jax
from .sites import Sites
from ..global_defs import PARTICLE_TYPE

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class Lattice(Sites):
    """
    A special kind of ``Sites`` with periodic structure in real space.
    """

    def __init__(
        self,
        extent: Sequence[int] | NDArray[np.integer],
        basis_vectors: Sequence[Sequence[float]] | NDArray,
        site_offsets: Sequence[Sequence[float]] | NDArray | None = None,
        boundary: int | Sequence[int] | NDArray[np.integer] = 1,
        particle_type: PARTICLE_TYPE | str = PARTICLE_TYPE.spin,
        Nparticles: int | tuple[int, int] | None = None,
        double_occ: bool | None = None,
    ):
        """
        :param extent:
            Number of copies in each basis vector direction.

        :param basis_vectors:
            Basis vectors of the lattice. Should be a 2D array with
            different rows for different basis vectors.

        :param site_offsets:
            The site coordinates in the unit cell. By default, there is only one site
            at the origin of the unit cell. Otherwise, this should be a 2D array with
            different rows for different sites in a cell.

        :param boundary:
            Boundary condition of the system. It can be an int specifying the boundary
            for all axes, or a sequence of ints each for an axis.
            The meaning of each number is

            - 1: Periodic boundary condition (PBC)
            - 0: Open boundary condition (OBC)
            - -1: Anti-periodic boundary condition (APBC)

            APBC is not allowed for spin systems.

        :param particle_type:
            The particle type of the system: spin, spinful fermion, or spinless
            fermion. Specify it with a `~quantax.PARTICLE_TYPE` member, or equivalently
            its name as a string (e.g. ``"spinful_fermion"``).

        :param Nparticles:
            The number of particles in the system.
            If unspecified, the particle number is non-conserved, except spin systems
            which default to ``Nsites`` (i.e. no magnetization conservation, since the
            total spin count is always ``Nsites``).
            If specified, use an int for the total particle number, or a tuple
            ``(n_up, n_dn)`` for the number of spin-up and spin-down particles. For spin
            systems the total is always ``Nsites``, so a magnetization sector must be
            fixed with a tuple ``(n_up, n_dn)`` summing to ``Nsites`` rather than an int.

        :param double_occ:
            Whether double occupancy is allowed. Default to True for spinful fermions and
            False otherwise.
        """
        ndim = len(extent)
        self._basis_vectors = np.asarray(basis_vectors, dtype=np.float64)
        self._reciprocal_vectors = 2 * np.pi * np.linalg.inv(self._basis_vectors).T
        if site_offsets is None:
            self._site_offsets = np.zeros([1, ndim], dtype=np.float64)
        else:
            self._site_offsets = np.asarray(site_offsets, dtype=np.float64)
        self._shape = (self._site_offsets.shape[0],) + tuple(extent)

        if isinstance(boundary, int):
            self._boundary = np.full(ndim, boundary, dtype=np.int64)
        else:
            self._boundary = np.asarray(boundary, dtype=np.int64)
        if np.any(self._boundary == -1) and particle_type == PARTICLE_TYPE.spin:
            raise ValueError(
                "Spin system can't have anti-periodic boundary conditions."
            )

        Nsites = np.prod(self._shape).item()
        index = np.arange(Nsites, dtype=np.int64)
        self._index_from_xyz = index.reshape(self._shape)
        self._xyz_from_index = np.stack(np.unravel_index(index, self._shape), axis=1)

        # coord = sum_axis(cell_index_along_axis * basis_vector) + offset in the cell
        cell, spatial = self._xyz_from_index[:, 0], self._xyz_from_index[:, 1:]
        coord = spatial @ self._basis_vectors + self._site_offsets[cell]

        super().__init__(Nsites, particle_type, Nparticles, double_occ, coord)

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Shape of the lattice. The first element is the number of sites in a unit cell,
        and the rest are the spatial extent.
        """
        return self._shape

    @property
    def ncells(self) -> int:
        """Number of lattice cells."""
        return np.prod(self.shape[1:]).item()

    @property
    def basis_vectors(self) -> NDArray[np.float64]:
        """Basis vectors of the lattice."""
        return self._basis_vectors

    @property
    def reciprocal_vectors(self) -> NDArray[np.floating]:
        """Reciprocal lattice vectors."""
        return self._reciprocal_vectors

    @property
    def site_offsets(self) -> NDArray[np.float64]:
        """Site offsets in a unit cell."""
        return self._site_offsets

    @property
    def boundary(self) -> NDArray[np.int64]:
        """Boundary condition for each dimension."""
        return self._boundary

    @property
    def index_from_xyz(self) -> NDArray[np.int64]:
        """
        A numpy array with ``index_from_xyz[index_in_unit_cell, x, y, z] = index``.
        """
        return self._index_from_xyz

    @property
    def xyz_from_index(self) -> NDArray[np.int64]:
        """
        A numpy array with ``xyz_from_index[index] = [index_in_unit_cell, x, y, z]``.
        """
        return self._xyz_from_index

    def _slice_diff(self, index: int, diff: NDArray) -> NDArray:
        """
        Slice the diff array with dimension [x1, y1, z1, x2, y2, z2, c1, c2]
        """
        xyz = self.xyz_from_index[index]
        dist_sliced = diff[..., xyz[0]]
        for axis, coord in enumerate(xyz[1:]):
            slices = [np.arange(-coord, 0), np.arange(self.shape[axis + 1] - coord)]
            slices = np.concatenate(slices)
            dist_sliced = dist_sliced.take(slices, axis)
        dist_sliced = np.moveaxis(dist_sliced, -1, 0)
        dist_sliced = dist_sliced.flatten()
        return dist_sliced

    def _get_dist_sign(self) -> tuple[NDArray, NDArray]:
        """
        Computes the distance between sites. The boundary condition is considered
        and only the distance through the shortest path will be obtained.
        """
        # displacement vector without offsets
        displacement = self.xyz_from_index[: self.ncells, 1:]
        displacement = displacement.reshape(*self.shape[1:], self.ndim)
        for axis, extent in enumerate(self.shape[1:]):
            flip = displacement.take(np.arange(extent - 1, 0, -1), axis)
            flip[..., axis] *= -1
            displacement = np.concatenate([displacement, flip], axis)
        # now displacement[x, y, z] = [x, y, z] for x, y, z from -L+1 to L-1

        displacement = displacement.astype(np.float64)
        displacement = np.einsum("...i,ij->...j", displacement, self.basis_vectors)
        # displacement vector of offsets
        offset = self.site_offsets[:, None, :] - self.site_offsets[None, :, :]
        # total displacement vector
        displacement = displacement[..., None, None, :] + offset
        # distance
        dist = np.linalg.norm(displacement, axis=-1, keepdims=True)
        sign = np.ones_like(dist, dtype=np.int64)
        for axis, bc in enumerate(self.boundary):
            if bc != 0:
                indices = [0]
                indices += list(range(-self.shape[axis + 1] + 1, 0))
                indices += list(range(1, self.shape[axis + 1]))
                indices = np.asarray(indices)
                dist_pbc = dist.take(indices, axis)
                dist = np.concatenate([dist, dist_pbc], axis=-1)
                sign = np.concatenate([sign, bc * sign], axis=-1)

        argmin = np.argmin(dist, axis=-1, keepdims=True)
        dist = np.take_along_axis(dist, argmin, axis=-1)[..., 0]  # min(dist, axis=-1)
        sign = np.take_along_axis(sign, argmin, axis=-1)[..., 0]
        dist = [self._slice_diff(idx, dist) for idx in range(self.Nsites)]
        dist = np.stack(dist, axis=0)
        sign = [self._slice_diff(idx, sign) for idx in range(self.Nsites)]
        sign = np.stack(sign, axis=0)
        return dist, sign

    def orbitals(
        self, use_real: bool = False
    ) -> NDArray[np.floating | np.complexfloating]:
        r"""
        Get the single-particle orbitals in momentum space, sorted by tight-binding
        energy.

        :param use_real:
            Whether to return real-valued orbitals.

        :return:
            Orbital $\phi_{i\alpha}$ of shape (Nsites, Nsites), where i represents
            different sites and $\alpha$ represents different k-orbitals. For lattices
            with multiple sites per unit cell, the orbitals are block-diagonal in the
            sublattice, i.e. each k-orbital is a plane wave localized on one sublattice.
        """
        shape = np.asarray(self.shape[1:])
        ncells = self.ncells

        # k-points in the first Brillouin zone, one per unit cell. The first ``ncells``
        # sites are sublattice 0 and run over all cells, so their xyz are the cell grid.
        kpts = self.xyz_from_index[:ncells, 1:]
        k = (kpts / shape[None]) @ self.reciprocal_vectors
        if use_real:
            # Keep one k of each {k, -k} pair (the smaller flat index); self-paired
            # points are kept once. Works for even and odd extents alike.
            flat = np.ravel_multi_index(kpts.T, shape)
            neg_flat = np.ravel_multi_index(((-kpts) % shape).T, shape)
            k = k[flat <= neg_flat]

        ka = np.einsum("ni,mi->nm", k, self.basis_vectors)
        E0 = -2 * np.sum(np.cos(ka), axis=1)  # tight-binding energy
        k = k[np.argsort(E0)]

        # plane waves on a single sublattice (the first ``ncells`` sites)
        kr = np.einsum("ki,ni->nk", k, self.coord[:ncells])
        if use_real:
            orbs1 = np.cos(kr) * np.sqrt(2 / ncells)
            orbs2 = np.sin(kr) * np.sqrt(2 / ncells)
            orbs = np.stack([orbs1, orbs2], axis=2).reshape(ncells, -1)
            all_zero = np.all(np.isclose(orbs, 0.0), axis=0)
            orbs[:, np.flatnonzero(all_zero) - 1] /= np.sqrt(2)
            orbs = orbs[:, ~all_zero]
        else:
            orbs = np.exp(1j * kr) / np.sqrt(ncells)

        # tile block-diagonally over the sublattices; a no-op when there is one site
        return np.kron(np.eye(self.shape[0]), orbs)

    @overload
    def to_neighbor_repr(self, x: NDArray) -> NDArray: ...
    @overload
    def to_neighbor_repr(self, x: jax.Array) -> jax.Array: ...

    def to_neighbor_repr(self, x: NDArray | jax.Array) -> NDArray | jax.Array:
        """
        Rearrange per-site features so that sites adjacent in the array are also
        neighbors on the lattice. For a generic lattice the two orderings already
        coincide, so this is the identity; lattices whose default site ordering does
        not match adjacency (e.g. `TriangularB`) override it.
        The output array type matches the input (NumPy in, NumPy out; JAX in, JAX out).
        """
        return x

    @overload
    def to_original_repr(self, x: NDArray) -> NDArray: ...
    @overload
    def to_original_repr(self, x: jax.Array) -> jax.Array: ...

    def to_original_repr(self, x: NDArray | jax.Array) -> NDArray | jax.Array:
        """
        Inverse of `to_neighbor_repr`, mapping the neighbor representation back to the
        original site ordering. Identity for a generic lattice.
        The output array type matches the input (NumPy in, NumPy out; JAX in, JAX out).
        """
        return x

    def plot(
        self,
        figsize: ArrayLike = (10, 10),
        markersize: int | float | None = None,
        color: str | tuple[str, ...] | None = None,
        show_index: bool = True,
        index_fontsize: int | float | None = None,
        neighbor_bonds: int | Sequence[int] = 1,
    ) -> "Figure":
        """
        Plot the sites and neighbor bonds in the real space, with the adjusted color
        for lattice.

        :param figsize: Figure size.
        :param markersize: Size of markers that represent the sites.
        :param color:
            A tuple containing colors for different sites with the same
            offset in the unit cell. The length should be the same as the number of
            sites in a single unit cell.
        :param show_index: Whether to show index number at each site.
        :param index_fontsize: Fontsize if the index number is shown.
        :param neighbor_bonds:
            The n'th-nearest neighbor bonds to show.
            If this is a sequence, then multiple neighbors will be shown.
            Set this to 0 to hide all neighbor bonds.

        :return: A matplotlib figure containing the plot of lattice.
        """
        if color is not None:
            if isinstance(color, str):
                color_site = tuple(color for _ in range(self.shape[0]))
            else:
                color_site = color
        else:
            color_site = tuple(f"C{i}" for i in range(self.shape[0]))
        color_site = tuple(color for color in color_site for _ in range(self.ncells))
        return super().plot(
            figsize, markersize, color_site, show_index, index_fontsize, neighbor_bonds
        )
