from typing import Optional, Union, Sequence, Tuple
from matplotlib.figure import Figure
import numpy as np
from .sites import Sites


class Lattice(Sites):
    """
    A special kind of ``Sites`` with periodic structure in real space.
    """

    def __init__(
        self,
        extent: Sequence[int],
        basis_vectors: Sequence[float],
        site_offsets: Optional[Sequence[float]] = None,
        boundary: Union[int, Sequence[int]] = 1,
        Nparticle: Union[None, int, Tuple[int, int]] = None,
        is_fermion: bool = False,
        double_occ: Optional[bool] = None,
    ):
        """
        :param extent: Number of copies in each basis vector direction.
        :param basis_vectos:
            Basis vectors of the lattice. Should be a 2D array with
            different rows for different basis vectors.
        :param site_offsets:
            The atom coordinates in the unit cell. Set it to None to indicate 1 atom
            per cell without offest.
            Otherwise, this should be a 2D array with different rows for different sites
            in a cell.
        :param boundary:
            Boundary condition of the system. It can be an int specifying the boundary
            for all axes, or a sequence of ints each for an axis.
            The meaning of each number is

                1: Periodic boundary condition (PBC)

                0: Open boundary condition (OBC)

                -1: Anti-periodic boundary condition (APBC)

        :param is_fermion:
            Whether the system is made of fermions or spins. Default to False (spins).
        :param double_occ: Whether double occupancy is allowed. Default to False
            for spin systems and True for fermion systems.
        """
        ndim = len(extent)
        self._basis_vectors = np.asarray(basis_vectors, dtype=float)
        if site_offsets is None:
            self._site_offsets = np.zeros([1, ndim], dtype=float)
        else:
            self._site_offsets = np.asarray(site_offsets, dtype=float)
        self._shape = (self._site_offsets.shape[0],) + tuple(extent)

        if isinstance(boundary, int):
            self._boundary = np.full(ndim, boundary, dtype=int)
        else:
            self._boundary = np.asarray(boundary, dtype=int)
        if np.any(self._boundary == -1) and not is_fermion:
            raise ValueError(
                "Spin system can't have anti-periodic boundary conditions."
            )

        N = np.prod(self._shape).item()
        index = np.arange(N, dtype=int)
        xyz = []
        for i in range(len(self._shape)):
            num_later = np.prod(self._shape[i + 1 :], dtype=int)
            xyz.append(index // num_later % self._shape[i])
        self._index_from_xyz = index.reshape(self._shape)
        self._xyz_from_index = np.stack(xyz, axis=1)

        coord = np.zeros(ndim, dtype=float)
        for ext, basis in zip(extent, self._basis_vectors):
            grid = np.arange(ext, dtype=float)
            grid = np.einsum("i,j->ji", basis, grid)
            coord = np.expand_dims(coord, -2) + grid
        offsets = self._site_offsets.reshape([-1] + [1] * len(extent) + [ndim])
        coord = np.expand_dims(coord, 0) + offsets
        coord = coord.reshape(-1, ndim)

        super().__init__(N, Nparticle, is_fermion, double_occ, coord)

    @property
    def shape(self) -> np.ndarray:
        """
        Shape of the lattice. The first element is the site number in a unit cell,
        and the remainings are the spatial extent.
        """
        return self._shape

    @property
    def ncells(self) -> np.ndarray:
        """Number of lattice cells."""
        return np.prod(self.shape[1:])

    @property
    def basis_vectors(self) -> np.ndarray:
        """Basis vectors of the lattice"""
        return self._basis_vectors

    @property
    def site_offsets(self) -> np.ndarray:
        """Site offsets in a unit cell"""
        return self._site_offsets

    @property
    def boundary(self) -> np.ndarray:
        """Boundary condition for each dimension"""
        return self._boundary

    @property
    def index_from_xyz(self) -> np.ndarray:
        """
        A numpy array with ``index_from_xyz[index_in_unit_cell, x, y, z] = index``
        """
        return self._index_from_xyz

    @property
    def xyz_from_index(self) -> np.ndarray:
        """
        A numpy array with ``xyz_from_index[index] = [index_in_unit_cell, x, y, z]``
        """
        return self._xyz_from_index

    def _get_dist_sign(self) -> Tuple[np.ndarray, np.ndarray]:
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

        displacement = displacement.astype(float)
        displacement = np.einsum("...i,ij->...j", displacement, self.basis_vectors)
        # displacement vector of offsets
        offset = self.site_offsets[:, None, :] - self.site_offsets[None, :, :]
        # total displacement vector
        displacement = displacement[..., None, None, :] + offset
        # distance
        dist = np.linalg.norm(displacement, axis=-1, keepdims=True)
        sign = np.ones_like(dist, dtype=int)
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
        dist = [self._slice_diff(idx, dist) for idx in range(self.N)]
        dist = np.stack(dist, axis=0)
        sign = [self._slice_diff(idx, sign) for idx in range(self.N)]
        sign = np.stack(sign, axis=0)
        return dist, sign

    def _slice_diff(self, index: int, diff: np.ndarray) -> np.ndarray:
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

    def plot(
        self,
        figsize: Sequence[Union[int, float]] = (10, 10),
        markersize: Optional[Union[int, float]] = None,
        color_in_cell: Optional[Sequence[str]] = None,
        show_index: bool = True,
        index_fontsize: Optional[Union[int, float]] = None,
        neighbor_bonds: Union[int, Sequence[int]] = 1,
    ) -> Figure:
        """
        Plot the sites and neighbor bonds in the real space, with the adjusted color
        for lattice.

        :param figsize: Figure size.
        :param markersize: Size of markers that represent the sites.
        :param color_in_cell:
            A list containing colors for different sites with the same
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
        if color_in_cell is not None:
            color_site = color_in_cell
        else:
            color_site = [f"C{i}" for i in range(self.shape[0])]
        color_site = [color for color in color_site for _ in range(self.ncells)]
        return super().plot(
            figsize, markersize, color_site, show_index, index_fontsize, neighbor_bonds
        )
