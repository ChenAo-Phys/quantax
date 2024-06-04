from typing import Optional, Union, Sequence
import numpy as np
from .sites import Sites


class Lattice(Sites):
    """
    A lattice with periodic structure in real space. Contains the coordinates of sites
    and the Hilbert space dimension at all sites.
    """

    def __init__(
        self,
        extent: Sequence[int],
        basis_vectors: Sequence[float],
        site_offsets: Optional[Sequence[float]] = None,
        boundary: Union[int, Sequence[int]] = 1,
        is_fermion: bool = False,
    ):
        """
        Constructs 'Lattice' given the unit cell and translations.

        Args:
            extent: Number of copies in each basis vector direction.
            basis_vectos: Basis vectors of the lattice. 2D array with different rows
                for different basis vectors.
            site_offsets: The atom coordinates in the unit cell. If None then only
                1 atom without offest. If 2D array, different rows stand for different
                sites.
            boundary: Boundary condition of the system.
                1: Periodic boundary condition (PBC)
                0: Open boundary (OBC)
                -1: Anti-periodic boundary (APBC)
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
        if np.any(self._boundary == -1) and not self._is_fermion:
            raise ValueError(
                "Spin system can't have anti-periodic boundary conditions."
            )

        nsites = np.prod(self._shape)
        index = np.arange(nsites, dtype=int)
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
        coord = np.expand_dims(coord, -2) + self._site_offsets
        coord = coord.reshape(-1, ndim)

        super().__init__(nsites, is_fermion, coord)

    @property
    def shape(self) -> np.ndarray:
        """
        Shape of the lattice. The first element is the site number in a unit cell,
        and the remainings are the extent.
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
        """Whether the periodic boundary condition is used in different directions"""
        return self._boundary

    @property
    def index_from_xyz(self) -> np.ndarray:
        """
        A jax.numpy array with index_from_xyz[x, y, z, index_in_unit_cell] = index
        """
        return self._index_from_xyz

    @property
    def xyz_from_index(self) -> np.ndarray:
        """
        A jax.numpy array with xyz_from_index[index] = [x, y, z, index_in_unit_cell]
        """
        return self._xyz_from_index

    def _compute_dist(self) -> None:
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
        # now displacement[x, y, z] = [x, y, z] for x, y, z from -L-1 to L-1

        displacement = displacement.astype(float)
        displacement = np.einsum("...i,ij->...j", displacement, self.basis_vectors)
        # displacement vector of offsets
        offset = self.site_offsets[:, None, :] - self.site_offsets[None, :, :]
        # total displacement vector
        displacement = displacement[..., None, None, :] + offset
        # distance
        dist_from_diff = np.linalg.norm(displacement, axis=-1)
        dist_from_diff = dist_from_diff[..., None]
        for axis, bc in enumerate(self.boundary):
            if bc != 0:
                indices = [0]
                indices += list(range(-self.shape[axis + 1] + 1, 0))
                indices += list(range(1, self.shape[axis + 1]))
                indices = np.asarray(indices)
                dist_pbc = dist_from_diff.take(indices, axis)
                dist_from_diff = np.concatenate([dist_from_diff, dist_pbc], axis=-1)
        dist_from_diff = np.min(dist_from_diff, axis=-1)
        dist = [self._index_to_dist(idx, dist_from_diff) for idx in range(self.nsites)]
        self._dist = np.stack(dist, axis=0)

    def _index_to_dist(self, index: int, dist_from_diff: np.ndarray) -> np.ndarray:
        """
        Calculates the distance of 'index' site to all other sites by slicing the
        'dist_from_diff' matrix.
        """
        xyz = self.xyz_from_index[index]
        dist_sliced = dist_from_diff[..., xyz[0]]
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
    ):
        """
        Plot the sites and neighbor bonds in the real space, with the adjusted color
        for lattice.

        Args:
            figsize: Figure size.
            markersize: Size of markers that represent the sites.
            color_in_cell: A list containing colors for different sites with the same
                offset in the unit cell. The length should be the same as the number of
                sites in a single unit cell.
            show_index: Whether to show index number at each site.
            index_fontsize: Fontsize if the index number is shown.
            neighbor_bonds: The n'th-nearest neighbor bonds to show. If is
                Sequence[int] then multiple neighbors. If don't want to show bonds then
                set this value to 0.
        Returns:
            A matplotlib.plt figure containing the geometrical information of lattice.
        """
        if color_in_cell is not None:
            color_site = color_in_cell
        else:
            color_site = [f"C{i}" for i in range(self.shape[0])]
        color_site = [color for color in color_site for _ in range(self.ncells)]
        return super().plot(
            figsize, markersize, color_site, show_index, index_fontsize, neighbor_bonds
        )
