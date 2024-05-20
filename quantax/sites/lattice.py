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
        pbc: Union[bool, Sequence[bool]] = True,
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
            pbc: Whether using periodic boundary condition.
                If boolean then all dimensions use the same boundary condition.
                If Sequence[bool] different boundary conditions are applied to
                different dimensions.
        """
        self._check_basis(extent, basis_vectors, site_offsets)
        self._check_pbc(pbc)
        super().__init__(np.prod(self._shape), is_fermion)
        self._create_converting_matrix()
        self._compute_coord()

    def _check_basis(self, extent, basis_vectors, site_offsets):
        """Checks the input validity of 'extent', 'basis_vectors' and 'site_offsets'"""
        extent = np.array(extent, dtype=int)
        if extent.ndim != 1:
            raise ValueError("'extent' should be a 1D array.")
        if extent[extent <= 0].size:
            raise ValueError("'extent' should be positive.")

        basis_vectors = np.array(basis_vectors, dtype=float)
        if basis_vectors.ndim != 2:
            raise ValueError("'basis_vector' should be a 2D array")
        if extent.size != basis_vectors.shape[0]:
            raise ValueError("'basis_vectors' number doesn't match extent.")
        self._basis_vectors = basis_vectors

        if site_offsets is None:
            site_offsets = np.zeros([1, basis_vectors.shape[1]], dtype=float)
        else:
            site_offsets = np.array(site_offsets, dtype=float)
            if site_offsets.ndim != 2:
                raise ValueError("'site_offsets' should be a 2D array")
            if site_offsets.shape[1] != basis_vectors.shape[1]:
                raise ValueError(
                    "Vectors in 'basis_vectors' and 'site_offsets' should have the same"
                    "dimension."
                )
        self._site_offsets = site_offsets
        self._shape = tuple(extent) + (site_offsets.shape[0],)

    def _check_pbc(self, pbc):
        """Checks the input validity of pbc"""
        spatial_dim = len(self._shape) - 1
        if isinstance(pbc, bool):
            pbc = np.full(spatial_dim, pbc, dtype=bool)
        else:
            pbc = np.array(pbc, dtype=bool)
            if pbc.ndim != 1:
                raise ValueError("'pbc' should be a 1D-array.")
            if pbc.size != spatial_dim:
                raise ValueError("'pbc' size doesn't match extent.")
        self._pbc = pbc

    def _create_converting_matrix(self):
        """Creates the converting matrix between xyz and index"""
        index = np.arange(self._nsites, dtype=np.int_)
        xyz = []
        for i in range(len(self._shape)):
            num_later = np.prod(self._shape[i + 1 :], dtype=np.int_)
            xyz.append(index // num_later % self._shape[i])
        self._index_from_xyz = index.reshape(self._shape)
        self._xyz_from_index = np.stack(xyz, axis=1)

    def _compute_coord(self):
        """Computes the spatial coordinates for different sites"""
        coord = np.zeros(self._basis_vectors.shape[1])
        for i_basis in range(len(self._shape) - 1):
            grid = np.arange(self._shape[i_basis], dtype=float)
            grid = np.einsum("i,j->ji", self._basis_vectors[i_basis], grid)
            coord = np.expand_dims(coord, -2)
            coord = coord + grid
        coord = np.expand_dims(coord, -2)
        coord = coord + self._site_offsets
        self.coord = coord.reshape(-1, self._basis_vectors.shape[1])

    @property
    def shape(self) -> np.ndarray:
        """
        Shape of the lattice. The first several elements are the extent and the last
        one is the atom number in a unit cell.
        """
        return self._shape

    @property
    def basis_vectors(self) -> np.ndarray:
        """Basis vectors of the lattice"""
        return self._basis_vectors

    @property
    def site_offsets(self) -> np.ndarray:
        """Site offsets in a unit cell"""
        return self._site_offsets

    @property
    def pbc(self) -> np.ndarray:
        """Whether the periodic boundary condition is used in different directions"""
        return self._pbc

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
        displacement_shape = np.append(self._shape[:-1], len(self._shape) - 1)
        displacement = self._xyz_from_index[:: self._shape[-1], :-1]
        displacement = displacement.reshape(displacement_shape)
        for axis in range(len(self._shape) - 1):
            flip = displacement.take(np.arange(self._shape[axis] - 1, 0, -1), axis)
            flip[..., axis] *= -1
            displacement = np.concatenate([displacement, flip], axis)
        # now displacement[x, y, z] = [x, y, z] for x, y, z from -L-1 to L-1

        displacement = displacement.astype(float)
        displacement = np.einsum("...i,ij->...j", displacement, self._basis_vectors)
        # displacement vector of offsets
        offset = (
            self._site_offsets[:, np.newaxis, :] - self._site_offsets[np.newaxis, :, :]
        )
        # total displacement vector
        displacement = displacement[..., np.newaxis, np.newaxis, :] + offset
        # distance
        dist_from_diff = np.linalg.norm(displacement, axis=-1)
        dist_from_diff = dist_from_diff[..., np.newaxis]
        for axis, pbc in enumerate(self._pbc):
            if pbc:
                indices = (
                    [0]
                    + list(range(-self._shape[axis] + 1, 0))
                    + list(range(1, self._shape[axis]))
                )
                indices = np.array(indices)
                dist_pbc = dist_from_diff.take(indices, axis)
                dist_from_diff = np.concatenate([dist_from_diff, dist_pbc], axis=-1)
        dist_from_diff = np.min(dist_from_diff, axis=-1)
        dist_list = [
            self._index_to_dist(index, dist_from_diff) for index in range(self._nsites)
        ]
        self._dist = np.stack(dist_list, axis=0)

    def _index_to_dist(self, index: int, dist_from_diff: np.ndarray) -> np.ndarray:
        """
        Calculates the distance of 'index' site to all other sites by slicing the
        'dist_from_diff' matrix.
        """
        xyz = self._xyz_from_index[index]
        dist_sliced = dist_from_diff[..., xyz[-1]]
        for axis, coord in enumerate(xyz[:-1]):
            slices = [np.arange(-coord, 0), np.arange(self._shape[axis] - coord)]
            slices = np.concatenate(slices)
            dist_sliced = dist_sliced.take(slices, axis)
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
            color_site = [f"C{i}" for i in range(self._shape[-1])]
        color_site = color_site * np.prod(self._shape[:-1])
        return super().plot(
            figsize, markersize, color_site, show_index, index_fontsize, neighbor_bonds
        )
