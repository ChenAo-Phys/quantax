from __future__ import annotations
from typing import Optional, Union, Sequence, List
from warnings import warn
import numpy as np
import jax
import jax.numpy as jnp


class Sites:
    """
    Contains the information of both real space and Hilbert space. The
    geometrical coordinates of the sites and the Hilbert space dimension in each site
    are recorded.
    """

    _SITES = None

    def __init__(self, nsites: int, is_fermion: bool = False):
        """
        Constructs 'Sites' given number of sites and the hilbert space dimension at
        each site.

        Args:
            nsites: Total number of sites.
        """
        if nsites <= 0:
            raise ValueError("'nsites' should be positive.")
        self._nsites = nsites
        self._nstates = 2 * self._nsites if is_fermion else self._nsites
        self._neighbors: List[np.ndarray] = []
        if Sites._SITES is not None:
            warn("A second 'sites' is defined.")
        Sites._SITES = self
        self._is_fermion = is_fermion

    @property
    def nsites(self) -> int:
        """Total number of sites"""
        return self._nsites

    @property
    def nstates(self) -> int:
        """Total number of fock states, double for spinful fermions"""
        return self._nstates

    @property
    def dim(self) -> int:
        """Real space dimension"""
        return self._dim

    @property
    def is_fermion(self) -> bool:
        """Return True/False if self is a fermionic/spin system"""
        return self._is_fermion

    @property
    def coord(self) -> np.ndarray:
        """Real space coordinates of all sites in order"""
        return self._coord

    @coord.setter
    def coord(self, coord: Sequence[float]) -> None:
        """Set real space coordinates of sites"""
        coord_np = np.array(coord, dtype=float)
        if coord_np.ndim != 2:
            raise ValueError("'coord' should be a 2D-array.")
        if coord_np.shape[0] != self._nsites:
            raise ValueError("'coord' size doesn't match number of sites.")
        self._dim = coord_np.shape[1]
        self._coord = coord_np

    @property
    def dist(self) -> np.ndarray:
        """
        Matrix of the real space distance between all site pairs.
        Example: dist[2, 3] is the distance between site 2 and 3.
        """
        if not hasattr(self, "_dist"):
            self._compute_dist()
        return self._dist

    def _compute_dist(self) -> None:
        """Computes distance between sites"""
        coord1 = self.coord[None, :, :]
        coord2 = self.coord[:, None, :]
        self._dist = np.linalg.norm(coord1 - coord2, axis=2)

    def get_neighbor(
        self, n_neighbor: Union[int, Sequence[int]] = 1, del_dist: bool = False
    ) -> Union[jax.Array, Sequence]:
        """
        Gets n'th-nearest neighbor site pairs with the distance given by 'self._dist'.
        If 'self._dist' doesn't exist it will be calculated.

        Args:
            n_neighbor: The n'th-nearest neighbor to obtain. If array-like then
                multiple neighbors will be returned in the same order.
                'n_neighbor == 1' represents nearest neighbor.
            del_dist: Whether to delete 'self._dist' after obtaining the neighbors.
        Returns:
            If n_neighbor is int, then a 2-dimensional jax.numpy array with each row
            a pair of neighbor site index.
            If n_neighbor is Sequence[int], then a list containing corresponding to all
            n_neighbor values.
        """
        if not hasattr(self, "_dist"):
            self._compute_dist()
        max_neighbor = n_neighbor if isinstance(n_neighbor, int) else max(n_neighbor)
        if len(self._neighbors) < max_neighbor:
            self._compute_neighbor(max_neighbor)
        if hasattr(self, "_dist") and del_dist:
            del self._dist
        if isinstance(n_neighbor, int):
            return jnp.asarray(self._neighbors[n_neighbor - 1])
        else:
            return [jnp.asarray(self._neighbors[n - 1]) for n in n_neighbor]

    def _compute_neighbor(self, max_neighbor: int = 1) -> None:
        """Calculates all n'th-nearest neighbor with n < max_neighbor"""
        tol = 1e-6
        if self._neighbors:
            sitei, sitej = self._neighbors[-1][0]
            min_dist = self._dist[sitei, sitej] * (1 + tol)
            min_neighbor = len(self._neighbors) + 1
        else:
            self._neighbors = []
            min_dist = tol
            min_neighbor = 1
        for _ in range(min_neighbor, max_neighbor + 1):
            min_dist = np.min(self._dist[self._dist > min_dist])
            neighbors = np.argwhere(np.abs((self._dist - min_dist) / min_dist) < tol)
            neighbors = neighbors[neighbors[:, 0] < neighbors[:, 1]]
            self._neighbors.append(neighbors)
            min_dist *= 1 + tol

    def plot(
        self,
        figsize: Sequence[Union[int, float]] = (10, 10),
        markersize: Optional[Union[int, float]] = None,
        color: Union[str, Sequence[str]] = "C0",
        show_index: bool = True,
        index_fontsize: Optional[Union[int, float]] = None,
        neighbor_bonds: Union[int, Sequence[int]] = 1,
    ):
        """
        Plot the sites and neighbor bonds in the real space.

        Args:
            figsize: Figure size.
            markersize: Size of markers that represent the sites.
            color: Color of sites in the figure.
            show_index: Whether to show index number at each site.
            index_fontsize: Fontsize if the index number is shown.
            neighbor_bonds: The n'th-nearest neighbor bonds to show. If is
                Sequence[int] then multiple neighbors. If don't want to show bonds then
                set this value to 0.
        Returns:
            A matplotlib.plt figure containing the geometrical information of sites.
        """
        # pylint: disable=import-outside-toplevel
        if self._dim > 3:
            raise NotImplementedError("'Sites' can only plot for dimension <= 3.")
        if self._dim == 3:
            from mpl_toolkits.mplot3d import Axes3D  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore

        fig = plt.figure(figsize=figsize)
        axes = fig.add_subplot() if self._dim < 3 else Axes3D(fig)
        figsize = fig.get_size_inches()

        def coord_for_print(coord: np.ndarray) -> np.ndarray:
            if self._dim == 1:
                y_dim1 = np.zeros_like(coord)
                coord = np.concatenate([coord, y_dim1], axis=-1)
            return coord.transpose()

        # scatter
        if markersize is None:
            markersize = figsize[0] * figsize[1] * 0.8
            if self._dim == 3:
                markersize /= 4
        axes.scatter(
            *coord_for_print(self._coord), s=markersize, c=color, alpha=1, zorder=2
        )

        # neighbor bonds
        # neighbors connected through boundary conditions are not shown
        neighbors = self.get_neighbor(neighbor_bonds)
        neighbors_list: List[np.ndarray] = []
        if isinstance(neighbors, np.ndarray):
            neighbors_list = [neighbors]
        elif isinstance(neighbors, list):
            neighbors_list = neighbors
        for i, neighbor in enumerate(neighbors_list):
            color = f"C{5 + i}"
            for pair_site in neighbor:
                coord = self._coord[pair_site]
                # judge whether connected through boundaries
                dist_boundary = self._dist[pair_site[0], pair_site[1]]
                dist_no_boundary = np.linalg.norm(coord[0] - coord[1])
                if np.abs(dist_no_boundary - dist_boundary) / dist_boundary < 1e-6:
                    axes.plot(*coord_for_print(coord), c=color, zorder=0)

        # index
        if show_index:
            if index_fontsize is None:
                index_fontsize = 2 * np.sqrt(figsize[0] * figsize[1])
                if self._dim == 3:
                    index_fontsize /= 2
            for index, coord in enumerate(self._coord):
                axes.text(
                    *coord_for_print(coord), index, fontsize=index_fontsize, zorder=1
                )

        return fig
