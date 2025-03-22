from __future__ import annotations
from typing import Optional, Union, Sequence, Tuple, List
from matplotlib.figure import Figure

from warnings import warn
import numpy as np
import matplotlib.pyplot as plt


class Sites:
    """
    A collection of multiple spins or fermions that make up the quantum system.
    """

    _SITES = None

    def __init__(
        self,
        N: int,
        Nparticle: Union[None, int, Tuple[int, int]] = None,
        is_fermion: bool = False,
        double_occ: Optional[bool] = None,
        coord: Optional[np.ndarray] = None,
    ):
        """
        :param N: The number of sites in the system.
        :param Nparticle: The number of particles in the system, given by a tuple of
            (n_up, n_down).
        :param is_fermion: Whether the system is made of fermions or spins. Default to
            False (spins).
        :param double_occ: Whether double occupancy is allowed. Default to False
            for spin systems and True for fermion systems.
        :param coord: The coordinates of sites. This doesn't have to be specified if
            the spatial information is unnecessary.
        """
        if Sites._SITES is not None:
            warn("A second 'sites' is defined.")
        Sites._SITES = self

        self._N = N

        if Nparticle is None:
            if not is_fermion:
                Nparticle = N
        elif isinstance(Nparticle, int):
            if Nparticle != N and not is_fermion:
                raise ValueError(
                    "Specify spin conservation with an integer is ambiguous. "
                    "Please use a tuple (Nup, Ndown)."
                )
        else:
            Nparticle = tuple(Nparticle)
        self._Nparticle = Nparticle

        self._is_fermion = is_fermion

        if double_occ is None:
            self._double_occ = is_fermion
        else:
            if double_occ and not is_fermion:
                raise ValueError("Spin systems don't support double occupacy.")
            self._double_occ = double_occ

        if coord is not None:
            self._coord = np.asarray(coord, dtype=float)
        else:
            self._coord = None
        self._dist = None
        self._sign = None
        self._neighbors = []

    @property
    def N(self) -> int:
        """The number of sites"""
        return self._N

    @property
    def nstates(self) -> int:
        """
        The number of qubits, which should be ``N`` for spins
        and ``2 * N`` for spinful fermions.
        """
        return 2 * self._N if self._is_fermion else self._N

    @property
    def Nparticle(self) -> Union[None, int, Tuple[int, int]]:
        """
        The number of particles.
        None: No particle conservation
        int: Conservation of total particle number
        Tuple[int, int]: Conservation of Nup and Ndown
        """
        return self._Nparticle

    @property
    def Ntotal(self) -> Optional[int]:
        """The total number of particles."""
        if self.Nparticle is None:
            return None
        elif isinstance(self.Nparticle, int):
            return self.Nparticle
        else:
            return sum(self.Nparticle)

    @property
    def ndim(self) -> int:
        """The number of spatial dimensions, e.g. 2 for square lattice and 3 for cubic."""
        if self._coord is None:
            raise RuntimeError(
                "The number of dimension is unknown because the coordinates are "
                "unavailable."
            )
        return self._coord.shape[1]

    @property
    def double_occ(self) -> bool:
        """Whether the system allows double occupancy"""
        return self._double_occ

    @property
    def is_fermion(self) -> bool:
        """Whether the system is made of fermions"""
        return self._is_fermion

    @property
    def coord(self) -> np.ndarray:
        """Real space coordinates of all sites"""
        if self._coord is None:
            raise RuntimeError("The coordinates are unavailable.")
        return self._coord

    @property
    def dist(self) -> np.ndarray:
        """
        Matrix of the real space distance between all site pairs.

        .. tip:: ``dist[2, 3]`` is the distance between site 2 and 3.
        """
        if self._dist is None:
            self._dist, self._sign = self._get_dist_sign()
        return self._dist

    @property
    def sign(self) -> np.ndarray:
        """
        Matrix of the sign between all site pairs, which is non-trivial only for
        fermionic systems with anti-periodic boundary conditions.

        .. tip:: ``sign[2, 3]`` is the sign of the bond connecting site 2 and 3.
        """
        if self._sign is None:
            self._dist, self._sign = self._get_dist_sign()
        return self._sign

    def _get_dist_sign(self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes distance between sites"""
        coord1 = self.coord[None, :, :]
        coord2 = self.coord[:, None, :]
        dist = np.linalg.norm(coord1 - coord2, axis=2)
        sign = np.ones_like(dist, dtype=int)
        return dist, sign

    def get_neighbor(
        self, n_neighbor: Union[int, Sequence[int]] = 1, return_sign: bool = False
    ) -> Union[np.ndarray, Sequence[np.ndarray], tuple]:
        """
        Gets n'th-nearest neighbor site pairs.

        :param n_neighbor:
            The n'th-nearest neighbor to obtain. The nearest neighbor is given by 1.
            If it's a sequence, then multiple neighbors will be returned in the same order.

        :param return_sign:
            Whether this function should also return the sign of neighbor bonds.
            The sign is non-trivial only for fermionic systems with anti-periodic
            boundary conditions.

        :return:
            neighbor
                If ``n_neighbor`` is int, then a 2D numpy array with each row a pair of
                neighbor site indeces.
                If ``n_neighbor`` is sequence, then a list with each item a 2D
                numpy array corresponding to ``n_neighbor`` items.

            sign
                The sign of neighbor bonds. Only provided if ``return_sign`` is True.
        """
        max_neighbor = n_neighbor if isinstance(n_neighbor, int) else max(n_neighbor)
        if len(self._neighbors) < max_neighbor:
            self._compute_neighbor(max_neighbor)
        if isinstance(n_neighbor, int):
            neighbor = self._neighbors[n_neighbor - 1]
            if return_sign:
                sign = self.sign[neighbor[:, 0], neighbor[:, 1]]
                return neighbor, sign
            else:
                return neighbor
        else:
            neighbor = [self._neighbors[n - 1] for n in n_neighbor]
            if return_sign:
                sign = [self.sign[nb[:, 0], nb[:, 1]] for nb in neighbor]
                return neighbor, sign
            else:
                return neighbor

    def _compute_neighbor(self, max_neighbor: int = 1) -> None:
        """Calculates all n'th-nearest neighbor with n < max_neighbor"""
        tol = 1e-6
        if self._neighbors:
            sitei, sitej = self._neighbors[-1][0]
            min_dist = self.dist[sitei, sitej] * (1 + tol)
            min_neighbor = len(self._neighbors) + 1
        else:
            self._neighbors = []
            min_dist = tol
            min_neighbor = 1
        for _ in range(min_neighbor, max_neighbor + 1):
            min_dist = np.min(self.dist[self.dist > min_dist])
            neighbors = np.argwhere(np.abs((self.dist - min_dist) / min_dist) < tol)
            neighbors = neighbors[neighbors[:, 0] < neighbors[:, 1]]  # i < j
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
    ) -> Figure:
        """
        Plot the sites and neighbor bonds in the real space.

        :param figsize: Figure size.
        :param markersize: Size of markers that represent the sites.
        :param color: Color of sites in the figure.
        :param show_index: Whether to show index number at each site.
        :param index_fontsize: Fontsize if the index number is shown.
        :param neighbor_bonds: The n'th-nearest neighbor bonds to show.
            Set this value to 0 to hide all bonds.

        :return:
            A matplotlib figure containing the geometrical plot of sites.
        """
        # pylint: disable=import-outside-toplevel
        if self.ndim > 3:
            raise NotImplementedError("'Sites' can only plot for dimension <= 3.")
        if self.ndim == 3:
            from mpl_toolkits.mplot3d import Axes3D  # type: ignore

        fig = plt.figure(figsize=figsize)
        axes = fig.add_subplot() if self.ndim < 3 else Axes3D(fig)
        figsize = fig.get_size_inches()

        def coord_for_print(coord: np.ndarray) -> np.ndarray:
            if self.ndim == 1:
                y_dim1 = np.zeros_like(coord)
                coord = np.concatenate([coord, y_dim1], axis=-1)
            return coord.transpose()

        # scatter
        if markersize is None:
            markersize = figsize[0] * figsize[1] * 0.8
            if self.ndim == 3:
                markersize /= 4
        axes.scatter(
            *coord_for_print(self.coord), s=markersize, c=color, alpha=1, zorder=2
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
                coord = self.coord[pair_site]
                # judge whether connected through boundaries
                dist_boundary = self.dist[pair_site[0], pair_site[1]]
                dist_no_boundary = np.linalg.norm(coord[0] - coord[1])
                if np.abs(dist_no_boundary - dist_boundary) / dist_boundary < 1e-6:
                    axes.plot(*coord_for_print(coord), c=color, zorder=0)

        # index
        if show_index:
            if index_fontsize is None:
                index_fontsize = 2 * np.sqrt(figsize[0] * figsize[1])
                if self.ndim == 3:
                    index_fontsize /= 2
            for index, coord in enumerate(self.coord):
                axes.text(
                    *coord_for_print(coord), index, fontsize=index_fontsize, zorder=1
                )

        return fig
