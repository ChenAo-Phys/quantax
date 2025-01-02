from typing import Optional, Union, Sequence, Tuple
import numpy as np
from .sites import Sites


class Cluster(Sites):
    """
    A system with sites connected to each other.
    """

    def __init__(
        self,
        n_coupled: int,
        n_decoupled: Optional[int] = 0,  # total site will be n_coupled+n_decoupled
        Nparticle: Union[None, int, Tuple[int, int]] = None,
        is_fermion: bool = False,
        double_occ: Optional[bool] = None,
    ):
        """
        A cluster structure on a single site with no periodicity.
        The n_coupled defines the physical orbital number, which is half of the spin orbital (fermion) of the system.
        The n_decoupled is the number of independent bath sites that only have interactions with coupled orbitals.

        Parameters
        ----------
        :param double_occ: int
            the coupled orbital number in this cluster
        :param double_occ: int, optional
            the decoupled orbital number in this cluster
        :param double_occ: bool, optional
            whether the system is composed of fermion, by default False
        :param double_occ: Whether double occupancy is allowed. Default to False
            for spin systems and True for fermion systems.

        Raises
        ------
        ValueError
            _description_
        """

        self.n_coupled = n_coupled
        self.n_decoupled = n_decoupled

        N = n_coupled + n_decoupled

        super().__init__(N, Nparticle, is_fermion, double_occ)

    def get_neighbor(
        self, n_neighbor: Union[int, Sequence[int]] = 1, return_sign: bool = False
    ) -> np.ndarray:
        if (isinstance(n_neighbor, int) and n_neighbor != 1) or n_neighbor[0] != 1:
            raise ValueError(f"`Cluster` only contains the nearest neighbor coupling.")

        neighbors = []
        for i in range(self.n_coupled):
            for j in range(i + 1, self.N):
                neighbors.append((i, j))
        neighbors = np.asarray(neighbors)

        if isinstance(n_neighbor, int):
            if return_sign:
                return neighbors, np.ones_like(neighbors)
            else:
                return neighbors
        else:
            if return_sign:
                return [neighbors], [np.ones_like(neighbors)]
            else:
                return [neighbors]
