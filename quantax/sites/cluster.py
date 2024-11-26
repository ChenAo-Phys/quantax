from typing import Optional, Union, Sequence
import numpy as np
from .sites import Sites


class Cluster(Sites):
    """
    A system with sites connected to each other.
    """

    def __init__(
        self,
        n_coupled: int,
        n_decoupled: Optional[int] = 0, # total site will be n_coupled+n_decoupled
        is_fermion: bool = False,
    ):
        """
        A cluster structure on a single site with no periodicity. 
        The n_coupled defines the physical orbital number, which is half of the spin orbital (fermion) of the system. 
        The n_decoupled is the number of independent bath sites that only have interactions with coupled orbitals.

        Parameters
        ----------
        n_coupled : int
            the coupled orbital number in this cluster
        n_decoupled : int, optional
            the decoupled orbital number in this cluster
        is_fermion : bool, optional
            whether the system is composed of fermion, by default False

        Raises
        ------
        ValueError
            _description_
        """

        self.n_coupled = n_coupled
        self.n_decoupled = n_decoupled

        nsites = n_coupled + n_decoupled

        super().__init__(nsites, is_fermion)

    def get_neighbor(
        self, n_neighbor: Union[int, Sequence[int]] = 1, return_sign: bool = False
    ) -> np.ndarray:
        if (isinstance(n_neighbor, int) and n_neighbor != 1) or n_neighbor[0] != 1:
            raise ValueError(f"`Cluster` only contains the nearest neighbor coupling.")

        neighbors = []
        for i in range(self.n_coupled):
            for j in range(i + 1, self.nsites):
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
