from typing import Optional, Union, Sequence, Tuple
from matplotlib.figure import Figure
import numpy as np
from .sites import Sites


class Cluster(Sites):
    """
    A special kind of ``Sites`` with periodic structure in real space.
    """

    def __init__(
        self,
        n_orbitals: int,
        n_bath: int,
        is_fermion: bool = False,
        bath_independent: bool = False,
    ):
        """A cluster structure on a single site with no periodicity. The orbitals defines the physical orbital number,
        which is half of the spin orbital (fermion) of the system. The bath is the number of bath sites the cluster coupled with.
        The bath can be independent or kinetically connected with hopping.

        Parameters
        ----------
        n_orbitals : int
            the orbital number in this cluster
        n_bath : int
            the bath number in this cluster
        is_fermion : bool, optional
            whether the system is composed of fermion, by default False
        bath_independent : bool, optional
            is the bath orbitals independent with each other, by default False

        Raises
        ------
        ValueError
            _description_
        """

        self._shape = (n_orbitals+n_bath,)
        self.n_orbitals = n_orbitals
        self.n_bath = n_bath
        self.bath_independent = bath_independent

        nsites = np.prod(self._shape).item()

        super().__init__(nsites, is_fermion)

    @property
    def shape(self) -> np.ndarray:
        """
        Shape of the cluster. equals to n_orbitals+n_bath.
        """
        return self._shape

    def get_neighbor(self, n_neighbor: Union[int, Sequence[int]] = 0, return_sign: bool = False) -> np.ndarray:
        assert n_neighbor == 0 or n_neighbor == [0], "The cluster has no periodicity, thus n=0 indicate intra site orbital hopping."

        neighbours = []
        for i in range(self.n_orbitals+self.n_bath):
            for j in range(i, self.n_orbitals+self.n_bath):
                if i < self.n_orbitals or j == i:
                    neighbours.append((i, j))
                else:
                    if not self.bath_independent:
                        neighbours.append((i, j))
        
        if return_sign:
            return [np.array(neighbours)], [np.ones(len(neighbours), dtype=int)]
        
        return [np.array(neighbours)]



