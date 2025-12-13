from typing import Optional, Union, Sequence, Tuple
import numpy as np
from .sites import Sites
from ..global_defs import PARTICLE_TYPE


class Cluster(Sites):
    """
    A system with sites connected to each other.
    """

    def __init__(
        self,
        n_coupled: int,
        n_decoupled: Optional[int] = 0,  # total site will be n_coupled+n_decoupled
        particle_type: Union[PARTICLE_TYPE, str] = PARTICLE_TYPE.spin,
        Nparticles: Union[None, int, Tuple[int, int]] = None,
        double_occ: Optional[bool] = None,
    ):
        """
        A cluster structure on a single site with no periodicity.
        The n_coupled defines the physical orbital number, which is half of the spin orbital (fermion) of the system.
        The n_decoupled is the number of independent bath sites that only have interactions with coupled orbitals.

        Parameters
        ----------
        :param n_coupled: int
            the coupled orbital number in this cluster
        :param n_decoupled: int, optional
            the decoupled orbital number in this cluster
        :param particle_type: The particle type of the system, including spin,
            spinful fermion, or spinless fermion.
        :param Nparticles: The number of particles in the system.
            If unspecified, the number of particles is non-conserved.
            If specified, use an int to specify the total particle number, or use a tuple
            (n_up, n_down) to specify the number of spin-up and spin-down particles.
        :param double_occ: Whether double occupancy is allowed. Default to False
            for spin systems and True for fermion systems.

        Raises
        ------
        ValueError
            _description_
        """

        self.n_coupled = n_coupled
        self.n_decoupled = n_decoupled

        Nsites = n_coupled + n_decoupled

        super().__init__(Nsites, particle_type, Nparticles, double_occ)

    def get_neighbor(
        self, n_neighbor: Union[int, Sequence[int]] = 1, return_sign: bool = False
    ) -> np.ndarray:
        if (isinstance(n_neighbor, int) and n_neighbor != 1) or n_neighbor[0] != 1:
            raise ValueError(f"`Cluster` only contains the nearest neighbor coupling.")

        neighbors = []
        for i in range(self.n_coupled):
            for j in range(i + 1, self.Nsites):
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
