from typing import Sequence
from numpy.typing import NDArray
import numpy as np
import jax
from .lattice import Lattice
from ..global_defs import PARTICLE_TYPE


class Grid(Lattice):
    """
    Grid lattice with basis vectors orthogonal to each other and only 1 site in each
    unit cell.
    """

    def __init__(
        self,
        extent: Sequence[int] | NDArray[np.integer],
        boundary: int | Sequence[int] | NDArray[np.integer] = 1,
        particle_type: PARTICLE_TYPE | str = PARTICLE_TYPE.spin,
        Nparticles: int | tuple[int, int] | None = None,
        double_occ: bool | None = None,
    ):
        basis_vectors = np.eye(len(extent), dtype=np.float64)
        super().__init__(
            extent, basis_vectors, None, boundary, particle_type, Nparticles, double_occ
        )


def Chain(
    L: int,
    boundary: int | Sequence[int] | NDArray[np.integer] = 1,
    particle_type: PARTICLE_TYPE | str = PARTICLE_TYPE.spin,
    Nparticles: int | tuple[int, int] | None = None,
    double_occ: bool | None = None,
):
    """1D chain lattice."""
    return Grid([L], boundary, particle_type, Nparticles, double_occ)


def Square(
    L: int,
    boundary: int | Sequence[int] | NDArray[np.integer] = 1,
    particle_type: PARTICLE_TYPE | str = PARTICLE_TYPE.spin,
    Nparticles: int | tuple[int, int] | None = None,
    double_occ: bool | None = None,
):
    """2D square lattice."""
    return Grid([L, L], boundary, particle_type, Nparticles, double_occ)


def Cube(
    L: int,
    boundary: int | Sequence[int] | NDArray[np.integer] = 1,
    particle_type: PARTICLE_TYPE | str = PARTICLE_TYPE.spin,
    Nparticles: int | tuple[int, int] | None = None,
    double_occ: bool | None = None,
):
    """3D cube lattice."""
    return Grid([L, L, L], boundary, particle_type, Nparticles, double_occ)


class Pyrochlore(Lattice):
    """
    Pyrochlore lattice with 4 atoms per unit cell.
    """

    def __init__(
        self,
        extent: int | Sequence[int] | NDArray[np.integer],
        boundary: int | Sequence[int] | NDArray[np.integer] = 1,
        particle_type: PARTICLE_TYPE | str = PARTICLE_TYPE.spin,
        Nparticles: int | tuple[int, int] | None = None,
        double_occ: bool | None = None,
    ):
        if isinstance(extent, int):
            extent = [extent] * 3
        if len(extent) != 3:
            raise ValueError("'extent' should contain 3 values.")
        h = 2 * np.sqrt(2.0 / 3.0)
        r = 2 * np.sqrt(1.0 / 3.0)
        basis_vectors = np.array(
            [
                [r * np.cos(0.0), r * np.sin(0.0), h],
                [r * np.cos(2 * np.pi / 3), r * np.sin(2 * np.pi / 3), h],
                [r * np.cos(4 * np.pi / 3), r * np.sin(4 * np.pi / 3), h],
            ]
        )
        origin = np.array([[0.0, 0.0, 0.0]])
        site_offsets = np.concatenate([origin, basis_vectors / 2], axis=0)
        super().__init__(
            extent,
            basis_vectors,
            site_offsets,
            boundary,
            particle_type,
            Nparticles,
            double_occ,
        )


class Triangular(Lattice):
    """2D triangular lattice."""

    def __init__(
        self,
        extent: int | Sequence[int] | NDArray[np.integer],
        boundary: int | Sequence[int] | NDArray[np.integer] = 1,
        particle_type: PARTICLE_TYPE | str = PARTICLE_TYPE.spin,
        Nparticles: int | tuple[int, int] | None = None,
        double_occ: bool | None = None,
    ):
        if isinstance(extent, int):
            extent = [extent] * 2
        basis_vectors = np.array([[1, 0], [0.5, np.sqrt(0.75)]])
        super().__init__(
            extent, basis_vectors, None, boundary, particle_type, Nparticles, double_occ
        )


class TriangularB(Lattice):
    r"""
    2D triangular lattice type B.
    See `PhysRevB.47.5861 <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.47.5861>`_
    Fig.1 N=12 as an example. The total number of particles is given by
    :math:`N = 3 \times \mathrm{L} ^ 2`.
    """

    def __init__(
        self,
        L: int,
        boundary: int | Sequence[int] | NDArray[np.integer] = 1,
        particle_type: PARTICLE_TYPE | str = PARTICLE_TYPE.spin,
        Nparticles: int | tuple[int, int] | None = None,
        double_occ: bool | None = None,
    ):
        extent = [L * 3, L]
        basis_vectors = np.array([[1, 0], [1.5, np.sqrt(0.75)]])
        super().__init__(
            extent, basis_vectors, None, boundary, particle_type, Nparticles, double_occ
        )

    def to_neighbor_repr(self, x: NDArray | jax.Array) -> NDArray | jax.Array:
        """
        Rearrange features to neighbor representations.
        """
        permutation = np.arange(self.Nsites, dtype=np.uint16)
        permutation = permutation.reshape(self.shape[1:])
        for i in range(permutation.shape[1]):
            permutation[:, i] = np.roll(permutation[:, i], shift=i)

        in_shape = x.shape
        x = x.reshape(-1, self.Nsites)
        x = x[..., permutation]
        return x.reshape(in_shape)

    def to_original_repr(self, x: NDArray | jax.Array) -> NDArray | jax.Array:
        """
        Rearrange neighbor representation of features back to original representation
        """
        permutation = np.arange(self.Nsites, dtype=np.uint16)
        permutation = permutation.reshape(self.shape[1:])
        for i in range(permutation.shape[1]):
            permutation[:, i] = np.roll(permutation[:, i], shift=-i)

        in_shape = x.shape
        x = x.reshape(-1, self.Nsites)
        x = x[:, permutation]
        return x.reshape(in_shape)
