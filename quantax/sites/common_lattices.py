from typing import Union, Sequence
import numpy as np
from .lattice import Lattice


class Grid(Lattice):
    """
    Grid lattice with basis vectors orthogonal to each other and only 1 site in each
    unit cell.
    """

    def __init__(
        self,
        extent: Sequence[int],
        pbc: Union[bool, Sequence[bool]] = True,
        is_fermion: bool = False,
    ):
        """
        Constructs grid lattice.

        Args:
            extent: The number of unit cell copies in each spatial dimension.
            local_hilbert_dim: The Hilbert space dimension in each site. If int then
                all sites have the same dimension. If Sequence[int] then each entry
                represents the dimension at a site.
            pbc: Whether using periodic boundary condition.
                If boolean then all dimensions use the same boundary condition.
                If Sequence[bool] different boundary conditions are applied to
                different dimensions.
        """
        basis_vectors = np.eye(len(extent), dtype=np.float_)
        super().__init__(extent, basis_vectors, pbc=pbc, is_fermion=is_fermion)


def Chain(L: int, pbc: Union[bool, Sequence[bool]] = True):
    """1D chain lattice"""
    return Grid([L], pbc)


def Square(L: int, pbc: Union[bool, Sequence[bool]] = True):
    """2D square lattice"""
    return Grid([L, L], pbc)


def Cube(L: int, pbc: Union[bool, Sequence[bool]] = True):
    """3D cube lattice"""
    return Grid([L, L, L], pbc)


class Pyrochlore(Lattice):
    """
    Pyrochlore lattice. 'extent' is the number of unit cell copies in each direction.
    If 'extent' is int then the copy number is the same for 3 directions.
    """

    def __init__(
        self,
        extent: Union[int, Sequence[int]],
        pbc: Union[bool, Sequence[bool]] = True,
        is_fermion: bool = False,
    ):
        if isinstance(extent, int):
            extent = [extent] * 3
        if len(extent) != 3:
            raise ValueError("'extent' should contain 3 values.")
        h = 2 * np.sqrt(2.0 / 3.0)  # pylint: disable=invalid-name
        r = 2 * np.sqrt(1.0 / 3.0)  # pylint: disable=invalid-name
        basis_vectors = np.array(
            [
                [r * np.cos(0.0), r * np.sin(0.0), h],
                [r * np.cos(2 * np.pi / 3), r * np.sin(2 * np.pi / 3), h],
                [r * np.cos(4 * np.pi / 3), r * np.sin(4 * np.pi / 3), h],
            ]
        )
        origin = np.array([[0.0, 0.0, 0.0]])
        site_offsets = np.concatenate([origin, basis_vectors / 2], axis=0)
        super().__init__(extent, basis_vectors, site_offsets, pbc, is_fermion)


class Triangular(Lattice):
    """
    2D triangular lattice.
    """

    def __init__(
        self,
        extent: Union[int, Sequence[int]],
        pbc: Union[bool, Sequence[bool]] = True,
        is_fermion: bool = False,
    ):
        if isinstance(extent, int):
            extent = [extent] * 2
        basis_vectors = np.array([[1, 0], [0.5, np.sqrt(0.75)]])
        super().__init__(extent, basis_vectors, pbc=pbc, is_fermion=is_fermion)


class TriangularB(Lattice):
    """
    2D triangular lattice type B.
    See PhysRevB.47.5861 Fig.1 N=12 as an example. In general, N = 3 * extent **2
    """

    def __init__(
        self,
        extent: int,
        pbc: Union[bool, Sequence[bool]] = True,
        is_fermion: bool = False,
    ):
        extent = [extent * 3, extent]
        basis_vectors = np.array([[1, 0], [1.5, np.sqrt(0.75)]])
        super().__init__(extent, basis_vectors, pbc=pbc, is_fermion=is_fermion)
