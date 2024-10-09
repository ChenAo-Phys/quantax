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
        boundary: Union[int, Sequence[int]] = 1,
        is_fermion: bool = False,
    ):
        basis_vectors = np.eye(len(extent), dtype=np.float64)
        super().__init__(
            extent, basis_vectors, boundary=boundary, is_fermion=is_fermion
        )


def Chain(
    L: int,
    boundary: Union[int, Sequence[int]] = 1,
    is_fermion: bool = False,
):
    """1D chain lattice"""
    return Grid([L], boundary, is_fermion)


def Square(
    L: int,
    boundary: Union[int, Sequence[int]] = 1,
    is_fermion: bool = False,
):
    """2D square lattice"""
    return Grid([L, L], boundary, is_fermion)


def Cube(
    L: int,
    boundary: Union[int, Sequence[int]] = 1,
    is_fermion: bool = False,
):
    """3D cube lattice"""
    return Grid([L, L, L], boundary, is_fermion)


class Pyrochlore(Lattice):
    """
    Pyrochlore lattice with 4 atoms per unit cell
    """

    def __init__(
        self,
        extent: Union[int, Sequence[int]],
        boundary: Union[int, Sequence[int]] = 1,
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
        super().__init__(extent, basis_vectors, site_offsets, boundary, is_fermion)


class Triangular(Lattice):
    """2D triangular lattice"""

    def __init__(
        self,
        extent: Union[int, Sequence[int]],
        boundary: Union[int, Sequence[int]] = 1,
        is_fermion: bool = False,
    ):
        if isinstance(extent, int):
            extent = [extent] * 2
        basis_vectors = np.array([[1, 0], [0.5, np.sqrt(0.75)]])
        super().__init__(
            extent, basis_vectors, boundary=boundary, is_fermion=is_fermion
        )


class TriangularB(Lattice):
    """
    2D triangular lattice type B.
    See `PhysRevB.47.5861 <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.47.5861>`_
    Fig.1 N=12 as an example. N = 3 * extent ^ 2.
    """

    def __init__(
        self,
        extent: int,
        boundary: Union[int, Sequence[int]] = 1,
        is_fermion: bool = False,
    ):
        extent = [extent * 3, extent]
        basis_vectors = np.array([[1, 0], [1.5, np.sqrt(0.75)]])
        super().__init__(
            extent, basis_vectors, boundary=boundary, is_fermion=is_fermion
        )
