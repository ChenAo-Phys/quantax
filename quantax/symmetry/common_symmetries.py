"""
docstring here
"""

from typing import Union, Sequence, Tuple
import numpy as np
from .symmetry import Symmetry
from ..global_defs import get_lattice


_Identity = None
_TotalSz = dict()
_SpinInverse = dict()


def Identity() -> Symmetry:
    global _Identity
    if _Identity is None:
        _Identity = Symmetry()
    return _Identity


def TotalSz(total_sz: int = 0) -> Symmetry:
    global _TotalSz
    if total_sz not in _TotalSz:
        _TotalSz[total_sz] = Symmetry(total_sz=total_sz)
    return _TotalSz[total_sz]


def SpinInverse(eigval: int = 1) -> Symmetry:
    if eigval not in (1, -1):
        raise ValueError("'eigval' of spin inversion should be 1 or -1.")

    global _SpinInverse
    if eigval not in _SpinInverse:
        _SpinInverse[eigval] = Symmetry(spin_inversion=eigval)
    return _SpinInverse[eigval]


def Translation(vector: Sequence, sector: int = 0) -> Symmetry:
    lattice = get_lattice()
    vector = np.asarray(vector, dtype=np.int64)
    xyz = lattice.xyz_from_index.copy()
    for axis, stride in enumerate(vector):
        if stride:
            if not lattice.pbc[axis]:
                raise ValueError(
                    f"'lattice' doesn't have perioidic boundary in axis {axis}"
                )
            xyz[:, axis] = (xyz[:, axis] + stride) % lattice.shape[axis]
    xyz_tuple = tuple(tuple(row) for row in xyz.T)
    generator = lattice.index_from_xyz[xyz_tuple]
    return Symmetry(generator, sector)


def TransND(sector: Union[int, Tuple[int, ...]] = 0) -> Symmetry:
    dim = get_lattice().dim
    if isinstance(sector, int):
        sector = [sector] * dim
    vector = np.identity(dim)
    symm_list = [Translation(vec, sec) for vec, sec in zip(vector, sector)]
    symm = sum(symm_list, start=Identity())
    return symm


def Trans1D(sector: int = 0) -> Symmetry:
    return Translation([1], sector)


def Trans2D(sector: Union[int, Tuple[int, int]] = 0) -> Symmetry:
    if isinstance(sector, int):
        sector = [sector, sector]
    return Translation([1, 0], sector[0]) + Translation([0, 1], sector[1])


def Trans3D(sector: Union[int, Tuple[int, int, int]] = 0) -> Symmetry:
    if isinstance(sector, int):
        sector = [sector, sector, sector]
    return (
        Translation([1, 0, 0], sector[0])
        + Translation([0, 1, 0], sector[1])
        + Translation([0, 0, 1], sector[2])
    )


'''
def Flip(axis: Union[int, Sequence] = 0, sector: int = 0) -> Symmetry:
    """Flip performed on specified axis"""
    lattice = get_lattice()
    index = np.arange(lattice.nsites).reshape(lattice.shape)
    generator = np.flip(index, axis).flatten()
    return Symmetry(generator, sector)


def RotGrid(axes: Sequence = (0, 1), sector: int = 0) -> Symmetry:
    """Rotation of specified axes for grid lattice"""
    lattice = get_lattice()
    if not isinstance(lattice, Grid):
        raise ValueError("RotGrid symmetry only works for Grid lattice")
    if lattice.shape[axes[0]] != lattice.shape[axes[1]]:
        raise ValueError("RotGrid symmetry can only rotate axes with the same length")
    if lattice.pbc[axes[0]] != lattice.pbc[axes[1]]:
        raise ValueError("RotGrid symmetry can only rotate axes with the same boundary")
    index = np.arange(lattice.nsites).reshape(lattice.shape)
    generator = np.rot90(index, axes=axes).flatten()
    return Symmetry(generator, sector)
'''


def LinearTransform(matrix: np.ndarray, sector: int = 0) -> Symmetry:
    """
    The symmetry applies linear transformation to the lattice
    """
    tol = 1e-6
    lattice = get_lattice()

    coord = lattice.coord
    center = np.mean(coord, axis=0)
    new_coord = np.einsum("ij,nj->ni", matrix, coord)
    basis = lattice.basis_vectors.T
    new_xyz = np.linalg.solve(basis, new_coord.T).T  # dimension: ni
    offsets_xyz = np.linalg.solve(basis, lattice.site_offsets.T).T  # oi

    # site n, offset o, coord i
    new_xyz = new_xyz[:, None, :] - offsets_xyz[None, :, :]
    correct_offsets = np.abs(np.round(new_xyz) - new_xyz) < tol
    correct_offsets = np.all(correct_offsets, axis=2)
    offsets_idx = np.nonzero(correct_offsets)[1]
    new_xyz = np.rint(new_xyz[correct_offsets]).astype(np.int64)

    shape = np.array(lattice.shape[:-1], dtype=np.int64)[None, :]
    shift = new_xyz // shape
    new_xyz = new_xyz - shift * shape

    slicing = tuple(item for item in new_xyz.T) + (offsets_idx,)
    generator = lattice.index_from_xyz[slicing]
    return Symmetry(generator, sector)


def Flip(axis: Union[int, Sequence] = 0, sector: int = 0) -> Symmetry:
    matrix = np.ones(get_lattice().dim)
    matrix[np.asarray(axis)] = -1
    matrix = np.diag(matrix)
    return LinearTransform(matrix, sector)


def Rotation(angle: float, axes: Sequence = (0, 1), sector: int = 0) -> Symmetry:
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    matrix = np.eye(get_lattice().dim)
    x, y = axes
    matrix[x, x] = cos_theta
    matrix[x, y] = -sin_theta
    matrix[y, x] = sin_theta
    matrix[y, y] = cos_theta
    return LinearTransform(matrix, sector)


def C4v(repr: str = "A1") -> Symmetry:
    if repr == "A1":
        return Rotation(angle=np.pi / 2, sector=0) + Flip(sector=0)
    if repr == "A2":
        return Rotation(angle=np.pi / 2, sector=0) + Flip(sector=1)
    if repr == "B1":
        return Rotation(angle=np.pi / 2, sector=2) + Flip(sector=0)
    if repr == "B2":
        return Rotation(angle=np.pi / 2, sector=2) + Flip(sector=1)
    if repr == "E":
        return Rotation(angle=np.pi, sector=[2, -2])
    raise ValueError(
        "'repr' should be one of the following: 'A1', 'A2', 'B1', 'B2' or 'E'"
    )


def D6(repr: str = "A1") -> Symmetry:
    if repr == "A1":
        return Rotation(angle=np.pi / 3, sector=0) + Flip(sector=0)
    if repr == "A2":
        return Rotation(angle=np.pi / 3, sector=0) + Flip(sector=1)
    if repr == "B1":
        return Rotation(angle=np.pi / 3, sector=3) + Flip(sector=0)
    if repr == "B2":
        return Rotation(angle=np.pi / 3, sector=3) + Flip(sector=1)
    if repr == "E1":
        return Rotation(angle=np.pi / 3, sector=[2, 1, -1, -2, -1, 1])
    if repr == "E2":
        return Rotation(angle=np.pi / 3, sector=[2, -1, -1, 2, -1, -1])
    raise ValueError(
        "'repr' should be one of the following: 'A1', 'A2', 'B1', 'B2', 'E1' or 'E2'"
    )
