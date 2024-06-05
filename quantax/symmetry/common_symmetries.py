"""
docstring here
"""

from typing import Union, Sequence, Optional, Tuple, List
import numpy as np
from .symmetry import Symmetry
from ..global_defs import get_sites, get_lattice


_Identity = None
_TotalSz = dict()
_Z2Inverse = dict()


def Identity() -> Symmetry:
    global _Identity
    if _Identity is None:
        _Identity = Symmetry()
    return _Identity


def ParticleConserve(Nparticle: Optional[Union[int, Tuple, List]] = None) -> Symmetry:
    """
    Particle conservation symmetry. Conserved number of spin-up for spin systems,
    and conserved (Nup, Ndown) fermions for fermion systems. The default behavior when
    `Nparticle = None` is to choose spin-up = spin-down for spin systems and
    Nup = Ndown = Nsites for fermion systems, which is different from the default
    behavior in `Symmetry`.
    """
    global _TotalSz
    sites = get_sites()

    if Nparticle is None:
        nsites = sites.nstates
        if sites.is_fermion:
            Nparticle = ((nsites, nsites),)
        else:
            if nsites % 2 == 0:
                Nparticle = (nsites // 2,)
            else:
                raise ValueError(
                    "The default number of spin-up is ill-defined for odd sites"
                )
    else:
        if sites.is_fermion:
            if isinstance(Nparticle[0], int):
                Nparticle = (tuple(Nparticle),)
            else:
                Nparticle = tuple(tuple(Np) for Np in Nparticle)
        else:
            if isinstance(Nparticle, int):
                Nparticle = (Nparticle,)
            else:
                Nparticle = tuple(Nparticle)
    if Nparticle not in _TotalSz:
        _TotalSz[Nparticle] = Symmetry(Nparticle=list(Nparticle))
    return _TotalSz[Nparticle]


def Z2Inversion(eigval: int = 1) -> Symmetry:
    if eigval not in (1, -1):
        raise ValueError("'eigval' of Z2Inversion should be 1 or -1.")

    global _Z2Inverse
    if eigval not in _Z2Inverse:
        _Z2Inverse[eigval] = Symmetry(Z2_inversion=eigval)
    return _Z2Inverse[eigval]


def SpinInverse(eigval: int = 1) -> Symmetry:
    sites = get_sites()
    if sites.is_fermion:
        if eigval == 1:
            sector = 0
        elif eigval == -1:
            sector = 1
        else:
            raise ValueError("'eigval' of spin inversion should be 1 or -1.")
        nsites = sites.nsites
        generator = np.concatenate([np.arange(nsites, 2 * nsites), np.arange(nsites)])
        return Symmetry(generator, sector)
    else:
        return Z2Inversion(eigval)


def ParticleHole(eigval: int = 1) -> Symmetry:
    if not get_sites().is_fermion:
        raise RuntimeError("`ParticleHole` symmetry is only for fermion systems.")
    return Z2Inversion(eigval)


def Translation(vector: Sequence, sector: int = 0) -> Symmetry:
    lattice = get_lattice()
    vector = np.asarray(vector, dtype=np.int64)
    xyz = lattice.xyz_from_index.copy()
    for axis, stride in enumerate(vector):
        if stride != 0:
            if lattice.boundary[axis] == 0:
                raise ValueError(f"Lattice has open boundary in axis {axis}")
            xyz[:, axis + 1] = (xyz[:, axis + 1] + stride) % lattice.shape[axis + 1]
    xyz_tuple = tuple(tuple(row) for row in xyz.T)
    generator = lattice.index_from_xyz[xyz_tuple]
    if lattice.is_fermion:
        generator = np.concatenate([generator, generator + lattice.nsites])
    return Symmetry(generator, sector)


def TransND(sector: Union[int, Tuple[int, ...]] = 0) -> Symmetry:
    dim = get_lattice().ndim
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


def LinearTransform(matrix: np.ndarray, sector: int = 0) -> Symmetry:
    """
    The symmetry applies linear transformation to the lattice
    """
    tol = 1e-6
    lattice = get_lattice()

    coord = lattice.coord
    new_coord = np.einsum("ij,nj->ni", matrix, coord)
    basis = lattice.basis_vectors.T
    new_xyz = np.linalg.solve(basis, new_coord.T).T  # dimension: ni
    offsets_xyz = np.linalg.solve(basis, lattice.site_offsets.T).T  # oi

    # site n, offset o, coord i
    new_xyz = new_xyz[None, :, :] - offsets_xyz[:, None, :]
    correct_offsets = np.abs(np.round(new_xyz) - new_xyz) < tol
    correct_offsets = np.all(correct_offsets, axis=2)
    offsets_idx = np.nonzero(correct_offsets)[0]
    new_xyz = np.rint(new_xyz[correct_offsets]).astype(np.int64)

    shape = np.asarray(lattice.shape[1:])[None, ...]
    shift = new_xyz // shape
    new_xyz = new_xyz - shift * shape

    slicing = (offsets_idx,) + tuple(item for item in new_xyz.T)
    generator = lattice.index_from_xyz[slicing]
    if lattice.is_fermion:
        generator = np.concatenate([generator, generator + lattice.nsites])
    return Symmetry(generator, sector)


def Flip(axis: Union[int, Sequence] = 0, sector: int = 0) -> Symmetry:
    matrix = np.ones(get_lattice().ndim)
    matrix[np.asarray(axis)] = -1
    matrix = np.diag(matrix)
    return LinearTransform(matrix, sector)


def Rotation(angle: float, axes: Sequence = (0, 1), sector: int = 0) -> Symmetry:
    ndim = get_lattice().ndim
    if max(axes) >= ndim:
        raise ValueError(
            f"The rotated axis {max(axes)} is out-of-bound for a {ndim}-D system"
        )
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    matrix = np.eye(ndim)
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
