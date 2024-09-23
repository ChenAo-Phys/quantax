from typing import Union, Sequence, Optional, Tuple, List
import numpy as np
import jax
import jax.numpy as jnp
from .symmetry import Symmetry
from ..global_defs import get_sites, get_lattice, get_default_dtype


_Identity = None
_TotalSz = dict()
_Z2Inverse = dict()


def Identity() -> Symmetry:
    """Identity symmetry (no additional symmetry)"""
    global _Identity
    if _Identity is None:
        _Identity = Symmetry()
    return _Identity


def ParticleConserve(Nparticle: Optional[Union[int, Tuple, List]] = None) -> Symmetry:
    """
    Particle conservation symmetry. Conserved number of spin-up for spin systems,
    and conserved (Nup, Ndown) fermions for fermion systems. The default behavior when
    ``Nparticle = None`` is to choose spin-up = spin-down for spin systems and
    Nup = Ndown = Nsites for fermion systems.

    .. note::
        The default behavior here is the same amount of spin-up and spin-down,
        while in the initialization of class `~quantax.symmetry.Symmetry` the default
        behavior is no particle conservation symmetry.
    """
    global _TotalSz
    sites = get_sites()

    if Nparticle is None:
        if sites.nsites % 2 == 0:
            Nhalf = sites.nsites // 2
            Nparticle = ((Nhalf, Nhalf),) if sites.is_fermion else (Nhalf,)
        else:
            raise ValueError(
                "The default number of particles is ill-defined for odd sites"
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
    """
    The Z2 inversion symmetry. This represents the spin flip in spin-1/2 systems,
    and the particle-hole symmetry in fermionic systems.

    :param eigval: An integer specifying the symmetry sector. The meaning of each number is

        1: Eigenvalue 1 after Z2 inversion

        0: No Z2 inversion symmetry

        -1: Eigenvalue -1 after Z2 inversion
    """
    if eigval not in (1, -1):
        raise ValueError("'eigval' of Z2Inversion should be 1 or -1.")

    global _Z2Inverse
    if eigval not in _Z2Inverse:
        _Z2Inverse[eigval] = Symmetry(Z2_inversion=eigval)
    return _Z2Inverse[eigval]


def SpinInverse(eigval: int = 1) -> Symmetry:
    """
    Spin inverse symmetry for both spin-1/2 and spinful fermion systems.
    In spin systems it flips all spins.
    In spinful fermion systems it exchanges particles between spin-up and spin-down sites.

    :param eigval: An integer specifying the symmetry sector. The meaning of each number is

        1: Eigenvalue 1 after spin inversion

        0: No spin inversion

        -1: Eigenvalue -1 after spin inversion
    """
    sites = get_sites()
    if sites.is_fermion:
        if eigval == 1:
            sector = 0
        elif eigval == -1:
            sector = 1
        else:
            return Identity()
        nsites = sites.nsites
        generator = np.concatenate([np.arange(nsites, 2 * nsites), np.arange(nsites)])
        return Symmetry(generator, sector)
    else:
        return Z2Inversion(eigval)


def ParticleHole(eigval: int = 1) -> Symmetry:
    """
    Particle-hole symmetry for fermion systems

    :param eigval:
        An integer specifying the symmetry sector. The meaning of each number is

            1: Eigenvalue 1 after particle-hole inversion

            0: No particle-hole inversion

            -1: Eigenvalue -1 after particle-hole inversion
    """
    if not get_sites().is_fermion:
        raise RuntimeError("`ParticleHole` symmetry is only for fermion systems.")
    return Z2Inversion(eigval)


def Translation(vector: Sequence, sector: int = 0) -> Symmetry:
    """
    Translation symmetry

    :param vector:
        The vector generating the translation

    :param sector:
        The symmetry sector
    """
    lattice = get_lattice()
    if len(vector) != lattice.ndim:
        raise ValueError("The translation vector doesn't match the lattice dimension.")
    vector = np.asarray(vector, dtype=np.int64)
    if np.any((vector != 0) & (lattice.boundary == 0)):
        raise ValueError("Translation symmetry can't be imposed on open boundary.")

    xyz = lattice.xyz_from_index.copy()
    xyz[:, 1:] += vector[None, :]
    generator_sign = lattice.boundary[None, :] ** (xyz[:, 1:] // lattice.shape[1:])
    generator_sign = np.prod(generator_sign, axis=1)
    xyz[:, 1:] %= lattice.shape[1:]

    xyz_tuple = tuple(tuple(row) for row in xyz.T)
    generator = lattice.index_from_xyz[xyz_tuple]
    if lattice.is_fermion:
        generator = np.concatenate([generator, generator + lattice.nsites])
        generator_sign = np.concatenate([generator_sign, generator_sign])
    return Symmetry(generator, sector, generator_sign)


def TransND(sector: Union[int, Tuple[int, ...]] = 0) -> Symmetry:
    """N-dimensional translation symmetry"""
    dim = get_lattice().ndim
    if isinstance(sector, int):
        sector = [sector] * dim
    vector = np.identity(dim)
    symm_list = [Translation(vec, sec) for vec, sec in zip(vector, sector)]
    symm = sum(symm_list, start=Identity())
    return symm


def Trans1D(sector: int = 0) -> Symmetry:
    """1D translation"""
    return Translation([1], sector)


def Trans2D(sector: Union[int, Tuple[int, int]] = 0) -> Symmetry:
    """2D trnaslation"""
    if isinstance(sector, int):
        sector = [sector, sector]
    return Translation([1, 0], sector[0]) + Translation([0, 1], sector[1])


def Trans3D(sector: Union[int, Tuple[int, int, int]] = 0) -> Symmetry:
    """3D translation"""
    if isinstance(sector, int):
        sector = [sector, sector, sector]
    return (
        Translation([1, 0, 0], sector[0])
        + Translation([0, 1, 0], sector[1])
        + Translation([0, 0, 1], sector[2])
    )


def LinearTransform(
    matrix: np.ndarray, sector: int = 0, eigval: Optional[jax.Array] = None
) -> Symmetry:
    """
    The symmetry generated by a linear transformation to the lattice

    :param matrix:
        A 2D array for the linear transformation matrix

    :param sector:
        The symmetry sector

    :param eigval:
        The eigenvalue (character) of the symmetry group elements.
        This is useful when the symmetry sector doesn't have a 1D group representation.

    .. warning::
        The users are responsible to ensure that the linear transformation generates
        a symmetry group.
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
    return Symmetry(generator, sector, eigval=eigval)


def Flip(axis: Union[int, Sequence] = 0, sector: int = 0) -> Symmetry:
    """
    Flip the lattice in a spatial dimension

    :param axis:
        The axis to flip the lattice

    :param sector:
        The symmetry sector
    """
    matrix = np.ones(get_lattice().ndim)
    matrix[np.asarray(axis)] = -1
    matrix = np.diag(matrix)
    return LinearTransform(matrix, sector)


def Rotation(
    angle: float,
    axes: Sequence = (0, 1),
    sector: int = 0,
    eigval: Optional[jax.Array] = None,
) -> Symmetry:
    """
    Rotation symmetry

    :param angle:
        Rotation angle

    :param axes:
        Two rotated axes

    :param sector:
        The symmetry sector

    :param eigval:
        The eigenvalue (character) of the symmetry group elements.
        This is useful when the symmetry sector doesn't have a 1D group representation.
    """
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
    return LinearTransform(matrix, sector, eigval=eigval)


def C4v(repr: str = "A1") -> Symmetry:
    """
    `C4v <http://symmetry.jacobs-university.de/cgi-bin/group.cgi?group=404&option=4>`_
    symmetry group

    :param repr:
        The representation of the group, chosen among 'A1', 'A2', 'B1', 'B2' or 'E'.
    """
    if repr == "A1":
        return Rotation(angle=np.pi / 2, sector=0) + Flip(sector=0)
    if repr == "A2":
        return Rotation(angle=np.pi / 2, sector=0) + Flip(sector=1)
    if repr == "B1":
        return Rotation(angle=np.pi / 2, sector=2) + Flip(sector=0)
    if repr == "B2":
        return Rotation(angle=np.pi / 2, sector=2) + Flip(sector=1)
    if repr == "E":
        return Rotation(angle=np.pi, eigval=jnp.array([2, -2], get_default_dtype()))
    raise ValueError(
        "'repr' should be one of the following: 'A1', 'A2', 'B1', 'B2' or 'E'"
    )


def D6(repr: str = "A1") -> Symmetry:
    """
    `D6 <http://symmetry.jacobs-university.de/cgi-bin/group.cgi?group=306&option=4>`_
    symmetry group

    :param repr:
        The representation of the group, chosen among 'A1', 'A2', 'B1', 'B2', 'E1' or 'E2'.
    """
    if repr == "A1":
        return Rotation(angle=np.pi / 3, sector=0) + Flip(sector=0)
    if repr == "A2":
        return Rotation(angle=np.pi / 3, sector=0) + Flip(sector=1)
    if repr == "B1":
        return Rotation(angle=np.pi / 3, sector=3) + Flip(sector=0)
    if repr == "B2":
        return Rotation(angle=np.pi / 3, sector=3) + Flip(sector=1)
    if repr == "E1":
        return Rotation(
            angle=np.pi / 3,
            eigval=jnp.array([2, 1, -1, -2, -1, 1], get_default_dtype()),
        )
    if repr == "E2":
        return Rotation(
            angle=np.pi / 3,
            eigval=jnp.array([2, -1, -1, 2, -1, -1], get_default_dtype()),
        )
    raise ValueError(
        "'repr' should be one of the following: 'A1', 'A2', 'B1', 'B2', 'E1' or 'E2'"
    )
