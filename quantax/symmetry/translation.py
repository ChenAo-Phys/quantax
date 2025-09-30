from typing import Sequence, Tuple, Union
import numpy as np
from .symmetry import Symmetry
from ..global_defs import PARTICLE_TYPE, get_lattice


class Translation(Symmetry):
    """Translation symmetry."""

    def __init__(self, vectors: Sequence, sector: int = 0):
        """
        Translation symmetry.

        :param vectors:
            The translation vectors, in the unit of lattice basis vectors.

        :param sector:
            The symmetry sector.
        """
        lattice = get_lattice()
        vectors = np.asarray(vectors, dtype=np.int64).reshape(-1, lattice.ndim)
        if np.any((vectors != 0) & (lattice.boundary[None, :] == 0)):
            raise ValueError("Translation symmetry can't be imposed on open boundary.")

        generator = []
        generator_sign = []
        for vec in vectors:
            for axis, vi in enumerate(vec):
                if vi != 0 and lattice.shape[axis + 1] % vi != 0:
                    raise ValueError(
                        "Translation vector must be compatible with lattice shape, "
                        f"got lattice shape {lattice.shape[1:]} and vector {vec}."
                    )

            xyz = lattice.xyz_from_index.copy()
            xyz[:, 1:] += vec[None, :]
            sign = lattice.boundary[None, :] ** (xyz[:, 1:] // lattice.shape[1:])
            sign = np.prod(sign, axis=1)
            xyz[:, 1:] %= lattice.shape[1:]

            xyz_tuple = tuple(tuple(row) for row in xyz.T)
            g = lattice.index_from_xyz[xyz_tuple]
            generator.append(g)
            generator_sign.append(sign)

        generator = np.stack(generator, axis=0)
        generator_sign = np.stack(generator_sign, axis=0)

        if lattice.particle_type == PARTICLE_TYPE.spinful_fermion:
            generator = np.concatenate([generator, generator + lattice.Nsites], axis=1)
            generator_sign = np.concatenate([generator_sign, generator_sign], axis=1)

        self._vectors = vectors
        super().__init__(generator, sector, generator_sign)

    @property
    def vectors(self) -> np.ndarray:
        """The translation vectors."""
        return self._vectors

    def get_sublattice_coord(self) -> np.ndarray:
        """Get the coordinate of lattice sites in the sublattice."""
        vectors = self.vectors
        lattice = get_lattice()
        if vectors.shape[0] != lattice.ndim:
            raise ValueError("Incompatible lattice and sublattice vector dimensions.")

        shape = np.asarray(lattice.shape[1:])
        sub_shape = []
        for i, vec in enumerate(vectors):
            vectors[i] = np.where(vec != 0, vec, shape)
            sub_shape.append(np.max(shape // np.abs(vec)))
        sub_shape = np.asarray(sub_shape)

        coord = lattice.xyz_from_index.copy()
        new_coord = coord.copy()
        new_coord[:, 0] *= np.prod(sub_shape)
        for i, vec in enumerate(vectors):
            new_coord[:, i + 1] = np.max(coord[:, 1:] // vec[None], axis=1)
            coord[:, 1:] %= vec[None]

        is_cell0 = np.all(new_coord[:, 1:] == 0, axis=1)
        cell_coord = coord[is_cell0, 1:]
        for i, coord_i in enumerate(cell_coord):
            coord_equal = np.all(coord[:, 1:] == coord_i[None, :], axis=1)
            new_coord[coord_equal, 0] += i

        return new_coord


def TransND(sector: Union[int, Tuple[int, ...]] = 0) -> Symmetry:
    """N-dimensional translation symmetry with unit lattice vectors."""
    dim = get_lattice().ndim
    vector = np.identity(dim)
    return Translation(vector, sector)
