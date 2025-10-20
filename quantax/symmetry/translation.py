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
        return self._vectors.copy()

    def get_sublattice_coord(self) -> np.ndarray:
        """Get the coordinate of lattice sites in the sublattice."""
        vectors = self.vectors
        lattice = get_lattice()
        if vectors.shape[0] != lattice.ndim:
            raise ValueError("Incompatible lattice and sublattice vector dimensions.")
        period = lattice.shape[1:] // np.where(vectors != 0, vectors, lattice.shape[1:])
        period = np.max(period, axis=1)

        coord = lattice.xyz_from_index.copy()
        channel = coord[:, 0]
        coord = coord[:, 1:]
        coord_sub = np.linalg.solve(vectors.T, coord.T).T
        coord_subint = np.floor(coord_sub + 1e-8).astype(np.int32)
        vec_left = coord - coord_subint @ vectors
        _, channel_sub = np.unique(vec_left, axis=0, return_inverse=True)
        channel_sub += channel * np.max(channel_sub + 1)
        coord_subint %= period
        new_coord = np.concatenate([channel_sub[:, None], coord_subint], axis=1)
        return new_coord


def TransND(sector: Union[int, Tuple[int, ...]] = 0) -> Translation:
    """N-dimensional translation symmetry with unit lattice vectors."""
    dim = get_lattice().ndim
    vector = np.identity(dim)
    return Translation(vector, sector)
