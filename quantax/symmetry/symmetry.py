from __future__ import annotations
from functools import partial
from typing import Sequence, Optional, Union, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from ..global_defs import PARTICLE_TYPE, get_sites, get_default_dtype, is_default_cpl
from ..utils import PsiArray


def _get_perm(
    generator: np.ndarray, sector: list, generator_sign: np.ndarray
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    Nmodes = generator.shape[1]
    if np.array_equiv(generator, np.arange(Nmodes)):
        perm = jnp.arange(Nmodes)[None]
        character = jnp.array([1], dtype=get_default_dtype())
        perm_sign = jnp.ones((1, Nmodes), dtype=jnp.int8)
        return perm, character, perm_sign

    s0 = jnp.arange(Nmodes, dtype=jnp.uint16)
    sign0 = jnp.ones((Nmodes,), dtype=jnp.int8)
    perm = s0.reshape(1, -1)
    character = jnp.array([1], dtype=get_default_dtype())
    perm_sign = sign0.reshape(1, -1)

    for g, sec, gs in zip(generator, sector, generator_sign):
        g = jnp.asarray(g, dtype=jnp.uint16)
        new_perm = [s0]
        new_sign = [sign0]
        s_perm = g
        while not jnp.array_equal(s0, s_perm):
            new_perm.append(s_perm)
            new_sign.append(new_sign[-1][g] * gs)
            s_perm = s_perm[g]
        new_perm = jnp.stack(new_perm, axis=0)
        new_sign = jnp.stack(new_sign, axis=0)
        nperms = new_perm.shape[0]

        if not 0 <= sec < nperms:
            raise ValueError(f"Sector {sec} out of range.")
        if not is_default_cpl():
            if (sec * 2) % nperms != 0:
                raise ValueError("Default dtype is real, but got complex characters.")
            chi = -1 if sec else 1
        else:
            chi = np.exp(-2j * np.pi * sec / nperms)
        new_character = chi ** jnp.arange(nperms)

        perm = perm[:, new_perm].reshape(-1, Nmodes)
        character = jnp.einsum("i,j->ij", character, new_character).flatten()

        perm_sign = jnp.einsum("is,js->ijs", perm_sign, new_sign).reshape(-1, Nmodes)
        at_set = lambda arr, idx: arr.at[idx].set(arr)
        perm_sign = jax.vmap(at_set)(perm_sign, perm)

    return perm, character, perm_sign


@jax.jit
@partial(jax.vmap, in_axes=(None, 0, 0))
def _permutation_sign(
    spins: jax.Array, perm: jax.Array, perm_sign: jax.Array
) -> jax.Array:
    """Compute the permutation sign. This function will be slow if Nparticles is None"""
    perm = jnp.argsort(perm)  # invert permmutation

    Ntotal = get_sites().Ntotal
    if Ntotal is None:
        Nmodes = jnp.max(perm) + 1
        perm = jnp.where(spins > 0, perm, Nmodes)
        arg = jnp.argsort(perm == Nmodes, stable=True)
        perm = perm[arg]
    else:
        indices = jnp.flatnonzero(spins > 0, size=Ntotal)
        perm = perm[indices]

    compare = perm[None, :] > perm[:, None]
    compare = compare[jnp.tril_indices_from(compare, k=-1)]
    sign = jnp.where(jnp.sum(compare) % 2, -1, 1)

    additional_sign = jnp.sum((spins > 0) & (perm_sign < 0))
    perm_sign = jnp.where(additional_sign % 2 == 0, 1, -1).astype(perm_sign.dtype)
    return sign * perm_sign


class Symmetry:
    """
    Symmetry of the system.
    """

    def __init__(
        self,
        generator: Optional[np.ndarray] = None,
        sector: Union[int, Sequence] = 0,
        generator_sign: Optional[np.ndarray] = None,
        Z2_inversion: int = 0,
        perm: Optional[jax.Array] = None,
        character: Optional[jax.Array] = None,
        perm_sign: Optional[jax.Array] = None,
    ):
        r"""
        :param generator:
            The permutation of the group generator(s), default to identity.
            The input can be a 1D generator array, or a 2D array with each row a
            different generator. The group is generated as
            :math:`G = \{ g_1^m g_2^n ... | m,n \in \mathbb{Z} \}`.

        :param sector:
            The symmetry sector :math:`q` of each generator.
            Assume the periodicities of generators :math:`g_1, g_2` are :math:`M, N`,
            then the characters of the group element :math:`g = g_1^m g_2^n`
            is given by :math:`\chi = \exp[-2 \pi i (m q_1 / M + n q_2 / N)]`.
            The sector is 0 by default, giving :math:`\chi = 1`.

        :param generator_sign:
            The additional sign associated with the generator, usually caused by the
            anti-periodic boundary condition.
            In most other cases, this can be left as None by default.

        :param Z2_inversion:
            The Z2 inversion symmetry. It represents the spin flip in spin-1/2 systems,
            and the particle-hole symmetry in fermionic systems.
            The meaning of each number is

            - 1: Eigenvalue 1 after Z2 inversion
            - 0: No Z2 inversion symmetry
            - -1: Eigenvalue -1 after Z2 inversion

        :param perm:
            Directly specify the permutations of all group elements.
            If not specified, this value will be computed from `generator`.

        :param character:
            Directly specify the characters :math:`\chi` of all group elements.
            If not specified, this value will be computed from `sector`.
            One needs to specify this argument if the group representation is
            high-dimensional, in which case the characters can't be computed
            from `generator` and `sector`.

        :param perm_sign:
            Directly specify the signs associated to the permutations generated by
            the generator sign.
            If not specified, this value will be computed from `generator_sign`.
        """
        sites = get_sites()
        self._Nmodes = sites.Nmodes
        self._Nparticles = sites.Nparticles
        self._particle_type = sites.particle_type
        self._double_occ = sites.double_occ

        if generator is None:
            generator = np.atleast_2d(np.arange(self._Nmodes, dtype=np.uint16))
        else:
            generator = np.atleast_2d(generator).astype(np.uint16)
            if generator.shape[1] != self._Nmodes:
                raise ValueError(
                    f"Got a generator with size {generator.shape[1]}, incompatible with "
                    f"the system size {self._Nmodes}."
                )
        self._generator = generator

        if generator_sign is None:
            generator_sign = np.ones((1, self._Nmodes), dtype=np.int8)
        else:
            generator_sign = np.atleast_2d(generator_sign).astype(np.int8)
        self._generator_sign = generator_sign

        if isinstance(sector, int):
            self._sector = [sector] * generator.shape[0]
        else:
            self._sector = np.asarray(sector).flatten().tolist()
        self._Z2_inversion = Z2_inversion

        if perm is None or character is None or perm_sign is None:
            new_perm, new_character, new_perm_sign = _get_perm(
                self._generator, self._sector, self._generator_sign
            )
            if perm is None:
                perm = new_perm
            if character is None:
                character = new_character
            if perm_sign is None:
                perm_sign = new_perm_sign

        self._perm = jnp.asarray(perm, dtype=jnp.uint16)
        self._character = jnp.asarray(character, dtype=get_default_dtype())
        self._perm_sign = jnp.asarray(perm_sign, dtype=jnp.int8)
        self._basis = None
        self._is_basis_made = False

    @property
    def Nsites(self) -> int:
        M = self.Nmodes
        return M // 2 if self.particle_type == PARTICLE_TYPE.spinful_fermion else M

    @property
    def Nmodes(self) -> int:
        return self._Nmodes

    @property
    def Nparticles(self) -> Optional[Tuple[int, int]]:
        return self._Nparticles

    @property
    def particle_type(self) -> PARTICLE_TYPE:
        return self._particle_type

    @property
    def double_occ(self) -> bool:
        return self._double_occ

    @property
    def is_fermion(self) -> bool:
        return self._particle_type in (
            PARTICLE_TYPE.spinful_fermion,
            PARTICLE_TYPE.spinless_fermion,
        )

    @property
    def is_spinful(self) -> bool:
        return self._particle_type in (
            PARTICLE_TYPE.spin,
            PARTICLE_TYPE.spinful_fermion,
        )

    @property
    def character(self) -> jax.Array:
        r"""Characters :math:`\chi` of all group elements."""
        return self._character

    @property
    def nsymm(self) -> int:
        """Number of elements in the symmetry group."""
        nsymm = self._character.size
        return nsymm if self.Z2_inversion == 0 else 2 * nsymm

    @property
    def Z2_inversion(self) -> int:
        """
        The Z2 inversion symmetry. It represents the spin flip in spin-1/2 systems,
        and the particle-hole symmetry in fermionic systems.
        The meaning of each number is

        - 1: Eigenvalue 1 after Z2 inversion
        - 0: No Z2 inversion symmetry
        - -1: Eigenvalue -1 after Z2 inversion
        """
        return self._Z2_inversion

    @property
    def basis(self):
        """
        The `QuSpin basis <https://quspin.github.io/QuSpin/basis.html>`_
        corresponding to the symmetry.
        """
        if self._basis is not None:
            return self._basis

        if not np.all(self._perm_sign == 1):
            raise RuntimeError(
                "QuSpin doesn't support non-trivial permmutation sign. This happens "
                "when anti-periodic boundary is used for translation symmetry."
            )

        if not jnp.allclose(jnp.abs(self.character), 1.0):
            raise RuntimeError(
                "QuSpin doesn't support eigenvalues with absolute values not equal to 1."
                "This happens when a high-dimensional group representation is utilized."
            )
        from quspin.basis import (
            spin_basis_general,
            spinful_fermion_basis_general,
            spinless_fermion_basis_general,
        )

        blocks = dict()
        for i, (g, s) in enumerate(zip(self._generator, self._sector)):
            if not np.allclose(g, np.arange(g.size)):
                blocks[f"block{i}"] = (g, s)

        if self.Z2_inversion != 0:
            sector = 0 if self.Z2_inversion == 1 else 1
            blocks["inversion"] = (-np.arange(self.Nsites) - 1, sector)

        if self.particle_type == PARTICLE_TYPE.spin:
            Nup = self.Nparticles[0] if isinstance(self.Nparticles, tuple) else None
            basis = spin_basis_general(
                self.Nsites, Nup, pauli=0, make_basis=False, **blocks
            )
        elif self.particle_type == PARTICLE_TYPE.spinful_fermion:
            if self.Nparticles is None or isinstance(self.Nparticles, tuple):
                Nparticles = self.Nparticles
            else:
                Ntotal = self.Nparticles
                Nparticles = [(Nup, Ntotal - Nup) for Nup in range(Ntotal + 1)]
            basis = spinful_fermion_basis_general(
                self.Nsites,
                Nparticles,
                simple_symm=False,
                make_basis=False,
                double_occupancy=self.double_occ,
                **blocks,
            )
        elif self.particle_type == PARTICLE_TYPE.spinless_fermion:
            basis = spinless_fermion_basis_general(
                self.Nsites, self.Nparticles, make_basis=False, **blocks
            )
        else:
            raise NotImplementedError

        self._basis = basis
        return self._basis

    def basis_make(self) -> None:
        """
        `Make <https://quspin.github.io/QuSpin/generated/quspin.basis.spin_basis_general.html#quspin.basis.spin_basis_general.make>`_
        the QuSpin basis stored in the symmetry.
        """
        if not self._is_basis_made:
            self.basis.make()
        self._is_basis_made = True

    @partial(jax.jit, static_argnums=0)
    def get_symm_spins(self, spins: jax.Array) -> jax.Array:
        r"""
        Get all symmetrized bases :math:`s'=Ts` generated by the symmetry permutations.

        :param spins:
            Input spin or fermion configuration :math:`s`.

        .. note::

            This function is jittable.
        """
        if spins.ndim > 1:
            raise ValueError(f"Input spins should be 1D, got dimension {spins.ndim}")
        spins = spins[self._perm]
        if self.Z2_inversion != 0:
            spins = jnp.concatenate([spins, -spins], axis=-2)
        return spins

    @partial(jax.jit, static_argnums=0)
    def symmetrize(self, psi: PsiArray, spins: Optional[jax.Array] = None) -> PsiArray:
        r"""
        Symmetrize the wavefunction as

        .. math::

            \psi^{\mathrm{symm}}(s) = \frac{d}{|G|} \sum_g \mathrm{sign}(s, g) \chi_g \psi(T_g s),
        
        where :math:`d` is the dimension of the group representation,
        :math:`|G|` is the number of elements in the symmetry group, and
        :math:`\mathrm{sign}(s, g)` is the additional sign generated by the symmetry
        depending on the input Fock state (only for fermionic systems).

        :param psi:
            Wavefunctions :math:`\psi(T_i s)`.

        :param spins:
            Input spin or fermion configuration :math:`s`.
            This is only necessary in fermionic systems, where additional
            minus signs :math:`\mathrm{sign}(s, g)` can be generated by the symmetry.

        :return:
            The symmetrized wave function

        .. note::

            This function is jittable.
        """
        if psi.ndim > 1:
            raise ValueError(
                f"Input wavefunction should be 1D, got dimension {psi.ndim}"
            )
        if spins is not None and spins.ndim > 1:
            raise ValueError(f"Input spins should be 1D, got dimension {spins.ndim}")

        if self.is_fermion:
            if spins is None:
                raise RuntimeError(
                    "`symmetrize` can't be performed without input fermion states."
                )
            sign = _permutation_sign(spins, self._perm, self._perm_sign)
            character = sign * self._character
        else:
            character = self._character

        if self.Z2_inversion != 0:
            character = jnp.concatenate([character, self.Z2_inversion * character])
        character = (character * character[0] / character.size).astype(psi.dtype)
        return (psi * character).sum()

    def __matmul__(self, other: Symmetry) -> Symmetry:
        r"""
        Generate superposition of two symmetries.

        .. code-block:: python

            from quantax.sites import Square
            from quantax.symmetry import Translation

            lattice = Square(4)
            trans1 = Translation((1, 0))  # translation symmetry in x axis
            trans2 = Translation((0, 1))  # translation symmetry in y axis
            trans2D = trans1 @ trans2  # 2D translation symmetry

        .. warning::
            Users are responsible to ensure the compatibility of different symmetries,
            especially if their generators don't commute.
        """
        generator = np.concatenate([self._generator, other._generator], axis=0)
        sector = [*self._sector, *other._sector]
        g_sign = np.concatenate([self._generator_sign, other._generator_sign], axis=0)
        perm = self._perm[:, other._perm].reshape(-1, self.Nmodes)
        character = jnp.einsum("i,j->ij", self._character, other._character).flatten()
        p_sign = jnp.einsum("is,js->ijs", self._perm_sign, other._perm_sign)
        p_sign = p_sign.reshape(-1, self.Nmodes)

        if self.Z2_inversion == 0:
            Z2_inversion = other.Z2_inversion
        elif other.Z2_inversion == 0:
            Z2_inversion = self.Z2_inversion
        elif self.Z2_inversion == other.Z2_inversion:
            Z2_inversion = self.Z2_inversion
        else:
            raise ValueError("Symmetry with different Z2_inversion can't be added")

        new_symm = Symmetry(
            generator, sector, g_sign, Z2_inversion, perm, character, p_sign
        )
        return new_symm
