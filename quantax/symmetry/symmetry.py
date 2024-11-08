from __future__ import annotations
from functools import partial
from typing import Sequence, Optional, Union, Tuple, List
import numpy as np
import jax
import jax.numpy as jnp
from quspin.basis import spin_basis_general, spinful_fermion_basis_general
from ..global_defs import get_sites, get_default_dtype, is_default_cpl


def _get_perm(
    generator: np.ndarray, sector: list, generator_sign: np.ndarray
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    nstates = generator.shape[1]
    if np.array_equiv(generator, np.arange(nstates)):
        perm = jnp.arange(nstates)[None]
        eigval = jnp.array([1], dtype=get_default_dtype())
        perm_sign = jnp.ones((1, nstates), dtype=jnp.int8)
        return perm, eigval, perm_sign

    s0 = jnp.arange(nstates, dtype=jnp.uint16)
    sign0 = jnp.ones((nstates,), dtype=jnp.int8)
    perm = s0.reshape(1, -1)
    eigval = jnp.array([1], dtype=get_default_dtype())
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
            character = -1 if sec else 1
        else:
            character = np.exp(-2j * np.pi * sec / nperms)
        new_eigval = character ** jnp.arange(nperms)

        perm = perm[:, new_perm].reshape(-1, nstates)
        eigval = jnp.einsum("i,j->ij", eigval, new_eigval).flatten()

        perm_sign = jnp.einsum("is,js->ijs", perm_sign, new_sign).reshape(-1, nstates)
        at_set = lambda arr, idx: arr.at[idx].set(arr)
        perm_sign = jax.vmap(at_set)(perm_sign, perm)

    return perm, eigval, perm_sign


@partial(jax.jit, static_argnums=3)
@partial(jax.vmap, in_axes=(None, 0, 0, None))
def _permutation_sign(
    spins: jax.Array, perm: jax.Array, perm_sign: jax.Array, Nparticle: Optional[int]
) -> jax.Array:
    perm = jnp.argsort(perm)  # invert permmutation
    N = jnp.max(perm) + 1
    perm = jnp.where(spins > 0, perm, N)
    arg = jnp.argsort(perm == N, stable=True)
    perm = perm[arg]
    if Nparticle is not None:
        perm = perm[:Nparticle]

    compare = perm[None, :] > perm[:, None]
    compare = compare[jnp.tril_indices_from(compare, k=-1)]
    sign = jnp.where(jnp.sum(compare) % 2, -1, 1)

    perm_sign = jnp.where(spins > 0, perm_sign, 1)
    perm_sign = jnp.prod(perm_sign)
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
        Nparticle: Union[None, int, Tuple[int, int]] = None,
        perm: Optional[jax.Array] = None,
        eigval: Optional[jax.Array] = None,
        perm_sign: Optional[jax.Array] = None,
    ):
        """
        :param generator:
            The permutation that generates the symmetry group, default to identity.
            The input can be a 1D generator array, or a 2D array with each row a generator.
        :param sector:
            The symmetry sector :math:`q` used to compute the eigenvalues.
            Assume the periodicity of the generator is :math:`m`, then the eigenvalues
            are given by :math:`\omega_i = e^{-2 \pi i q / m}`.
            The sector is 0 by default, giving all eigenvalues equal to 1.
        :param generator_sign:
            The additional sign associated with the generator, usually caused by the
            anti-periodic boundary condition.
            In most other cases, this can be left as None by defaul.
        :param Z2_inversion:
            The Z2 inversion symmetry. This represents the spin flip in spin-1/2 systems,
            and the particle-hole symmetry in fermionic systems.
            The meaning of each number is

                1: Eigenvalue 1 after Z2 inversion

                0: No Z2 inversion symmetry

                -1: Eigenvalue -1 after Z2 inversion
        :param Nparticle:
            Number of particles conservation.
            This is the number of spin-up in spin-1/2 systems,
            and a pair indicating the number of spin-up fermions and spin-down fermions
            in spinful fermion systems. The amount of particles is not conserved by default.
        :param perm:
            Directly specify the permutations :math:`T` generated by the generator.
            If not specified, this value will be computed in initialization.
        :param eigval:
            Directly specify the eigenvalues :math:`\omega` generated by the sector.
            If not specified, this value will be computed in initialization.
        :param perm_sign:
            Directly specify the signs associated to the permutations generated by
            the generator sign.
            If not specified, this value will be computed in initialization.
        """
        sites = get_sites()
        self._nstates = sites.nstates
        self._is_fermion = sites.is_fermion

        if generator is None:
            generator = np.atleast_2d(np.arange(self._nstates, dtype=np.uint16))
        else:
            generator = np.atleast_2d(generator).astype(np.uint16)
            if generator.shape[1] != self._nstates:
                raise ValueError(
                    f"Got a generator with size {generator.shape[1]}, but it should be"
                    f"the same as the system size {self._nstates}."
                )
        self._generator = generator

        if generator_sign is None:
            generator_sign = np.ones((1, self._nstates), dtype=np.int8)
        else:
            generator_sign = np.atleast_2d(generator_sign).astype(np.int8)
        self._generator_sign = generator_sign

        self._sector = np.asarray(sector, dtype=np.uint16).flatten().tolist()
        self._Z2_inversion = Z2_inversion

        if Nparticle is not None:
            if sites.is_fermion:
                Nparticle = tuple(Nparticle)
            else:
                if isinstance(Nparticle, int):
                    Nparticle = (Nparticle, sites.nsites - Nparticle)
                else:
                    Nparticle = tuple(Nparticle)
        self._Nparticle = Nparticle

        if perm is None or eigval is None or perm_sign is None:
            new_perm, new_eigval, new_perm_sign = _get_perm(
                self._generator, self._sector, self._generator_sign
            )
            if perm is None:
                perm = new_perm
            if eigval is None:
                eigval = new_eigval
            if perm_sign is None:
                perm_sign = new_perm_sign

        self._perm = jnp.asarray(perm, dtype=jnp.uint16)
        self._eigval = jnp.asarray(eigval, dtype=get_default_dtype())
        self._perm_sign = jnp.asarray(perm_sign, dtype=jnp.int8)
        self._basis = None
        self._is_basis_made = False

    @property
    def nsites(self) -> int:
        return self.nstates // 2 if self.is_fermion else self.nstates

    @property
    def is_fermion(self) -> bool:
        return self._is_fermion

    @property
    def nstates(self) -> int:
        return self._nstates

    @property
    def eigval(self) -> jax.Array:
        """Eigenvalues"""
        return self._eigval

    @property
    def nsymm(self) -> int:
        """Number of elements in the symmetry group"""
        return self.eigval.size if self.Z2_inversion == 0 else 2 * self.eigval.size

    @property
    def Z2_inversion(self) -> int:
        """Z2 inversion symmetry"""
        return self._Z2_inversion

    @property
    def Nparticle(self) -> Optional[Tuple[int, int]]:
        """
        Number of particle conservation.
        Return a tuple of two integers for the number of spin-up and spin-down particles,
        or `None` if there is no particle number conservation.
        """
        return self._Nparticle

    @property
    def basis(self) -> Union[spin_basis_general, spinful_fermion_basis_general]:
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

        if not jnp.allclose(jnp.abs(self.eigval), 1.0):
            raise RuntimeError(
                "QuSpin doesn't support eigenvalues with absolute values not equal to 1."
                "This happens when a high-dimensional group representation is utilized."
            )

        blocks = dict()
        for i, (g, s) in enumerate(zip(self._generator, self._sector)):
            if not np.allclose(g, np.arange(g.size)):
                blocks[f"block{i}"] = (g, s)

        if self.Z2_inversion != 0:
            sector = 0 if self.Z2_inversion == 1 else 1
            blocks["inversion"] = (-np.arange(self.nsites) - 1, sector)

        if self.is_fermion:
            basis = spinful_fermion_basis_general(
                self.nsites,
                self.Nparticle,
                simple_symm=False,
                make_basis=False,
                **blocks,
            )
        else:
            basis = spin_basis_general(
                self.nsites, self.Nparticle[0], pauli=0, make_basis=False, **blocks
            )
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
        """
        For a input spin configuration `s`, obtain all symmetrized spins :math:`s'=Ts`
        generated by the symmetry permutations.

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
    def symmetrize(
        self, psi: jax.Array, spins: Optional[jax.Array] = None
    ) -> jax.Array:
        """
        Symmetrize the wave function as
        :math:`\psi^{symm}(s) = \sum_i \omega_i \, \psi(T_i s) / n_{symm}`.

        :param psi: Wave functions :math:`\psi(T_i s)`
        :param spins:
            Symmetry spins :math:`T_i s`.
            This is only necessary in fermionic systems, where additional
            minus signs can be generated by the symmetry depending on the input spins.

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
            Nparticle = None if self.Nparticle is None else sum(self.Nparticle)
            sign = _permutation_sign(spins, self._perm, self._perm_sign, Nparticle)
            eigval = sign * self._eigval
        else:
            eigval = self._eigval

        if self.Z2_inversion != 0:
            eigval = jnp.concatenate([eigval, self.Z2_inversion * eigval])
        eigval = (eigval / eigval.size).astype(psi.dtype)
        return jnp.dot(psi, eigval)

    def __add__(self, other: Symmetry) -> Symmetry:
        """
        Generate superposition of two symmetries.

        .. code-block:: python

            from quantax.sites import Square
            from quantax.symmetry import Translation

            lattice = Square(4)
            trans1 = Translation((1, 0))  # translation symmetry in x axis
            trans2 = Translation((0, 1))  # translation symmetry in y axis
            trans2D = trans1 + trans2  # 2D translation symmetry

        .. warning::
            Users are responsible to ensure the compatibility of different symmetries,
            especially if their generators don't commute.
        """
        generator = np.concatenate([self._generator, other._generator], axis=0)
        sector = [*self._sector, *other._sector]
        g_sign = np.concatenate([self._generator_sign, other._generator_sign], axis=0)
        perm = self._perm[:, other._perm].reshape(-1, self.nstates)
        eigval = jnp.einsum("i,j->ij", self._eigval, other._eigval).flatten()
        p_sign = jnp.einsum("is,js->ijs", self._perm_sign, other._perm_sign)
        p_sign = p_sign.reshape(-1, self.nstates)

        if self.Z2_inversion == 0:
            Z2_inversion = other.Z2_inversion
        elif other.Z2_inversion == 0:
            Z2_inversion = self.Z2_inversion
        elif self.Z2_inversion == other.Z2_inversion:
            Z2_inversion = self.Z2_inversion
        else:
            raise ValueError("Symmetry with different Z2_inversion can't be added")

        if self.Nparticle is None:
            Nparticle = other.Nparticle
        elif other.Nparticle is None:
            Nparticle = self.Nparticle
        elif self.Nparticle == other.Nparticle:
            Nparticle = self.Nparticle
        else:
            raise ValueError("Symmetry with different Nparticle can't be added")

        new_symm = Symmetry(
            generator, sector, g_sign, Z2_inversion, Nparticle, perm, eigval, p_sign
        )
        return new_symm
