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

        if not isinstance(sec, int):
            new_eigval = jnp.asarray(sec)
        else:
            if not 0 <= sec < nperms:
                raise ValueError(f"Sector {sec} out of range.")
            if not is_default_cpl():
                if (sec * 2) % nperms != 0:
                    raise ValueError(
                        "Default dtype is real, but got complex characters."
                    )
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

    Args:
        generator: The permutations Ti that generates the symmetry group.
        sector: Symmetry sectors specifying the eigenvalues.
        Z2_inversion: Spin inversion symmetry for spin systems, and particle-hole
            symmetry for fermion systems.
        Nparticle: Number of spin-up for spin systems, and a tuple (Nup, Ndown) for
            fermion systems.
        perm: All permutations computed from generator.
        eigval: All eigenvalues computed from sector.
    """

    def __init__(
        self,
        generator: Optional[np.ndarray] = None,
        sector: Union[int, Sequence] = 0,
        generator_sign: Optional[np.ndarray] = None,
        Z2_inversion: int = 0,
        Nparticle: Optional[Union[int, Sequence]] = None,
        perm: Optional[jax.Array] = None,
        eigval: Optional[jax.Array] = None,
        perm_sign: Optional[jax.Array] = None,
    ):
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
        self._Nparticle = Nparticle

        if perm is None or eigval is None or perm_sign is None:
            self._perm, self._eigval, self._perm_sign = _get_perm(
                self._generator, self._sector, self._generator_sign
            )
        else:
            self._perm, self._eigval, self._perm_sign = perm, eigval, perm_sign
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
        return self._eigval

    @property
    def nsymm(self) -> int:
        return self.eigval.size if self.Z2_inversion == 0 else 2 * self.eigval.size

    @property
    def Z2_inversion(self) -> int:
        return self._Z2_inversion

    @property
    def Nparticle(
        self,
    ) -> Optional[Union[int, Tuple[int, int], List[int], List[Tuple[int, int]]]]:
        return self._Nparticle

    @property
    def basis(self) -> Union[spin_basis_general, spinful_fermion_basis_general]:
        if self._basis is not None:
            return self._basis

        if not np.all(self._perm_sign == 1):
            raise RuntimeError(
                "QuSpin doesn't support non-trivial permmutation sign. This happens "
                "when anti-periodic boundary is used for translation symmetry."
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
                self.nsites, self.Nparticle, pauli=0, make_basis=False, **blocks
            )
        self._basis = basis
        return self._basis
    
    def basis_make(self):
        if not self._is_basis_made:
            self.basis.make()
        self._is_basis_made = True

    @partial(jax.jit, static_argnums=0)
    def get_symm_spins(self, spins: jax.Array) -> jax.Array:
        if spins.ndim > 1:
            raise ValueError(f"Input spins should be 1D, got dimension {spins.ndim}")
        spins = spins[self._perm]
        if self.Z2_inversion != 0:
            spins = jnp.concatenate([spins, -spins], axis=-2)
        return spins

    @partial(jax.jit, static_argnums=0)
    def symmetrize(
        self, inputs: jax.Array, spins: Optional[jax.Array] = None
    ) -> jax.Array:
        if inputs.ndim > 1:
            raise ValueError(f"Input spins should be 1D, got dimension {inputs.ndim}")
        if spins is not None and spins.ndim > 1:
            raise ValueError(f"Input spins should be 1D, got dimension {spins.ndim}")

        if self.is_fermion:
            if spins is None:
                raise RuntimeError(
                    "`symmetrize` can't be performed without input fermion states."
                )
            if self.Nparticle is not None and (
                isinstance(self.Nparticle, tuple) or len(self.Nparticle) == 1
            ):
                Nparticle = np.sum(self.Nparticle).item()
            else:
                Nparticle = None
            sign = _permutation_sign(spins, self._perm, self._perm_sign, Nparticle)
            eigval = sign * self._eigval
        else:
            eigval = self._eigval

        if self.Z2_inversion != 0:
            eigval = jnp.concatenate([eigval, self.Z2_inversion * eigval])
        eigval = (eigval / eigval.size).astype(inputs.dtype)
        return jnp.dot(inputs, eigval)

    def __add__(self, other: Symmetry) -> Symmetry:
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

    def __call__(self, state):
        from ..state import DenseState, Variational

        if isinstance(state, DenseState):
            return state.todense(symm=self)
        elif isinstance(state, Variational):
            if state.symm.nsymm != 1:
                raise RuntimeError(
                    "Symmetry projection can't be performed on a variational state "
                    "already with point-group symmetry."
                )
            if self.nsymm == 1:
                input_fn = lambda s: state.input_fn(s)
                output_fn = lambda x, s: state.output_fn(x, s)
            else:
                input_fn = lambda s: state.input_fn(self.get_symm_spins(s))
                output_fn = lambda x, s: self.symmetrize(state.output_fn(x, s), s)
            new_symm = state.symm + self
            return Variational(
                state.models, None, new_symm, state.max_parallel, input_fn, output_fn
            )
        else:
            return NotImplemented
