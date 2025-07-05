from __future__ import annotations
from typing import Optional, Tuple, NamedTuple, Union
from jaxtyping import PyTree
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import lrux
from ..global_defs import (
    get_sites,
    get_lattice,
    get_subkeys,
    is_default_cpl,
    PARTICLE_TYPE,
)
from ..nn import RefModel
from ..utils import fermion_idx, changed_inds, permute_sign, array_set


class MF_Internal(NamedTuple):
    idx: jax.Array
    inv: Union[jax.Array, lrux.DetCarrier, lrux.PfCarrier]
    psi: jax.Array


class Determinant(RefModel):
    U: jax.Array
    holomorphic: bool

    def __init__(
        self, U: Optional[jax.Array] = None, dtype: Optional[jnp.dtype] = None
    ):
        sites = get_sites()
        if sites.Ntotal is None:
            raise ValueError("Determinant should have a fixed amount of particles.")

        if sites.particle_type == PARTICLE_TYPE.spin:
            raise NotImplementedError(
                "Determinant is not yet implemented for spin systems."
            )

        shape = (2 * sites.N, sites.Ntotal)
        is_dtype_cpl = jnp.issubdtype(dtype, jnp.complexfloating)
        if is_default_cpl() and not is_dtype_cpl:
            shape = (2,) + shape

        if U is None:
            # https://www.quora.com/Suppose-A-is-an-NxN-matrix-whose-entries-are-independent-random-variables-with-mean-0-and-variance-%CF%83-2-What-is-the-mean-and-variance-of-X-det-A
            # scale = sqrt(1 / (n!)^(1/n)) ~ sqrt(e/n)
            if dtype is None:
                dtype = jnp.float64
            scale = np.sqrt(np.e / sites.Ntotal, dtype=dtype)
            self.U = jr.normal(get_subkeys(), shape, dtype) * scale
        else:
            if U.shape != shape:
                raise ValueError(f"Expected U to have shape {shape}, but got {U.shape}")
            if dtype is not None:
                U = U.astype(dtype)
            self.U = U

        self.holomorphic = is_default_cpl() and is_dtype_cpl

    @property
    def U_full(self) -> jax.Array:
        """
        Returns the full orbital matrix U.
        """
        return self.U if self.U.ndim == 2 else jax.lax.complex(self.U[0], self.U[1])

    def __call__(self, s: jax.Array) -> jax.Array:
        idx = fermion_idx(s)
        return jnp.linalg.det(self.U_full[idx, :])

    def init_internal(self, s: jax.Array) -> PyTree:
        """
        Initialize internal values for given input configurations
        """
        idx = fermion_idx(s)
        orbs = self.U_full[idx, :]
        return MF_Internal(idx=idx, inv=jnp.linalg.inv(orbs), psi=jnp.linalg.det(orbs))

    def ref_forward(
        self,
        s: jax.Array,
        s_old: jax.Array,
        nflips: int,
        internal: MF_Internal,
        return_update: bool = False,
    ) -> Union[jax.Array, Tuple[jax.Array, MF_Internal]]:
        """
        Accelerated forward pass through local updates and internal quantities.
        """
        nhops = nflips // 2 if get_sites().is_fermion else nflips
        idx_annihilate, idx_create = changed_inds(s, s_old, nhops)
        idx = internal.idx
        sign = permute_sign(idx, idx_annihilate, idx_create)
        U_full = self.U_full
        row_update = U_full[idx_create] - U_full[idx_annihilate]
        is_updated = jnp.isin(idx, idx_annihilate)
        row_update_idx = jnp.flatnonzero(is_updated, size=nhops, fill_value=idx.size)

        if return_update:
            idx = idx.at[row_update_idx].set(idx_create)
            ratio, inv = lrux.det_lru(internal.inv, row_update.T, row_update_idx, True)
            psi = internal.psi * ratio * sign
            internal = MF_Internal(idx, inv, psi)
            return psi, internal
        else:
            ratio = lrux.det_lru(internal.inv, row_update.T, row_update_idx, False)
            psi = internal.psi * ratio * sign
            return psi

    def rescale(self, maximum: jax.Array) -> Determinant:
        U = self.U / maximum.astype(self.U.dtype) ** (1 / get_sites().Ntotal)
        return eqx.tree_at(lambda tree: tree.U, self, U)


def _get_pair_product_indices(sublattice, N):
    if sublattice is None:
        if get_sites().is_fermion:
            nparams = N**2
            index = np.arange(nparams).reshape(N, N)
        else:
            nparams = N * (N - 1)
            index = np.zeros((N, N), dtype=np.uint32)
            index[~np.eye(N, dtype=np.bool_)] = np.arange(nparams)
    else:
        lattice = get_lattice()
        c = lattice.shape[0]
        sublattice = (c,) + sublattice

        index = np.ones((N, N), dtype=np.uint32)

        if get_sites().is_fermion:
            nparams = np.prod(sublattice).item() * N
        else:
            nparams = np.prod(sublattice).item() * (N - 1)
            index[np.eye(N, dtype=np.bool_)] = 0

        index = index.reshape(lattice.shape + lattice.shape)
        for axis, lsub in enumerate(sublattice):
            if axis > 0:
                index = np.take(index, range(lsub), axis)
        index[index == 1] = np.arange(nparams)

        for axis, (l, lsub) in enumerate(zip(lattice.shape[1:], sublattice[1:])):
            mul = l // lsub
            translated_axis = lattice.ndim + axis + 2
            index = [np.roll(index, i * lsub, translated_axis) for i in range(mul)]
            index = np.concatenate(index, axis=axis + 1)
        index = index.reshape(N, N)

    return index, nparams


class PfSinglet(RefModel):
    F: jax.Array
    index: jax.Array
    holomorphic: bool
    sublattice: Optional[tuple] = eqx.field(static=True)

    def __init__(
        self,
        F: Optional[jax.Array] = None,
        sublattice: Optional[tuple] = None,
        dtype: jnp.dtype = None,
    ):
        sites = get_sites()

        N = sites.N
        Nparticle = sites.Nparticle
        if (not isinstance(Nparticle, tuple)) or Nparticle[0] != Nparticle[1]:
            raise ValueError(
                "PfSinglet only works for equal number of spin-up and spin-down particles."
                f"Got Nparticle={Nparticle}."
            )

        index, nparams = _get_pair_product_indices(sublattice, N)
        self.sublattice = sublattice
        self.index = index
        shape = (nparams,)
        is_dtype_cpl = jnp.issubdtype(dtype, jnp.complexfloating)
        if is_default_cpl() and not is_dtype_cpl:
            shape = (2,) + shape

        if F is None:
            if dtype is None:
                dtype = jnp.float64
            scale = np.sqrt(2 * np.e / Nparticle[0], dtype=dtype)
            self.F = jr.normal(get_subkeys(), shape, dtype) * scale
        else:
            if F.shape != shape:
                raise ValueError(f"Expected F to have shape {shape}, but got {F.shape}")
            if dtype is not None:
                F = F.astype(dtype)
            self.F = F

        self.holomorphic = is_default_cpl() and is_dtype_cpl

    @property
    def F_full(self) -> jax.Array:
        F = self.F if self.F.ndim == 1 else jax.lax.complex(self.F[0], self.F[1])
        return F[self.index]

    def __call__(self, s: jax.Array) -> jax.Array:
        Nup = get_sites().Nparticle[0]
        idx = fermion_idx(s)
        idx = idx.at[Nup:].add(-s.size)
        F_full = self.F_full[idx[:Nup], :][:, idx[Nup:]]
        return jnp.linalg.det(F_full)

    @eqx.filter_jit
    def init_internal(self, s: jax.Array) -> PyTree:
        """
        Initialize internal values for given input configurations
        """
        sites = get_sites()

        if sites.particle_type == PARTICLE_TYPE.spinful_fermion:
            raise NotImplementedError(
                "Low-rank update is not implemented for `PfSinglet` with spinful fermions,"
                "because the number of spin-up and spin-down hoppings is not fixed."
            )

        Nup = sites.Nparticle[0]
        idx = fermion_idx(s)
        idx = idx.at[Nup:].add(-s.size)
        F_full = self.F_full[idx[:Nup], :][:, idx[Nup:]]
        inv = jnp.linalg.inv(F_full)
        psi = jnp.linalg.det(F_full)

        return MF_Internal(idx, inv, psi)

    def ref_forward(
        self,
        s: jax.Array,
        s_old: jax.Array,
        nflips: int,
        internal: MF_Internal,
        return_update: bool = False,
    ) -> Union[jax.Array, Tuple[jax.Array, MF_Internal]]:
        """
        Accelerated forward pass through local updates and internal quantities.
        """
        sites = get_sites()

        if sites.particle_type == PARTICLE_TYPE.spinful_fermion:
            raise NotImplementedError(
                "Low-rank update is not implemented for `PfSinglet` with spinful fermions,"
                "because the number of spin-up and spin-down hoppings is not fixed."
            )

        Nup = sites.Nparticle[0]
        idx_flip_down, idx_flip_up = changed_inds(s, s_old, nflips)
        idx = internal.idx
        sign_up = permute_sign(idx[:Nup], idx_flip_down, idx_flip_up)
        sign_down = permute_sign(idx[Nup:], idx_flip_up, idx_flip_down)
        F = self.F_full

        is_updated = jnp.isin(idx, idx_flip_down)
        row_update_idx = jnp.flatnonzero(is_updated, size=nflips, fill_value=idx.size)
        is_updated = jnp.isin(idx, idx_flip_up)
        col_update_idx = jnp.flatnonzero(is_updated, size=nflips, fill_value=idx.size)

        row_update = F[idx_flip_up][:, idx[Nup:]] - F[idx_flip_down][:, idx[Nup:]]
        idx = idx.at[row_update_idx].set(idx_flip_up)
        idx = idx.at[col_update_idx].set(idx_flip_down)
        col_update = F[idx[:Nup]][:, idx_flip_down] - F[idx[:Nup]][:, idx_flip_up]

        # See https://chenao-phys.github.io/lrux/lrux.det_lru.html#lrux.det_lru
        u = (row_update.T, col_update_idx - Nup)
        v = (col_update, row_update_idx)

        if return_update:
            ratio, inv = lrux.det_lru(internal.inv, u, v, True)
            psi = internal.psi * ratio * sign_up * sign_down
            internal = MF_Internal(idx, inv, psi)
            return psi, internal
        else:
            ratio = lrux.det_lru(internal.inv, u, v, False)
            psi = internal.psi * ratio * sign_up * sign_down
            return psi

    def rescale(self, maximum: jax.Array) -> PfSinglet:
        Nup = get_sites().Nparticle[0]
        F = self.F / maximum.astype(self.F.dtype) ** (2 / Nup)
        return eqx.tree_at(lambda tree: tree.F, self, F)


def _get_pfaffian_indices(sublattice, N):
    if sublattice is None:
        nparams = N * (N - 1) // 2
        index = np.zeros((N, N), dtype=np.uint32)
        index[np.triu_indices(N, k=1)] = np.arange(nparams)
    else:
        lattice = get_lattice()
        c = lattice.shape[0]

        ns = N // np.prod(lattice.shape)
        lattice_shape = (ns,) + lattice.shape
        sublattice = (ns, c) + sublattice

        index = np.ones((N, N), dtype=np.uint32)

        index = index.reshape(lattice_shape + lattice_shape)
        for axis, lsub in enumerate(sublattice):
            if axis > 1:
                index = np.take(index, range(lsub), axis)

        nparams = np.sum(index).item()
        index[index == 1] = np.arange(nparams)

        for axis, (l, lsub) in enumerate(zip(lattice_shape[2:], sublattice[2:])):
            mul = l // lsub
            translated_axis = lattice.ndim + axis + 4

            index = [np.roll(index, i * lsub, translated_axis) for i in range(mul)]
            index = np.concatenate(index, axis=axis + 2)

        index = index.reshape(N, N)

    return index, nparams


class Pfaffian(RefModel):
    F: jax.Array
    index: jax.Array
    holomorphic: bool
    sublattice: Optional[tuple] = eqx.field(static=True)

    def __init__(
        self,
        F: Optional[jax.Array] = None,
        sublattice: Optional[tuple] = None,
        dtype: Optional[jnp.dtype] = None,
    ):
        sites = get_sites()
        M = 2 * sites.N if sites.is_spinful else sites.N
        Ntotal = sites.Ntotal
        if Ntotal is None or Ntotal % 2 != 0:
            raise ValueError(
                f"Pfaffian only works for even number of particles, got {Ntotal}."
            )

        index, nparams = _get_pfaffian_indices(sublattice, M)
        self.sublattice = sublattice
        self.index = index
        shape = (nparams,)
        is_dtype_cpl = jnp.issubdtype(dtype, jnp.complexfloating)
        if is_default_cpl() and not is_dtype_cpl:
            shape = (2,) + shape

        if F is None:
            if dtype is None:
                dtype = jnp.float64
            scale = np.sqrt(np.e / Ntotal, dtype=dtype)
            self.F = jr.normal(get_subkeys(), shape, dtype) * scale
        else:
            if F.shape != shape:
                raise ValueError(f"Expected F to have shape {shape}, but got {F.shape}")
            if dtype is not None:
                F = F.astype(dtype)
            self.F = F

        self.holomorphic = is_default_cpl() and is_dtype_cpl

    @property
    def F_full(self) -> jax.Array:
        F = self.F if self.F.ndim == 1 else jax.lax.complex(self.F[0], self.F[1])
        F_full = F[self.index]
        F_full = (F_full - F_full.T) / 2
        return F_full

    def __call__(self, x: jax.Array) -> jax.Array:
        idx = fermion_idx(x)
        return lrux.pf(self.F_full[idx, :][:, idx])

    def init_internal(self, x: jax.Array) -> PyTree:
        """
        Initialize internal values for given input configurations
        """
        idx = fermion_idx(x)
        orbs = self.F_full[idx, :][:, idx]
        inv = jnp.linalg.inv(orbs)
        inv = (inv - inv.T) / 2  # Ensure antisymmetry
        psi = lrux.pf(orbs)
        return MF_Internal(idx, inv, psi)

    def ref_forward(
        self,
        s: jax.Array,
        s_old: jax.Array,
        nflips: int,
        internal: MF_Internal,
        return_update: bool = False,
    ) -> Union[jax.Array, Tuple[jax.Array, MF_Internal]]:
        """
        Accelerated forward pass through local updates and internal quantities.
        """
        nhops = nflips // 2 if get_sites().is_fermion else nflips
        idx_annihilate, idx_create = changed_inds(s, s_old, nhops)
        idx = internal.idx
        sign = permute_sign(idx, idx_annihilate, idx_create)
        is_updated = jnp.isin(idx, idx_annihilate)
        row_update_idx = jnp.flatnonzero(is_updated, size=nhops, fill_value=idx.size)

        F = self.F_full
        row_update = F[idx_create][:, idx] - F[idx_annihilate][:, idx]
        overlap = F[idx_create][:, idx_create] - F[idx_annihilate][:, idx_annihilate]
        row_update = array_set(row_update.T, jnp.triu(overlap).T, row_update_idx)
        u = (row_update, row_update_idx)

        if return_update:
            idx = idx.at[row_update_idx].set(idx_create)
            ratio, inv = lrux.pf_lru(internal.inv, u, True)
            psi = internal.psi * ratio * sign
            internal = MF_Internal(idx, inv, psi)
            return psi, internal
        else:
            ratio = lrux.pf_lru(internal.inv, u, False)
            psi = internal.psi * ratio * sign
            return psi
