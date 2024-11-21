from __future__ import annotations
from typing import Union, Optional, Tuple, Sequence
from jaxtyping import PyTree
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from functools import partial
from ..utils import det, pfaffian, array_set
from ..global_defs import get_sites, get_lattice, get_subkeys, is_default_cpl
from ..nn import RefModel


def _get_Nparticle(Nparticle: Union[None, int, Sequence[int]]) -> int:
    sites = get_sites()
    N = sites.nsites
    if sites.is_fermion:
        if Nparticle is None:
            Nparticle = N
        elif not isinstance(Nparticle, int):
            Nparticle = sum(Nparticle)
    else:
        Nparticle = N
    return Nparticle


def _get_fermion_idx(x: jax.Array, Nparticle: int) -> jax.Array:
    particle = jnp.ones_like(x)
    hole = jnp.zeros_like(x)
    if get_sites().is_fermion:
        x = jnp.where(x > 0, particle, hole)
    else:
        x_up = jnp.where(x > 0, particle, hole)
        x_down = jnp.where(x <= 0, particle, hole)
        x = jnp.concatenate([x_up, x_down])
    idx = jnp.flatnonzero(x, size=Nparticle).astype(jnp.int16)
    return idx


@partial(jax.vmap, in_axes=(0, 0, None))
def _single_electron_parity(n, o, i):
    """Computes parity of electron moves as if only one electron hops"""
    n_i = jnp.sum(n - i > 0)
    o_i = jnp.sum(o - i > 0.5)

    return n_i - o_i + jnp.where(n > o, 1, 0)


def _parity_det(n, o, i):

    sign1 = jnp.power(-1, _single_electron_parity(n, o, i) % 2)

    sign2 = jnp.sign(o[None] - n[:, None])
    sign2 = sign2.at[jnp.tril_indices(len(sign2))].set(
        -1 * sign2[jnp.tril_indices(len(sign2))]
    )
    sign2 = sign2.at[jnp.diag_indices(len(sign2))].set(1)

    return jnp.prod(sign1) * jnp.prod(sign2)


def _parity_pfa(n, o, i):

    sign1 = jnp.power(-1, _single_electron_parity(n, o, i) % 2)

    sign2 = jnp.sign(o[None] - n[:, None])
    sign2 = sign2.at[jnp.diag_indices(len(sign2))].set(1)

    return jnp.prod(sign1) * jnp.prod(sign2)


def _get_changed_inds(flips, nflips, N):
    old_idx = jnp.argwhere(flips < 0, size=nflips // 2).astype(jnp.int16).ravel()
    new_idx = jnp.argwhere(flips > 0, size=nflips // 2).astype(jnp.int16).ravel()

    if not get_sites().is_fermion:
        old_idx2 = jnp.concatenate((old_idx, new_idx + N))
        new_idx2 = jnp.concatenate((new_idx, old_idx + N))
        return old_idx2, new_idx2
    else:
        return old_idx, new_idx


class Determinant(RefModel):
    U: jax.Array
    Nparticle: int
    holomorphic: bool

    def __init__(
        self,
        Nparticle: Union[None, int, Sequence[int]] = None,
        dtype: jnp.dtype = jnp.float64,
    ):
        self.Nparticle = _get_Nparticle(Nparticle)

        N = get_sites().nsites
        shape = (2 * N, self.Nparticle)
        is_dtype_cpl = jnp.issubdtype(dtype, jnp.complexfloating)
        if is_default_cpl() and not is_dtype_cpl:
            shape = (2,) + shape
        # https://www.quora.com/Suppose-A-is-an-NxN-matrix-whose-entries-are-independent-random-variables-with-mean-0-and-variance-%CF%83-2-What-is-the-mean-and-variance-of-X-det-A
        # scale = sqrt(1 / (n!)^(1/n)) ~ sqrt(e/n)
        scale = np.sqrt(np.e / self.Nparticle, dtype=dtype)
        self.U = jr.normal(get_subkeys(), shape, dtype) * scale
        self.holomorphic = is_default_cpl() and is_dtype_cpl

    def __call__(self, x: jax.Array) -> jax.Array:
        U = self.U if self.U.ndim == 2 else jax.lax.complex(self.U[0], self.U[1])
        idx = _get_fermion_idx(x, self.Nparticle)
        return det(U[idx, :])

    def init_internal(self, x: jax.Array) -> PyTree:
        """
        Initialize internal values for given input configurations
        """

        idx = _get_fermion_idx(x, self.Nparticle)
        U = self.U if self.U.ndim == 2 else jax.lax.complex(self.U[0], self.U[1])

        orbs = U[idx, :]

        return {"idx": idx, "inv": jnp.linalg.inv(orbs), "psi": det(orbs)}

    def ref_forward_with_updates(
        self, x: jax.Array, x_old: jax.Array, nflips: int, internal: PyTree
    ) -> Tuple[jax.Array, PyTree]:
        """
        Accelerated forward pass through local updates and internal quantities.
        This function is designed for sampling.

        :return:
            The evaluated wave function and the updated internal values.
        """

        U = self.U if self.U.ndim == 2 else jax.lax.complex(self.U[0], self.U[1])

        occ_idx = internal["idx"]
        old_inv = internal["inv"]
        old_psi = internal["psi"]

        flips = (x - x_old) // 2

        old_idx, new_idx = _get_changed_inds(flips, nflips, len(x))

        update = U[new_idx] - U[old_idx]

        @jax.vmap
        def idx_to_canon(old_idx):
            return jnp.argwhere(old_idx == occ_idx, size=1)

        old_loc = jnp.ravel(idx_to_canon(old_idx))

        eye = jnp.eye(len(update), dtype=old_psi.dtype)
        low_rank_matrix = eye + update @ old_inv[:, old_loc]
        inv_times_update = update @ old_inv
        solve = jnp.linalg.solve(low_rank_matrix, inv_times_update)

        inv = old_inv - old_inv[:, old_loc] @ solve

        idx = occ_idx.at[old_loc].set(new_idx)
        sort = jnp.argsort(idx)

        psi = old_psi * det(low_rank_matrix) * _parity_det(new_idx, old_idx, occ_idx)

        return psi, {"idx": idx[sort], "inv": inv[:, sort], "psi": psi}

    def ref_forward(
        self,
        x: jax.Array,
        x_old: jax.Array,
        nflips: int,
        idx_segment: jax.Array,
        internal: jax.Array,
    ) -> jax.Array:
        """
        Accelerated forward pass through local updates and internal quantities.
        This function is designed for local observables.
        """

        U = self.U if self.U.ndim == 2 else jax.lax.complex(self.U[0], self.U[1])

        x_old = x_old[idx_segment]

        flips = (x - x_old) // 2

        old_idx, new_idx = _get_changed_inds(flips, nflips, len(x))

        occ_idx = internal["idx"][idx_segment]
        old_inv = internal["inv"][idx_segment]
        old_psi = internal["psi"][idx_segment]

        update = U[new_idx] - U[old_idx]

        @jax.vmap
        def idx_to_canon(old_idx):
            return jnp.argwhere(old_idx == occ_idx, size=1)

        old_loc = jnp.ravel(idx_to_canon(old_idx))

        eye = jnp.eye(len(update), dtype=old_psi.dtype)
        low_rank_matrix = eye + update @ old_inv[:, old_loc]

        return old_psi * det(low_rank_matrix) * _parity_det(new_idx, old_idx, occ_idx)

    def rescale(self, maximum: jax.Array) -> Determinant:
        U = self.U / maximum.astype(self.U.dtype) ** (1 / self.Nparticle)
        return eqx.tree_at(lambda tree: tree.U, self, U)


def pfa_eye(rank, dtype):

    a = jnp.zeros([rank, rank], dtype=dtype)
    b = jnp.eye(rank, dtype=dtype)

    return jnp.block([[a, -1 * b], [b, a]])


def det_eye(rank, dtype):

    return jnp.eye(rank, dtype=dtype)


class Pfaffian(RefModel):
    F: jax.Array
    Nparticle: int
    index: jax.Array
    holomorphic: bool

    def __init__(
        self,
        Nparticle: Union[None, int, Sequence[int]] = None,
        sublattice: Optional[tuple] = None,
        dtype: jnp.dtype = jnp.float64,
    ):
        self.Nparticle = _get_Nparticle(Nparticle)

        N = get_sites().nsites
        is_dtype_cpl = jnp.issubdtype(dtype, jnp.complexfloating)
        
        index, nparams = _get_pfaffian_indices(sublattice, 2*N)

        self.index = index
        shape = (nparams,)
        
        if is_default_cpl() and not is_dtype_cpl:
            shape = (2,) + shape
        scale = np.sqrt(np.e / self.Nparticle, dtype=dtype)
        self.F = jr.normal(get_subkeys(), shape, dtype) * scale
        self.holomorphic = is_default_cpl() and is_dtype_cpl

    @property
    def F_full(self) -> jax.Array:
        F = self.F if self.F.ndim == 1 else jax.lax.complex(self.F[0], self.F[1])
        N = get_sites().nsites
        F_full = F[self.index]
        F_full = F_full - F_full.T

        return F_full

    def __call__(self, x: jax.Array) -> jax.Array:
        idx = _get_fermion_idx(x, self.Nparticle)
        return pfaffian(self.F_full[idx, :][:, idx])

    def rescale(self, maximum: jax.Array) -> Pfaffian:
        F = self.F / maximum.astype(self.F.dtype) ** (2 / self.Nparticle)
        return eqx.tree_at(lambda tree: tree.F, self, F)

    def init_internal(self, x: jax.Array) -> PyTree:
        """
        Initialize internal values for given input configurations
        """
        idx = _get_fermion_idx(x, self.Nparticle)
        orbs = self.F_full[idx, :][:, idx]

        return {"idx": idx, "inv": jnp.linalg.inv(orbs), "psi": pfaffian(orbs)}

    def ref_forward_with_updates(
        self, x: jax.Array, x_old: jax.Array, nflips: int, internal: PyTree
    ) -> Tuple[jax.Array, PyTree]:
        """
        Accelerated forward pass through local updates and internal quantities.
        This function is designed for sampling.

        :return:
            The evaluated wave function and the updated internal values.
        """
        occ_idx = internal["idx"]
        old_inv = internal["inv"]
        old_psi = internal["psi"]

        flips = (x - x_old) // 2

        old_idx, new_idx = _get_changed_inds(flips, nflips, len(x))

        @jax.vmap
        def idx_to_canon(old_idx):
            return jnp.argwhere(old_idx == occ_idx, size=1)

        old_loc = jnp.ravel(idx_to_canon(old_idx))

        F_full = self.F_full
        update = F_full[new_idx][:, occ_idx] - F_full[old_idx][:, occ_idx]

        mat = jnp.tril(F_full[new_idx][:, new_idx] - F_full[old_idx][:, old_idx])

        update = array_set(update.T, mat.T, old_loc).T

        if get_sites().is_fermion:
            eye = pfa_eye(nflips // 2, F_full.dtype)
        else:
            eye = pfa_eye(nflips, F_full.dtype)

        mat11 = update @ old_inv @ update.T
        mat21 = update @ old_inv[:, old_loc]
        mat22 = old_inv[old_loc][:, old_loc]

        low_rank_matrix = -1 * eye + jnp.block([[mat11, mat21], [-1 * mat21.T, mat22]])

        parity = _parity_pfa(new_idx, old_idx, occ_idx)
        psi = old_psi * pfaffian(low_rank_matrix) * parity

        inv_times_update = jnp.concatenate((update @ old_inv, old_inv[old_loc]), 0)

        solve = jnp.linalg.solve(low_rank_matrix, inv_times_update)
        inv = old_inv + inv_times_update.T @ solve

        idx = occ_idx.at[old_loc].set(new_idx)

        sort = jnp.argsort(idx)

        return psi, {"idx": idx[sort], "inv": inv[sort][:, sort], "psi": psi}

    def ref_forward(
        self,
        x: jax.Array,
        x_old: jax.Array,
        nflips: int,
        idx_segment: jax.Array,
        internal: jax.Array,
    ) -> jax.Array:
        """
        Accelerated forward pass through local updates and internal quantities.
        This function is designed for local observables.
        """
        occ_idx = internal["idx"][idx_segment]
        old_inv = internal["inv"][idx_segment]
        old_psi = internal["psi"][idx_segment]
        x_old = x_old[idx_segment]

        flips = (x - x_old) // 2

        old_idx, new_idx = _get_changed_inds(flips, nflips, len(x))

        @jax.vmap
        def idx_to_canon(old_idx):
            return jnp.argwhere(old_idx == occ_idx, size=1)

        old_loc = jnp.ravel(idx_to_canon(old_idx))

        F_full = self.F_full
        update = F_full[new_idx][:, occ_idx] - F_full[old_idx][:, occ_idx]

        mat = jnp.tril(F_full[new_idx][:, new_idx] - F_full[old_idx][:, old_idx])

        update = array_set(update.T, mat.T, old_loc).T

        if get_sites().is_fermion:
            eye = pfa_eye(nflips // 2, F_full.dtype)
        else:
            eye = pfa_eye(nflips, F_full.dtype)

        mat11 = update @ old_inv @ update.T
        mat21 = update @ old_inv[:, old_loc]
        mat22 = old_inv[old_loc][:, old_loc]

        low_rank_matrix = -1 * eye + jnp.block([[mat11, mat21], [-1 * mat21.T, mat22]])

        parity = _parity_pfa(new_idx, old_idx, occ_idx)
        return old_psi * pfaffian(low_rank_matrix) * parity

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
            nparams = np.prod(sublattice) * N
        else:
            nparams = np.prod(sublattice) * (N - 1)
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

def _get_pfaffian_indices(sublattice, N):
    if sublattice is None:
        nparams = N*(N-1)//2
        index = np.zeros((N, N), dtype=np.uint32)
        index[np.triu_indices(N,k=1)] = np.arange(nparams)
    else:
        lattice = get_lattice()
        c = lattice.shape[0]
        
        ns = N//jnp.prod(jnp.array(lattice.shape))

        lattice_shape = (ns,) + lattice.shape
        sublattice = (ns,c,) + sublattice

        index = np.ones((N, N), dtype=np.uint32)
        index[np.tril_indices(N)] = 0

        #nparams = np.prod(sublattice) * (N-1) 

        index = index.reshape(lattice_shape + lattice_shape)
        for axis, lsub in enumerate(sublattice):
            if axis > 1:
                index = np.take(index, range(lsub), axis)

        nparams = int(jnp.sum(index))
        index[index == 1] = np.arange(nparams)

        for axis, (l, lsub) in enumerate(zip(lattice_shape[2:], sublattice[2:])):
            mul = l // lsub
            translated_axis = lattice.ndim + axis + 4
            index = [np.roll(index, i * lsub, translated_axis) for i in range(mul)]
            index = np.concatenate(index, axis=axis + 2)
        index = index.reshape(N, N)

    return index, nparams

class PairProduct(RefModel):
    F: jax.Array
    Nparticle: int
    index: jax.Array
    holomorphic: bool

    def __init__(
        self,
        Nparticle: Union[None, int, Sequence[int]] = None,
        sublattice: Optional[tuple] = None,
        dtype: jnp.dtype = jnp.float64,
    ):

        sites = get_sites()
        N = sites.nsites
        if N % 2 > 0:
            raise RuntimeError("`PairProductSpin` only supports even sites.")

        if sites.is_fermion:
            if Nparticle is None:
                self.Nparticle = N
            elif not isinstance(Nparticle, int):
                self.Nparticle = sum(Nparticle)
            else:
                self.Nparticle = Nparticle
        else:
            self.Nparticle = N

        index, nparams = _get_pair_product_indices(sublattice, N)

        self.index = index
        shape = (nparams,)

        is_dtype_cpl = jnp.issubdtype(dtype, jnp.complexfloating)
        if is_default_cpl() and not is_dtype_cpl:
            shape = (2,) + shape
        scale = np.sqrt(2 * np.e / N, dtype=dtype)
        self.F = jr.normal(get_subkeys(), shape, dtype) * scale
        self.holomorphic = is_default_cpl() and is_dtype_cpl

    @eqx.filter_jit
    def init_internal(self, x: jax.Array) -> PyTree:
        """
        Initialize internal values for given input configurations
        """

        F = self.F if self.F.ndim == 1 else jax.lax.complex(self.F[0], self.F[1])
        F_full = F[self.index]
        N = get_sites().nsites
        idx = _get_fermion_idx(x, N)
        F_full = F_full[idx[: N // 2], :][:, idx[N // 2 :] - N]

        return {"idx": idx, "inv": jnp.linalg.inv(F_full), "psi": det(F_full)}

    def __call__(self, x: jax.Array) -> jax.Array:
        F = self.F if self.F.ndim == 1 else jax.lax.complex(self.F[0], self.F[1])
        F_full = F[self.index]

        N = get_sites().nsites
        idx = _get_fermion_idx(x, N)
        F_full = F_full[idx[: N // 2], :][:, idx[N // 2 :] - N]
        return det(F_full)

    def rescale(self, maximum: jax.Array) -> PairProduct:
        N = get_sites().nsites
        F = self.F / maximum.astype(self.F.dtype) ** (2 / N)
        return eqx.tree_at(lambda tree: tree.F, self, F)

    def ref_forward_with_updates(
        self,
        x: jax.Array,
        x_old: jax.Array,
        nflips: int,
        internal: jax.Array,
    ) -> jax.Array:
        """
        Accelerated forward pass through local updates and internal quantities.
        This function is designed for local observables.
        """

        occ_idx = internal["idx"]
        old_inv = internal["inv"]
        old_psi = internal["psi"]
        flips = (x - x_old) // 2

        F = self.F if self.F.ndim == 1 else jax.lax.complex(self.F[0], self.F[1])
        F_full = F[self.index]
        N = get_sites().nsites

        old_idx, new_idx = _get_changed_inds(flips, nflips, len(x))

        old_idx_down, old_idx_up = jnp.split(old_idx, 2)
        new_idx_down, new_idx_up = jnp.split(new_idx, 2)
        occ_idx_down, occ_idx_up = jnp.split(occ_idx, 2)

        old_idx_up = old_idx_up - N
        new_idx_up = new_idx_up - N
        occ_idx_up = occ_idx_up - N

        @partial(jax.vmap, in_axes=(0, None))
        def idx_to_canon(old_idx, occ_idx):
            return jnp.argwhere(old_idx == occ_idx, size=1)

        old_loc_down = jnp.ravel(idx_to_canon(old_idx_down, occ_idx_down))
        old_loc_up = jnp.ravel(idx_to_canon(old_idx_up, occ_idx_up))

        update_lhs = (
            F_full[new_idx_down][:, occ_idx_up] - F_full[old_idx_down][:, occ_idx_up]
        )
        update_rhs = (
            F_full[occ_idx_down][:, new_idx_up] - F_full[occ_idx_down][:, old_idx_up]
        )

        mat = F_full[new_idx_down][:, new_idx_up] - F_full[old_idx_down][:, old_idx_up]

        update_lhs = array_set(update_lhs.T, 0, old_loc_up).T
        update_rhs = array_set(update_rhs, mat, old_loc_down)

        if get_sites().is_fermion:
            eye = det_eye(nflips // 2, F_full.dtype)
        else:
            eye = det_eye(nflips, F_full.dtype)

        mat11 = update_lhs @ old_inv[:, old_loc_down]
        mat21 = update_lhs @ old_inv @ update_rhs
        mat12 = old_inv[old_loc_up][:, old_loc_down]
        mat22 = old_inv[old_loc_up] @ update_rhs

        low_rank_matrix = eye + jnp.block([[mat11, mat21], [mat12, mat22]])

        psi = old_psi * det(low_rank_matrix) * _parity_det(new_idx, old_idx, occ_idx)

        lhs = jnp.concatenate((update_lhs @ old_inv, old_inv[old_loc_up]), 0)
        rhs = jnp.concatenate((old_inv[:, old_loc_down], old_inv @ update_rhs), 1)

        inv = old_inv - rhs @ jnp.linalg.solve(low_rank_matrix, lhs)

        idx_down = occ_idx_down.at[old_loc_down].set(new_idx_down)
        idx_up = occ_idx_up.at[old_loc_up].set(new_idx_up)

        sort_down = jnp.argsort(idx_down)
        sort_up = jnp.argsort(idx_up)

        internal = {
            "idx": jnp.concatenate((idx_down[sort_down], idx_up[sort_up] + N)),
            "inv": inv[sort_up][:, sort_down],
            "psi": psi,
        }
        return psi, internal

    def ref_forward(
        self,
        x: jax.Array,
        x_old: jax.Array,
        nflips: int,
        idx_segment: jax.Array,
        internal: jax.Array,
    ) -> jax.Array:
        """
        Accelerated forward pass through local updates and internal quantities.
        This function is designed for local observables.
        """

        occ_idx = internal["idx"][idx_segment]
        old_inv = internal["inv"][idx_segment]
        old_psi = internal["psi"][idx_segment]
        x_old = x_old[idx_segment]
        flips = (x - x_old) // 2

        F = self.F if self.F.ndim == 1 else jax.lax.complex(self.F[0], self.F[1])
        F_full = F[self.index]
        N = get_sites().nsites

        old_idx, new_idx = _get_changed_inds(flips, nflips, len(x))

        old_idx_down, old_idx_up = jnp.split(old_idx, 2)
        new_idx_down, new_idx_up = jnp.split(new_idx, 2)
        occ_idx_down, occ_idx_up = jnp.split(occ_idx, 2)

        old_idx_up = old_idx_up - N
        new_idx_up = new_idx_up - N
        occ_idx_up = occ_idx_up - N

        @partial(jax.vmap, in_axes=(0, None))
        def idx_to_canon(old_idx, occ_idx):
            return jnp.argwhere(old_idx == occ_idx, size=1)

        old_loc_down = jnp.ravel(idx_to_canon(old_idx_down, occ_idx_down))
        old_loc_up = jnp.ravel(idx_to_canon(old_idx_up, occ_idx_up))

        update_lhs = (
            F_full[new_idx_down][:, occ_idx_up] - F_full[old_idx_down][:, occ_idx_up]
        )
        update_rhs = (
            F_full[occ_idx_down][:, new_idx_up] - F_full[occ_idx_down][:, old_idx_up]
        )

        mat = F_full[new_idx_down][:, new_idx_up] - F_full[old_idx_down][:, old_idx_up]

        update_lhs = array_set(update_lhs.T, 0, old_loc_up).T
        update_rhs = array_set(update_rhs, mat, old_loc_down)

        if get_sites().is_fermion:
            eye = det_eye(nflips // 2, F_full.dtype)
        else:
            eye = det_eye(nflips, F_full.dtype)

        mat11 = update_lhs @ old_inv[:, old_loc_down]
        mat21 = update_lhs @ old_inv @ update_rhs
        mat12 = old_inv[old_loc_up][:, old_loc_down]
        mat22 = old_inv[old_loc_up] @ update_rhs

        low_rank_matrix = eye + jnp.block([[mat11, mat21], [mat12, mat22]])

        return old_psi * det(low_rank_matrix) * _parity_det(new_idx, old_idx, occ_idx)


# class HiddenDet(eqx.Module):
#     net: eqx.Module
#     net_symm: Symmetry = eqx.field(static=True)
#     U: jax.Array
#     Nvisible: int
#     Nhidden: int
#     holomorphic: bool

#     def __init__(
#         self,
#         net: eqx.Module,
#         net_symm: Symmetry = None,
#         Nvisible: Optional[int] = None,
#         Nhidden: Optional[int] = None,
#         dtype: jnp.dtype = jnp.float32,
#     ):
#         self.net = net
#         self.net_symm = Identity() if net_symm is None else net_symm
#         N = get_sites().nsites
#         self.Nvisible = N if Nvisible is None else Nvisible
#         self.Nhidden = self.Nvisible if Nhidden is None else Nhidden
#         Ntotal = self.Nvisible + self.Nhidden
#         scale = np.sqrt((np.e / Ntotal) ** (Ntotal / self.Nvisible), dtype=dtype)
#         is_dtype_cpl = jnp.issubdtype(dtype, jnp.complexfloating)
#         shape = (2 * N, Ntotal)
#         if is_default_cpl() and not is_dtype_cpl:
#             shape = (2,) + shape
#         self.U = jr.normal(get_subkeys(), shape, dtype) * scale
#         self.holomorphic = is_default_cpl() and is_dtype_cpl

#     def get_Uvisible(self, x: jax.Array):
#         U = self.U if self.U.ndim == 2 else jax.lax.complex(self.U[0], self.U[1])
#         idx = _get_fermion_idx(x, self.Nvisible)
#         return U[idx, :]

#     def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
#         shape = (self.Nhidden, self.Nvisible + self.Nhidden, self.net_symm.nsymm)
#         Uhidden = self.net(x, key=key).reshape(shape)
#         Uhidden = jnp.moveaxis(Uhidden, -1, 0)
#         x_symm = self.net_symm.get_symm_spins(x)
#         Uvisible = jax.vmap(self.get_Uvisible)(x_symm)
#         U = jnp.concatenate([Uvisible, Uhidden], axis=1)
#         out = det(U)
#         out = self.net_symm.symmetrize(out, x)
#         return out

#     def rescale(self, maximum: jax.Array) -> Determinant:
#         U = self.U / maximum.astype(self.U.dtype) ** (1 / self.Nvisible)
#         return eqx.tree_at(lambda tree: tree.U, self, U)


# class HiddenDet(eqx.Module):
#     net: eqx.Module
#     U: jax.Array
#     Nvisible: int
#     Nhidden: int

#     def __init__(
#         self,
#         net: eqx.Module,
#         Nvisible: Optional[int] = None,
#         Nhidden: Optional[int] = None,
#         dtype: jnp.dtype = jnp.float32,
#     ):
#         self.net = net
#         N = get_sites().nsites
#         self.Nvisible = N if Nvisible is None else Nvisible
#         self.Nhidden = self.Nvisible if Nhidden is None else Nhidden
#         Ntotal = self.Nvisible + self.Nhidden
#         # scale = np.sqrt((np.e / Ntotal) ** (Ntotal / self.Nvisible), dtype=dtype)
#         is_dtype_cpl = jnp.issubdtype(dtype, jnp.complexfloating)
#         shape = (2 * N, Ntotal)
#         if is_default_cpl() and not is_dtype_cpl:
#             shape = (2,) + shape
#         self.U = jr.normal(get_subkeys(), shape, dtype)  # * scale

#     def get_Uvisible(self, x: jax.Array):
#         U = self.U if self.U.ndim == 2 else jax.lax.complex(self.U[0], self.U[1])
#         idx = _get_fermion_idx(x, self.Nvisible)
#         return U[idx, :]

#     def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
#         Ntotal = self.Nvisible + self.Nhidden
#         shape = (self.Nhidden, Ntotal)
#         Uhidden = self.net(x, key=key).reshape(shape)
#         Uvisible = self.get_Uvisible(x)
#         U = jnp.concatenate([Uvisible, Uhidden], axis=0)
#         U *= np.sqrt(np.e / Ntotal, dtype=U.dtype)
#         return det(U)

# def rescale(self, maximum: jax.Array) -> Determinant:
#     U = self.U / maximum.astype(self.U.dtype) ** (1 / self.Nvisible)
#     return eqx.tree_at(lambda tree: tree.U, self, U)
