from __future__ import annotations
from typing import Union, Optional, Tuple, Sequence
from jaxtyping import PyTree
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from functools import partial
from ..symmetry import Symmetry
from ..nn import Sequential, RefModel, RawInputLayer
from ..utils import det, pfaffian, array_set
from ..global_defs import get_sites, get_lattice, get_subkeys, is_default_cpl


def _get_fermion_idx(x: jax.Array, Nparticle: int) -> jax.Array:
    particle = jnp.ones_like(x)
    hole = jnp.zeros_like(x)
    if get_sites().is_fermion:
        x = jnp.where(x > 0, particle, hole)
    else:
        x_up = jnp.where(x > 0, particle, hole)
        x_down = jnp.where(x <= 0, particle, hole)
        x = jnp.concatenate([x_up, x_down])
    idx = jnp.flatnonzero(x, size=Nparticle)
    return idx


# Computes parity of electron moves as if only one electron hops
@partial(jax.vmap, in_axes=(0, 0, None))
def single_electron_parity(n, o, i):
    n_i = jnp.sum(n - i > 0)
    o_i = jnp.sum(o - i > 0.5)

    return n_i - o_i + jnp.where(n > o, 1, 0)


def parity_det(n, o, i):

    sign1 = jnp.power(-1, single_electron_parity(n, o, i) % 2)

    sign2 = jnp.sign(o[None] - n[:, None])
    sign2 = sign2.at[jnp.tril_indices(len(sign2))].set(
        -1 * sign2[jnp.tril_indices(len(sign2))]
    )
    sign2 = sign2.at[jnp.diag_indices(len(sign2))].set(1)

    return jnp.prod(sign1) * jnp.prod(sign2)


def parity_pfa(n, o, i):

    sign1 = jnp.power(-1, single_electron_parity(n, o, i) % 2)

    sign2 = jnp.sign(o[None] - n[:, None])
    sign2 = sign2.at[jnp.diag_indices(len(sign2))].set(1)

    return jnp.prod(sign1) * jnp.prod(sign2)


def get_changed_inds(flips, nflips, N):
    old_idx = jnp.argwhere(flips < -0.5, size=nflips // 2).ravel()
    new_idx = jnp.argwhere(flips > 0.5, size=nflips // 2).ravel()

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
        sites = get_sites()
        N = sites.nsites
        if sites.is_fermion:
            if Nparticle is None:
                self.Nparticle = N
            elif not isinstance(Nparticle, int):
                self.Nparticle = sum(Nparticle)
            else:
                self.Nparticle = Nparticle
        else:
            self.Nparticle = N

        is_dtype_cpl = jnp.issubdtype(dtype, jnp.complexfloating)
        shape = (2 * N, self.Nparticle)
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

    @eqx.filter_jit
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

        flips = (x - x_old) / 2

        old_idx, new_idx = get_changed_inds(flips, nflips, len(x))

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

        psi = old_psi * det(low_rank_matrix) * parity_det(new_idx, old_idx, occ_idx)

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

        flips = (x - x_old) / 2

        old_idx, new_idx = get_changed_inds(flips, nflips, len(x))

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

        return old_psi * det(low_rank_matrix) * parity_det(new_idx, old_idx, occ_idx)

    def rescale(self, maximum: jax.Array) -> Determinant:
        U = self.U / maximum.astype(self.U.dtype) ** (1 / self.Nparticle)
        return eqx.tree_at(lambda tree: tree.U, self, U)


def pfa_eye(rank, dtype):
    if not get_sites().is_fermion:
        rank = 2 * rank

    a = jnp.zeros([rank, rank], dtype=dtype)
    b = jnp.eye(rank, dtype=dtype)

    return jnp.block([[a, -1 * b], [b, a]])


def det_eye(rank, dtype):
    if not get_sites().is_fermion:
        rank = 2 * rank

    return jnp.eye(rank, dtype=dtype)


class Pfaffian(RefModel):
    F: jax.Array
    Nparticle: int
    holomorphic: bool

    def __init__(
        self,
        Nparticle: Union[None, int, Sequence[int]] = None,
        dtype: jnp.dtype = jnp.float64,
    ):
        sites = get_sites()
        N = sites.nsites
        if sites.is_fermion:
            if Nparticle is None:
                self.Nparticle = N
            elif not isinstance(Nparticle, int):
                self.Nparticle = sum(Nparticle)
            else:
                self.Nparticle = Nparticle
        else:
            self.Nparticle = N

        is_dtype_cpl = jnp.issubdtype(dtype, jnp.complexfloating)
        shape = (N * (2 * N - 1),)
        if is_default_cpl() and not is_dtype_cpl:
            shape = (2,) + shape
        scale = np.sqrt(np.e / self.Nparticle, dtype=dtype)
        self.F = jr.normal(get_subkeys(), shape, dtype) * scale
        self.holomorphic = is_default_cpl() and is_dtype_cpl

    def __call__(self, x: jax.Array) -> jax.Array:
        F = self.F if self.F.ndim == 1 else jax.lax.complex(self.F[0], self.F[1])
        N = get_sites().nsites
        F_full = jnp.zeros((2 * N, 2 * N), F.dtype)
        F_full = array_set(F_full, F, jnp.tril_indices(2 * N, -1))
        F_full = F_full - F_full.T
        idx = _get_fermion_idx(x, self.Nparticle)
        return pfaffian(F_full[idx, :][:, idx])

    def rescale(self, maximum: jax.Array) -> Pfaffian:
        F = self.F / maximum.astype(self.F.dtype) ** (2 / self.Nparticle)
        return eqx.tree_at(lambda tree: tree.F, self, F)

    @eqx.filter_jit
    def init_internal(self, x: jax.Array) -> PyTree:
        """
        Initialize internal values for given input configurations
        """

        F = self.F if self.F.ndim == 1 else jax.lax.complex(self.F[0], self.F[1])
        N = get_sites().nsites
        F_full = jnp.zeros((2 * N, 2 * N), F.dtype)
        F_full = array_set(F_full, F, jnp.tril_indices(2 * N, -1))
        F_full = F_full - F_full.T

        idx = _get_fermion_idx(x, self.Nparticle)
        orbs = F_full[idx, :][:, idx]

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

        F = self.F if self.F.ndim == 1 else jax.lax.complex(self.F[0], self.F[1])
        N = get_sites().nsites
        F_full = jnp.zeros((2 * N, 2 * N), F.dtype)
        F_full = array_set(F_full, F, jnp.tril_indices(2 * N, -1))
        F_full = F_full - F_full.T

        occ_idx = internal["idx"]
        old_inv = internal["inv"]
        old_psi = internal["psi"]

        flips = (x - x_old) / 2

        old_idx, new_idx = get_changed_inds(flips, nflips, len(x))

        @jax.vmap
        def idx_to_canon(old_idx):
            return jnp.argwhere(old_idx == occ_idx, size=1)

        old_loc = jnp.ravel(idx_to_canon(old_idx))

        update = F_full[new_idx][:, occ_idx] - F_full[old_idx][:, occ_idx]

        mat = jnp.tril(F_full[new_idx][:, new_idx] - F_full[old_idx][:, old_idx])

        update = array_set(update.T, mat.T, old_loc).T

        one_hot = jax.nn.one_hot(old_loc, len(occ_idx), dtype=F_full.dtype)
        update = jnp.concatenate((update, one_hot), axis=0)

        eye = pfa_eye(nflips // 2, F_full.dtype)

        low_rank_matrix = -1 * eye + update @ old_inv @ update.T

        parity = parity_pfa(new_idx, old_idx, occ_idx)
        psi = old_psi * pfaffian(low_rank_matrix) * parity

        inv_times_update = update @ old_inv

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

        F = self.F if self.F.ndim == 1 else jax.lax.complex(self.F[0], self.F[1])
        N = get_sites().nsites
        F_full = jnp.zeros((2 * N, 2 * N), F.dtype)
        F_full = array_set(F_full, F, jnp.tril_indices(2 * N, -1))
        F_full = F_full - F_full.T

        occ_idx = internal["idx"][idx_segment]
        old_inv = internal["inv"][idx_segment]
        old_psi = internal["psi"][idx_segment]
        x_old = x_old[idx_segment]

        flips = (x - x_old) / 2

        old_idx, new_idx = get_changed_inds(flips, nflips, len(x))

        @jax.vmap
        def idx_to_canon(old_idx):
            return jnp.argwhere(old_idx == occ_idx, size=1)

        old_loc = jnp.ravel(idx_to_canon(old_idx))

        update = F_full[new_idx][:, occ_idx] - F_full[old_idx][:, occ_idx]

        mat = jnp.tril(F_full[new_idx][:, new_idx] - F_full[old_idx][:, old_idx])

        update = array_set(update.T, mat.T, old_loc).T

        one_hot = jax.nn.one_hot(old_loc, len(occ_idx), dtype=F_full.dtype)
        update = jnp.concatenate((update, one_hot), axis=0)

        eye = pfa_eye(nflips // 2, F_full.dtype)

        low_rank_matrix = -1 * eye + update @ old_inv @ update.T

        parity = parity_pfa(new_idx, old_idx, occ_idx)
        return old_psi * pfaffian(low_rank_matrix) * parity


class PairProductSpin(RefModel):
    F: jax.Array
    sublattice: Optional[tuple]
    index: np.ndarray
    holomorphic: bool

    def __init__(
        self, sublattice: Optional[tuple] = None, dtype: jnp.dtype = jnp.float64
    ):
        if get_sites().is_fermion:
            raise RuntimeError("`PairProductSpin` only works in spin systems.")

        N = get_sites().nsites
        if N % 2 > 0:
            raise RuntimeError("`PairProductSpin` only supports even sites.")

        self.sublattice = sublattice
        if sublattice is None:
            nparams = N * (N - 1)
            index = np.zeros((N, N), dtype=np.uint32)
            index[~np.eye(N, dtype=np.bool_)] = np.arange(nparams)
        else:
            lattice = get_lattice()
            c = lattice.shape[0]
            sublattice = (c,) + sublattice
            nparams = np.prod(sublattice) * (N - 1)

            index = np.ones((N, N), dtype=np.uint32)
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

    def rescale(self, maximum: jax.Array) -> PairProductSpin:
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
        flips = (x - x_old) / 2

        F = self.F if self.F.ndim == 1 else jax.lax.complex(self.F[0], self.F[1])
        F_full = F[self.index]
        N = get_sites().nsites

        old_idx, new_idx = get_changed_inds(flips, nflips, len(x))

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

        one_hot = jax.nn.one_hot(old_loc_up, len(occ_idx_up), dtype=F_full.dtype)
        update_lhs = jnp.concatenate((update_lhs, one_hot), axis=0)
        one_hot = jax.nn.one_hot(old_loc_down, len(occ_idx_down), dtype=F_full.dtype).T
        update_rhs = jnp.concatenate((one_hot, update_rhs), axis=1)

        eye = det_eye(nflips // 2, F_full.dtype)
        low_rank_matrix = eye + update_lhs @ old_inv @ update_rhs

        psi = old_psi * det(low_rank_matrix) * parity_det(new_idx, old_idx, occ_idx)

        lhs = update_lhs @ old_inv
        rhs = old_inv @ update_rhs

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
        flips = (x - x_old) / 2

        F = self.F if self.F.ndim == 1 else jax.lax.complex(self.F[0], self.F[1])
        F_full = F[self.index]
        N = get_sites().nsites

        old_idx, new_idx = get_changed_inds(flips, nflips, len(x))

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

        one_hot = jax.nn.one_hot(old_loc_up, len(occ_idx_up), dtype=F_full.dtype)
        update_lhs = jnp.concatenate((update_lhs, one_hot), 0)
        one_hot = jax.nn.one_hot(old_loc_down, len(occ_idx_down), dtype=F_full.dtype).T
        update_rhs = jnp.concatenate((one_hot, update_rhs), 1)

        eye = det_eye(nflips // 2, F_full.dtype)
        low_rank_matrix = eye + update_lhs @ old_inv @ update_rhs

        return old_psi * det(low_rank_matrix) * parity_det(new_idx, old_idx, occ_idx)


#class PfaffianAuxilliaryFermions(RefModel):
class PfaffianAuxilliaryFermions(eqx.Module):
    F: jax.Array
    UnpairedOrbs: eqx.Module
    Nparticle: int
    holomorphic: bool

    def __init__(
        self,
        UnpairedOrbs: Union[eqx.Module, int],
        Nparticle: Union[None, int, Sequence[int]] = None,
        dtype: jnp.dtype = jnp.float64,
    ):
        sites = get_sites()
        N = sites.nsites
        if sites.is_fermion:
            if Nparticle is None:
                self.Nparticle = N
            elif not isinstance(Nparticle, int):
                self.Nparticle = sum(Nparticle)
            else:
                self.Nparticle = Nparticle
        else:
            self.Nparticle = N

        is_dtype_cpl = jnp.issubdtype(dtype, jnp.complexfloating)
        shape = (N * (2 * N - 1),)
        if is_default_cpl() and not is_dtype_cpl:
            shape = (2,) + shape
        scale = np.sqrt(np.e / self.Nparticle, dtype=dtype)
        self.F = jr.normal(get_subkeys(), shape, dtype) * scale
        self.holomorphic = is_default_cpl() and is_dtype_cpl

        if isinstance(UnpairedOrbs, int):
            class GetOrbs(eqx.Module):
                
                U: jax.array

                def __init__(self):
                    shape = (2 * N, UnpairedOrbs)
                    if is_default_cpl() and not is_dtype_cpl:
                        shape = (2,) + shape
                    scale = np.sqrt(np.e / UnpairedOrbs, dtype=dtype)
                    self.U = jr.normal(get_subkeys(), shape, dtype) * scale
                def __call__(self, x: jax.Array):
                    if self.U.ndim == 3:
                        return jax.lax.complex(self.U[0],self.U[1])
                    else:
                        return self.U

            self.UnpairedOrbs = GetOrbs()
        else:
            self.UnpairedOrbs = UnpairedOrbs
    

    def __call__(self, x: jax.Array) -> jax.Array:
        F = self.F if self.F.ndim == 1 else jax.lax.complex(self.F[0], self.F[1])
        N = get_sites().nsites
        F_full = jnp.zeros((2 * N, 2 * N), F.dtype)
        F_full = array_set(F_full, F, jnp.tril_indices(2 * N, -1))
        F_full = F_full - F_full.T
        idx = _get_fermion_idx(x, self.Nparticle)
        
        sliced_pfa = F_full[idx, :][:, idx]

        sliced_det = self.UnpairedOrbs(x)[idx]

        nfree = sliced_det.shape[-1]
 
        return pfaffian(jnp.block([[sliced_pfa, sliced_det],[-1*sliced_det.T, jnp.zeros([nfree,nfree],dtype=sliced_det.dtype)]]))

    def rescale(self, maximum: jax.Array) -> Pfaffian:
        F = self.F / maximum.astype(self.F.dtype) ** (2 / self.Nparticle)
        return eqx.tree_at(lambda tree: tree.F, self, F)

def _get_sublattice_spins(
    s: jax.Array, trans_symm: Symmetry, sublattice: Optional[tuple]
) -> jax.Array:
    s_symm = trans_symm.get_symm_spins(s)

    if sublattice is None:
        return s_symm

    lattice_shape = get_lattice().shape[1:]
    nstates = s_symm.shape[-1]
    for l, subl in zip(lattice_shape, sublattice):
        s_symm = s_symm.reshape(*s_symm.shape[:-2], l // subl, -1, nstates)
        s_symm = s_symm[..., 0, :, :]
        s_symm = s_symm.reshape(*s_symm.shape[:-2], subl, -1, nstates)
    s_symm = s_symm.reshape(-1, nstates)
    return s_symm


def _sub_symmetrize(
    x_full: jax.Array,
    x_sub: jax.Array,
    s: jax.Array,
    trans_symm: Symmetry,
    sublattice: Optional[tuple],
) -> jax.Array:
    if sublattice is None:
        x_sub = x_sub.flatten()
    else:
        lattice_shape = get_lattice().shape[1:]
        lattice_mul = [l // subl for l, subl in zip(lattice_shape, sublattice)]
        x_sub = jnp.tile(x_sub.reshape(sublattice), lattice_mul).flatten()
    return trans_symm.symmetrize(x_full * x_sub, s)


class NeuralJastrow(Sequential, RefModel):
    layers: tuple
    holomorphic: bool = eqx.field(static=True)
    trans_symm: Optional[Symmetry] = eqx.field(static=True)
    sublattice: Optional[tuple] = eqx.field(static=True)

    def __init__(
        self,
        net: eqx.Module,
        fermion_mf: RefModel,
        trans_symm: Optional[Symmetry] = None,
    ):
        class FermionLayer(RawInputLayer):
            fermion_mf: RefModel
            trans_symm: Optional[Symmetry] = eqx.field(static=True)
            sublattice: Optional[tuple] = eqx.field(static=True)

            def __init__(self, fermion_mf, trans_symm):
                self.fermion_mf = fermion_mf
                self.trans_symm = trans_symm
                if hasattr(self.fermion_mf, "sublattice"):
                    self.sublattice = self.fermion_mf.sublattice
                else:
                    self.sublattice = None

            def __call__(self, x: jax.Array, s: jax.Array) -> jax.Array:
                if self.trans_symm is None:
                    return x * self.fermion_mf(s)
                else:
                    x = x.reshape(-1, get_lattice().ncells).mean(axis=0)
                    s_symm = _get_sublattice_spins(s, self.trans_symm, self.sublattice)
                    x_mf = jax.vmap(self.fermion_mf)(s_symm)
                    return _sub_symmetrize(x, x_mf, s, self.trans_symm, self.sublattice)

            def rescale(self, maximum: jax.Array) -> eqx.Module:
                if hasattr(self.fermion_mf, "rescale"):
                    fermion_mf = self.fermion_mf.rescale(maximum)
                    return eqx.tree_at(lambda tree: tree.fermion_mf, self, fermion_mf)
                else:
                    return self

        fermion_layer = FermionLayer(fermion_mf, trans_symm)
        self.trans_symm = trans_symm
        self.sublattice = fermion_layer.sublattice

        if isinstance(net, Sequential):
            layers = net.layers + (fermion_layer,)
        else:
            layers = (net, fermion_layer)

        if hasattr(net, "holomorphic"):
            holomorphic = net.holomorphic and fermion_mf.holomorphic
        else:
            holomorphic = False

        Sequential.__init__(self, layers, holomorphic)

    @eqx.filter_jit
    def init_internal(self, x: jax.Array) -> PyTree:
        """
        Initialize internal values for given input configurations
        """
        fn = self.layers[-1].fermion_mf.init_internal

        if self.trans_symm is None:
            return fn(x)
        else:
            x_symm = _get_sublattice_spins(x, self.trans_symm, self.sublattice)
            return jax.vmap(fn)(x_symm)

    def ref_forward_with_updates(
        self,
        x: jax.Array,
        x_old: jax.Array,
        nflips: int,
        internal: PyTree,
    ) -> Tuple[jax.Array, PyTree]:
        x_net = self[:-1](x)
        fn = self.layers[-1].fermion_mf.ref_forward_with_updates

        if self.trans_symm is None:
            x_mf, internal = fn(x, x_old, nflips, internal)
            return x_net * x_mf, internal
        else:
            x_net = x_net.reshape(-1, get_lattice().ncells).mean(axis=0)
            x_symm = _get_sublattice_spins(x, self.trans_symm, self.sublattice)
            x_old = _get_sublattice_spins(x_old, self.trans_symm, self.sublattice)
            fn_vmap = eqx.filter_vmap(fn, in_axes=(0, 0, None, 0))
            x_mf, internal = fn_vmap(x_symm, x_old, nflips, internal)
            psi = _sub_symmetrize(x_net, x_mf, x, self.trans_symm, self.sublattice)
            return psi, internal

    def ref_forward(
        self,
        x: jax.Array,
        x_old: jax.Array,
        nflips: int,
        idx_segment: jax.Array,
        internal: PyTree,
    ) -> jax.Array:
        x_net = self[:-1](x)
        fn = self.layers[-1].fermion_mf.ref_forward

        if self.trans_symm is None:
            x_mf = fn(x, x_old, nflips, idx_segment, internal)
            return x_net * x_mf
        else:
            x_net = x_net.reshape(-1, get_lattice().ncells).mean(axis=0)
            x_symm = _get_sublattice_spins(x, self.trans_symm, self.sublattice)
            x_old = jax.vmap(_get_sublattice_spins, in_axes=(0, None, None))(
                x_old, self.trans_symm, self.sublattice
            )
            fn_vmap = eqx.filter_vmap(fn, in_axes=(0, 1, None, None, 1))
            x_mf = fn_vmap(x_symm, x_old, nflips, idx_segment, internal)
            return _sub_symmetrize(x_net, x_mf, x, self.trans_symm, self.sublattice)

    def rescale(self, maximum: jax.Array) -> PairProductSpin:
        return Sequential.rescale(self, jnp.sqrt(maximum))


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
