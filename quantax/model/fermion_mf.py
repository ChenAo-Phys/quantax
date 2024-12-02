from __future__ import annotations
from typing import Union, Optional, Tuple, Sequence
from jaxtyping import PyTree
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from functools import partial
from ..utils import det, pfaffian
from ..global_defs import get_sites, get_lattice, get_subkeys, is_default_cpl
from ..nn import RefModel
from ..utils import array_set, _det_update_rows, _det_update_gen, _pfa_update

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

def _get_changed_inds(flips, nflips, N):
    old_idx = jnp.argwhere(flips < 0, size=nflips // 2).astype(jnp.int16).ravel()
    new_idx = jnp.argwhere(flips > 0, size=nflips // 2).astype(jnp.int16).ravel()

    if not get_sites().is_fermion:
        old_idx2 = jnp.concatenate((old_idx, new_idx + N))
        new_idx2 = jnp.concatenate((new_idx, old_idx + N))
        return old_idx2, new_idx2
    else:
        return old_idx, new_idx

def _full_idx_to_spin(idx, N):
    idx_down, idx_up = jnp.split(idx,2)

    return idx_down, idx_up - N

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

        return _low_rank_update_determinant(U, x, x_old, nflips, occ_idx, old_inv, old_psi, True)

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

        occ_idx = internal["idx"][idx_segment]
        old_inv = internal["inv"][idx_segment]
        old_psi = internal["psi"][idx_segment]
        x_old = x_old[idx_segment]

        return _low_rank_update_determinant(U, x, x_old, nflips, occ_idx, old_inv, old_psi, False)
    
    def rescale(self, maximum: jax.Array) -> Determinant:
        U = self.U / maximum.astype(self.U.dtype) ** (1 / self.Nparticle)
        return eqx.tree_at(lambda tree: tree.U, self, U)

class Pfaffian(RefModel):
    F: jax.Array
    Nparticle: int
    index: jax.Array
    holomorphic: bool
    sublattice: Optional[tuple] = eqx.field(static=True)

    def __init__(
        self,
        Nparticle: Union[None, int, Sequence[int]] = None,
        sublattice: Optional[tuple] = None,
        dtype: jnp.dtype = jnp.float64,
    ):
        self.Nparticle = _get_Nparticle(Nparticle)

        N = get_sites().nsites
        is_dtype_cpl = jnp.issubdtype(dtype, jnp.complexfloating)

        index, nparams = _get_pfaffian_indices(sublattice, 2 * N)

        self.index = index
        shape = (nparams,)

        if is_default_cpl() and not is_dtype_cpl:
            shape = (2,) + shape
        scale = np.sqrt(np.e / self.Nparticle, dtype=dtype)

        self.F = jr.normal(get_subkeys(), shape, dtype) * scale
        self.holomorphic = is_default_cpl() and is_dtype_cpl
        self.sublattice = sublattice

    @property
    def F_full(self) -> jax.Array:
        F = self.F if self.F.ndim == 1 else jax.lax.complex(self.F[0], self.F[1])

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
        psi = pfaffian(orbs)

        inv = jnp.linalg.inv(orbs)
        inv = (inv - inv.T) / 2

        return {"idx": idx, "inv": inv, "psi": psi}

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

        return _low_rank_update_pfaffian(self.F_full, x, x_old, nflips, occ_idx, old_inv, old_psi, True)

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

        return _low_rank_update_pfaffian(self.F_full, x, x_old, nflips, occ_idx, old_inv, old_psi, False)

class PairProduct(RefModel):
    F: jax.Array
    Nparticle: int
    index: jax.Array
    holomorphic: bool
    sublattice: Optional[tuple] = eqx.field(static=True)

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
        self.sublattice = sublattice

    @eqx.filter_jit
    def init_internal(self, x: jax.Array) -> PyTree:
        """
        Initialize internal values for given input configurations
        """
        N = get_sites().nsites
        idx = _get_fermion_idx(x, N)
        F_full = self.F_full[idx[: N // 2], :][:, idx[N // 2 :] - N]

        return {"idx": idx, "inv": jnp.linalg.inv(F_full), "psi": det(F_full)}

    def __call__(self, x: jax.Array) -> jax.Array:
        N = get_sites().nsites
        idx = _get_fermion_idx(x, N)
        F_full = self.F_full[idx[: N // 2], :][:, idx[N // 2 :] - N]
        return det(F_full)

    def rescale(self, maximum: jax.Array) -> PairProduct:
        N = get_sites().nsites
        F = self.F / maximum.astype(self.F.dtype) ** (2 / N)
        return eqx.tree_at(lambda tree: tree.F, self, F)

    @property
    def F_full(self) -> jax.Array:
        F = self.F if self.F.ndim == 1 else jax.lax.complex(self.F[0], self.F[1])
        F_full = F[self.index]

        return F_full

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

        return _low_rank_update_pair_product(self.F_full, x, x_old, nflips, occ_idx, old_inv, old_psi, True)

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

        return _low_rank_update_pair_product(self.F_full, x, x_old, nflips, occ_idx, old_inv, old_psi, False)

def _low_rank_update_determinant(U, x, x_old, nflips, occ_idx, old_inv, old_psi, return_inv):
    
    flips = (x - x_old) // 2

    old_idx, new_idx = _get_changed_inds(flips, nflips, len(x))

    update = U[new_idx] - U[old_idx]

    update_idx = _idx_to_canon(old_idx, occ_idx)

    parity = _parity_det(new_idx, old_idx, occ_idx)

    if return_inv == True:
        rat, inv = _det_update_rows(old_inv,update,update_idx,return_inv)
        psi = old_psi * rat * parity

        idx = occ_idx.at[update_idx].set(new_idx)
        sort = jnp.argsort(idx)

        return psi, {"idx": idx[sort], "inv": inv[:, sort], "psi": psi}

    else:
        rat  = _det_update_rows(old_inv,update,update_idx,return_inv)
        return old_psi * rat * parity


def _low_rank_update_pair_product(F_full, x, x_old, nflips, occ_idx, old_inv, old_psi, return_inv):
    N = get_sites().nsites

    flips = (x - x_old) // 2

    old_idx, new_idx = _get_changed_inds(flips, nflips, len(x))

    od, ou = _full_idx_to_spin(old_idx,N)
    nd, nu = _full_idx_to_spin(new_idx,N)
    ocd, ocu = _full_idx_to_spin(occ_idx,N)

    ud = _idx_to_canon(od, ocd)
    uu = _idx_to_canon(ou, ocu)

    row_update = F_full[:, ocu]
    row_update = row_update[nd] - row_update[od]
        
    column_update = F_full[ocd]
    column_update = column_update[:,nu] - column_update[:,ou]

    overlap_update = F_full[nd][:, nu] - F_full[od][:, ou]

    parity = _parity_det(new_idx, old_idx, occ_idx)

    if return_inv == True:
        rat, inv = _det_update_gen(
            old_inv, row_update, column_update, overlap_update, ud, uu, return_inv,
        )

        psi = old_psi*parity*rat

        idx_down = ocd.at[ud].set(nd)
        idx_up = ocu.at[uu].set(nu)

        sort_down = jnp.argsort(idx_down)
        sort_up = jnp.argsort(idx_up)

        internal = {
            "idx": jnp.concatenate((idx_down[sort_down], idx_up[sort_up] + N)),
            "inv": inv[sort_up][:, sort_down],
            "psi": psi,
        }
        
        return psi, internal

    else:
        rat = _det_update_gen(
            old_inv, row_update, column_update, overlap_update, ud, uu, return_inv,
        )

        return old_psi*parity*rat


def _low_rank_update_pfaffian(F_full, x, x_old, nflips, occ_idx, old_inv, old_psi, return_inv):
    
    flips = (x - x_old) // 2

    old_idx, new_idx = _get_changed_inds(flips, nflips, len(x))

    update_idx = _idx_to_canon(old_idx,occ_idx)

    update = F_full[new_idx][:, occ_idx] - F_full[old_idx][:, occ_idx]

    overlap_update = F_full[new_idx][:, new_idx] - F_full[old_idx][:, old_idx]

    update = array_set(update.T, overlap_update.T/2, update_idx).T

    parity = _parity_pfa(new_idx, old_idx, occ_idx)

    if return_inv == True:
        rat, inv = _pfa_update(old_inv, update, update_idx, return_inv)
        psi = old_psi * rat * parity

        idx = occ_idx.at[update_idx].set(new_idx)
        sort = jnp.argsort(idx)

        return psi, {"idx": idx[sort], "inv": inv[sort][:, sort], "psi": psi}

    else:
        rat = _pfa_update(old_inv, update, update_idx, return_inv)
        return old_psi * rat * parity

#Should be deleted once update functions are used in all pfaffians
def pfa_eye(rank, dtype):

    a = jnp.zeros([rank, rank], dtype=dtype)
    b = jnp.eye(rank, dtype=dtype)

    return jnp.block([[a, -1 * b], [b, a]])

@partial(jax.vmap, in_axes=(0,None))
def _idx_to_canon(idx, occ_idx):
    """ 
    Maps overall index to index in the location of occupied orbitals
    """
    return jnp.argwhere(idx == occ_idx, size=1)[0,0]

def _parity_pfa(n, o, i):

    sign1 = jnp.power(-1, _single_electron_parity(n, o, i) % 2)

    sign2 = jnp.sign(o[None] - n[:, None])
    sign2 = sign2.at[jnp.diag_indices(len(sign2))].set(1)

    return jnp.prod(sign1) * jnp.prod(sign2)

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