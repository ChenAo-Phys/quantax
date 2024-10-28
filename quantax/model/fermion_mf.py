from __future__ import annotations
from typing import Optional, Tuple
from jaxtyping import PyTree
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from ..symmetry import Symmetry
from ..nn import Sequential, RefModel, RawInputLayer
from ..utils import det, pfaffian
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

class Determinant(RefModel):
    U: jax.Array
    Nparticle: int
    holomorphic: bool

    def __init__(self, Nparticle: Optional[int] = None, dtype: jnp.dtype = jnp.float32):
        N = get_sites().nsites
        self.Nparticle = N if Nparticle is None else Nparticle
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
        self, x: jax.Array, nflips: int, flips: jax.Array, internal: PyTree
    ) -> Tuple[jax.Array, PyTree]:
        """
        Accelerated forward pass through local updates and internal quantities.
        This function is designed for sampling.

        :return:
            The evaluated wave function and the updated internal values.
        """
        
        U = self.U if self.U.ndim == 2 else jax.lax.complex(self.U[0], self.U[1])
       
        occ_idx = internal['idx']
        old_inv = internal['inv']
        old_psi = internal['psi']

        old_idx = jnp.argwhere(flips < -0.5,size=nflips//2).ravel()
        new_idx = jnp.argwhere(flips > 0.5,size=nflips//2).ravel()

        update = U[new_idx] - U[old_idx]

        @jax.vmap
        def idx_to_canon(old_idx):
            return jnp.argwhere(old_idx == occ_idx,size=1)

        old_loc = jnp.ravel(idx_to_canon(old_idx))
        
        low_rank_matrix = jnp.eye(len(update),dtype=old_psi.dtype) + update@old_inv[:,old_loc]

        psi = old_psi*det(low_rank_matrix)

        inv_times_update = update@old_inv

        inv = old_inv - old_inv[:,old_loc]@jnp.linalg.inv(low_rank_matrix)@inv_times_update 

        idx = occ_idx.at[old_loc].set(new_idx)
        sort = jnp.argsort(idx)
        
        return self(x), {"idx": idx[sort], "inv": inv[sort][:,sort], "psi": psi}

    def ref_forward(
        self,
        x: jax.Array,
        nflips: int,
        flips: jax.Array,
        idx_segment: jax.Array,
        internal: jax.Array,
    ) -> jax.Array:
        """
        Accelerated forward pass through local updates and internal quantities.
        This function is designed for local observables.
        """
        
        U = self.U if self.U.ndim == 2 else jax.lax.complex(self.U[0], self.U[1])

        old_idx = jnp.argwhere(flips < -0.5,size=nflips//2).ravel()
        new_idx = jnp.argwhere(flips > 0.5,size=nflips//2).ravel()

        occ_idx = internal['idx'][idx_segment]
        old_inv = internal['inv'][idx_segment]
        old_psi = internal['psi'][idx_segment]

        update = U[new_idx] - U[old_idx]

        @jax.vmap
        def idx_to_canon(old_idx):
            return jnp.argwhere(old_idx == occ_idx,size=1)

        old_loc = jnp.ravel(idx_to_canon(old_idx))

        low_rank_matrix = jnp.eye(len(update),dtype=old_psi.dtype) + update@old_inv[:,old_loc]

        return old_psi*jnp.linalg.det(low_rank_matrix)

    def rescale(self, maximum: jax.Array) -> Determinant:
        U = self.U / maximum.astype(self.U.dtype) ** (1 / self.Nparticle)
        return eqx.tree_at(lambda tree: tree.U, self, U)


class Pfaffian(eqx.Module):
    F: jax.Array
    Nparticle: int
    holomorphic: bool

    def __init__(self, Nparticle: Optional[int] = None, dtype: jnp.dtype = jnp.float32):
        N = get_sites().nsites
        self.Nparticle = N if Nparticle is None else Nparticle

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
        F_full = F_full.at[jnp.tril_indices(2 * N, -1)].set(F)
        F_full = F_full - F_full.T
        idx = _get_fermion_idx(x, self.Nparticle)
        return pfaffian(F_full[idx, :][:, idx])

    def rescale(self, maximum: jax.Array) -> Pfaffian:
        F = self.F / maximum.astype(self.F.dtype) ** (2 / self.Nparticle)
        return eqx.tree_at(lambda tree: tree.F, self, F)


class PairProductSpin(eqx.Module):
    F: jax.Array
    sublattice: Optional[tuple]
    index: np.ndarray
    holomorphic: bool

    def __init__(
        self, sublattice: Optional[tuple] = None, dtype: jnp.dtype = jnp.float32
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


class NeuralJastrow(Sequential):
    layers: tuple
    holomorphic: bool = eqx.field(static=True)

    def __init__(
        self,
        net: eqx.Module,
        fermion_mf: eqx.Module,
        trans_symm: Optional[Symmetry] = None,
    ):
        class FermionLayer(RawInputLayer):
            fermion_mf: eqx.Module
            trans_symm: Optional[Symmetry] = eqx.field(static=True)

            def __init__(self, fermion_mf, trans_symm):
                self.fermion_mf = fermion_mf
                self.trans_symm = trans_symm

            def __call__(self, x: jax.Array, s: jax.Array) -> jax.Array:
                if self.trans_symm is None:
                    return x * self.fermion_mf(s)

                lattice = get_lattice()
                lattice_shape = lattice.shape[1:]
                x = x.reshape(-1, lattice.ncells).mean(axis=0)

                s_symm = self.trans_symm.get_symm_spins(s)
                if hasattr(self.fermion_mf, "sublattice"):
                    sublattice = self.fermion_mf.sublattice
                else:
                    sublattice = None
                
                if sublattice is not None:
                    nstates = s_symm.shape[-1]
                    lattice_mul = []
                    for l, subl in zip(lattice_shape, sublattice):
                        mul = l // subl
                        lattice_mul.append(mul)
                        s_symm = s_symm.reshape(*s_symm.shape[:-2], mul, -1, nstates)
                        s_symm = s_symm[..., 0, :, :]
                        s_symm = s_symm.reshape(*s_symm.shape[:-2], subl, -1, nstates)
                    s_symm = s_symm.reshape(-1, s_symm.shape[-1])
                    x_mf = jax.vmap(self.fermion_mf)(s_symm)
                    x_mf = jnp.tile(x_mf.reshape(sublattice), lattice_mul)
                    x_mf = x_mf.flatten()
                else:
                    x_mf = jax.vmap(self.fermion_mf)(s_symm)

                return self.trans_symm.symmetrize(x * x_mf, s)

            def rescale(self, maximum: jax.Array) -> eqx.Module:
                if hasattr(self.fermion_mf, "rescale"):
                    fermion_mf = self.fermion_mf.rescale(maximum)
                    return eqx.tree_at(lambda tree: tree.fermion_mf, self, fermion_mf)
                else:
                    return self

        fermion_layer = FermionLayer(fermion_mf, trans_symm)
        if isinstance(net, Sequential):
            layers = net.layers + (fermion_layer,)
        else:
            layers = (net, fermion_layer)

        if hasattr(net, "holomorphic"):
            holomorphic = net.holomorphic and fermion_mf.holomorphic
        else:
            holomorphic = False

        super().__init__(layers, holomorphic)

    def rescale(self, maximum: jax.Array) -> PairProductSpin:
        return super().rescale(jnp.sqrt(maximum))


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
