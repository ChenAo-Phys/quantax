from __future__ import annotations

from ..symmetry import Symmetry
from ..nn import Sequential, RefModel, RawInputLayer
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from functools import partial
from ..symmetry import Symmetry
from ..utils import det, pfaffian, array_set
from ..global_defs import get_sites, get_lattice, get_subkeys, is_default_cpl
from .fermion_mf import _get_fermion_idx, _get_changed_inds, _parity_pfa, _parity_det, pfa_eye, det_eye

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

def invert_trans_group(x):
    x = jnp.roll(jnp.flip(x,axis=1),1,axis=1)
    if x.ndim == 3:
        x = jnp.roll(jnp.flip(x,axis=2),1,axis=2)

    return x


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


class PfaffianAuxilliaryFermions(RefModel):
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
                    shape = (UnpairedOrbs, 2*N)
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

        orbs = self.UnpairedOrbs(x)

        #Figure out whether we actually need to call this
        #orbs = invert_trans_group(orbs)
        orbs = orbs.reshape(len(orbs),-1)

        sliced_det = orbs[:,idx].T

        nfree = sliced_det.shape[-1]

        full_orbs = jnp.block([[sliced_pfa, sliced_det],[-1*sliced_det.T, pfa_eye(nfree//2,dtype=sliced_det.dtype)]])

        return pfaffian(full_orbs)
    
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
        
        orbs = self.UnpairedOrbs(x)
        
        #Figure out whether we actually need to call this
        #orbs = invert_trans_group(orbs)
        orbs = orbs.reshape(len(orbs),-1)
        
        F = self.F if self.F.ndim == 1 else jax.lax.complex(self.F[0], self.F[1])
        N = get_sites().nsites
        F_full = jnp.zeros((2 * N, 2 * N), F.dtype)
        F_full = array_set(F_full, F, jnp.tril_indices(2 * N, -1))
        F_full = F_full - F_full.T

        occ_idx = internal["idx"]
        old_inv = internal["inv"]
        old_psi = internal["psi"]
        
        flips = (x - x_old) // 2

        old_idx, new_idx = _get_changed_inds(flips, nflips, len(x))

        @jax.vmap
        def idx_to_canon(old_idx):
            return jnp.argwhere(old_idx == occ_idx, size=1)

        old_loc = jnp.ravel(idx_to_canon(old_idx))
        idx = occ_idx.at[old_loc].set(new_idx)

        update = F_full[new_idx][:, occ_idx] - F_full[old_idx][:, occ_idx]

        mat = jnp.tril(F_full[new_idx][:, new_idx] - F_full[old_idx][:, old_idx])

        update = array_set(update.T, mat.T, old_loc).T

        if get_sites().is_fermion:
            eye = pfa_eye(nflips // 2, F_full.dtype)
        else:
            eye = pfa_eye(nflips, F_full.dtype)
       
        mat11 = update @ old_inv @ update.T
        mat21 = update @ old_inv[:,old_loc]
        mat22 = old_inv[old_loc][:,old_loc]

        low_rank_matrix = -1 * eye + jnp.block([[mat11,mat21],[-1*mat21.T,mat22]])

        norbs = len(occ_idx)
        nfree = len(orbs)

        orbs = orbs[:,idx]
        full_update = jnp.concatenate((update,-1*orbs),0)
        full_update = jnp.concatenate((full_update,jnp.zeros([len(full_update),nfree])),1)

        full_old_loc = jnp.concatenate((old_loc,jnp.arange(norbs,norbs+nfree)))
        b = jnp.zeros([len(occ_idx),nfree])
        full_inv = jnp.block([[old_inv,b],[b.T,-1*pfa_eye(nfree//2,F_full.dtype)]])

        mat11 = full_update @ full_inv @ full_update.T
        mat21 = full_update @ full_inv[:,full_old_loc]
        mat22 = full_inv[full_old_loc][:,full_old_loc]

        elrm = jnp.block([[mat11,mat21],[-1*mat21.T,mat22]])
        elrm = elrm - pfa_eye(len(elrm)//2,F_full.dtype)

        parity = _parity_pfa(new_idx, old_idx, occ_idx)
        psi_mf = old_psi * pfaffian(low_rank_matrix) * parity
        psi = old_psi * pfaffian(elrm) * parity * jnp.power(-1,nfree // 4) 

        inv_times_update = jnp.concatenate((update @ old_inv,old_inv[old_loc]),0) 

        solve = jnp.linalg.solve(low_rank_matrix, inv_times_update)
        inv = old_inv + inv_times_update.T @ solve

        sort = jnp.argsort(idx)

        return psi, {"idx": idx[sort], "inv": inv[sort][:, sort], "psi": psi_mf}

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
        
        orbs = self.UnpairedOrbs(x)
        
        #Figure out whether we actually need to call this
        #orbs = invert_trans_group(orbs)
        orbs = orbs.reshape(len(orbs),-1)

        F = self.F if self.F.ndim == 1 else jax.lax.complex(self.F[0], self.F[1])
        N = get_sites().nsites
        F_full = jnp.zeros((2 * N, 2 * N), F.dtype)
        F_full = array_set(F_full, F, jnp.tril_indices(2 * N, -1))
        F_full = F_full - F_full.T

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
        idx = occ_idx.at[old_loc].set(new_idx)

        update = F_full[new_idx][:, occ_idx] - F_full[old_idx][:, occ_idx]

        mat = jnp.tril(F_full[new_idx][:, new_idx] - F_full[old_idx][:, old_idx])

        update = array_set(update.T, mat.T, old_loc).T

        norbs = len(occ_idx)
        nfree = len(orbs)

        orbs = orbs[:,idx]
        full_update = jnp.concatenate((update,-1*orbs),0)
        full_update = jnp.concatenate((full_update,jnp.zeros([len(full_update),nfree])),1)

        full_old_loc = jnp.concatenate((old_loc,jnp.arange(norbs,norbs+nfree)))
        b = jnp.zeros([len(occ_idx),nfree])
        full_inv = jnp.block([[old_inv,b],[b.T,-1*pfa_eye(nfree//2,F_full.dtype)]])

        mat11 = full_update @ full_inv @ full_update.T
        mat21 = full_update @ full_inv[:,full_old_loc]
        mat22 = full_inv[full_old_loc][:,full_old_loc]

        elrm = jnp.block([[mat11,mat21],[-1*mat21.T,mat22]])
        elrm = elrm - pfa_eye(len(elrm)//2,F_full.dtype)

        parity = _parity_pfa(new_idx, old_idx, occ_idx)
        return old_psi * pfaffian(elrm) * parity * jnp.power(-1,nfree // 4) 

    def rescale(self, maximum: jax.Array) -> Pfaffian:
        F = self.F / maximum.astype(self.F.dtype) ** (2 / self.Nparticle)
        return eqx.tree_at(lambda tree: tree.F, self, F)


