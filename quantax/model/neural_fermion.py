from __future__ import annotations
from typing import Optional, Tuple, Union
from jaxtyping import PyTree
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from ..nn import Sequential, RefModel, RawInputLayer, Scale, Exp
from ..symmetry import Symmetry, Identity
from ..symmetry.symmetry import _permutation_sign
from ..utils import pfa_eye, pfaffian, array_set
from ..global_defs import get_sites, get_lattice, get_subkeys, is_default_cpl
from .fermion_mf import (
    _get_pfaffian_indices,
    _get_fermion_idx,
    _low_rank_update_pfaffian,
)


def _to_sub_term(x: jax.Array, sublattice: Tuple) -> jax.Array:
    remaining_dims = x.shape[1:]
    x = x.reshape(get_lattice().shape[1:] + remaining_dims)
    for axis, subl in enumerate(sublattice):
        x = x.take(np.arange(subl), axis)
    x = x.reshape(-1, *remaining_dims)
    return x


def _get_sublattice_spins(
    s: jax.Array, trans_symm: Optional[Symmetry], sublattice: Optional[tuple]
) -> jax.Array:
    if trans_symm is None:
        return s[..., None, :]

    perm = _to_sub_term(trans_symm._perm, sublattice)

    nstates = trans_symm.nstates
    batch = s.shape[:-1]
    s = s.reshape(-1, nstates)
    s_symm = s[:, perm]
    s_symm = s_symm.reshape(*batch, -1, perm.shape[0], nstates)
    s_symm = jnp.swapaxes(s_symm, -3, -2)
    return s_symm.reshape(*batch, perm.shape[0], -1)


def _sub_symmetrize(
    x_sub: jax.Array,
    s: jax.Array,
    trans_symm: Optional[Symmetry],
    sublattice: Optional[tuple],
) -> jax.Array:
    if trans_symm is None:
        return x_sub[0]

    eigval = _to_sub_term(trans_symm._eigval, sublattice) / trans_symm.nsymm

    if trans_symm.is_fermion:
        perm = _to_sub_term(trans_symm._perm, sublattice)
        perm_sign = _to_sub_term(trans_symm._perm_sign, sublattice)
        sign = _permutation_sign(s, perm, perm_sign)
        eigval *= sign

    eigval = eigval.astype(x_sub.dtype)
    return jnp.dot(x_sub, eigval)


def _jastrow_sub_symmetrize(
    x_full: jax.Array,
    x_sub: jax.Array,
    s: jax.Array,
    trans_symm: Optional[Symmetry],
    sublattice: Optional[tuple],
) -> jax.Array:
    if trans_symm is None:
        return jnp.mean(x_full) * x_sub[0]
    
    x_full = x_full.reshape(get_lattice().shape[1:])
    for axis, subl in enumerate(sublattice):
        new_shape = x_full.shape[:axis] + (-1, subl) + x_full.shape[axis + 1 :]
        x_full = x_full.reshape(new_shape)
        x_full = jnp.mean(x_full, axis)
    return _sub_symmetrize(x_full.flatten() * x_sub, s, trans_symm, sublattice)


class _JastrowFermionLayer(RawInputLayer):
    fermion_mf: RefModel
    trans_symm: Optional[Symmetry] = eqx.field(static=True)
    sublattice: Optional[tuple] = eqx.field(static=True)

    def __init__(self, fermion_mf, trans_symm):
        self.fermion_mf = fermion_mf
        self.trans_symm = trans_symm
        if hasattr(fermion_mf, "sublattice"):
            self.sublattice = fermion_mf.sublattice
        else:
            self.sublattice = None

    def get_sublattice_spins(self, s: jax.Array) -> jax.Array:
        return _get_sublattice_spins(s, self.trans_symm, self.sublattice)

    def sub_symmetrize(
        self, x_net: jax.Array, x_mf: jax.Array, s: jax.Array
    ) -> jax.Array:
        return _jastrow_sub_symmetrize(x_net, x_mf, s, self.trans_symm, self.sublattice)

    def __call__(self, x: jax.Array, s: jax.Array) -> jax.Array:
        if x.size > 1:
            x = x.reshape(-1, get_lattice().ncells).mean(axis=0)

        s_symm = self.get_sublattice_spins(s)
        x_mf = jax.vmap(self.fermion_mf)(s_symm)
        return self.sub_symmetrize(x, x_mf, s)

    def rescale(self, maximum: jax.Array) -> eqx.Module:
        if hasattr(self.fermion_mf, "rescale"):
            fermion_mf = self.fermion_mf.rescale(maximum)
            return eqx.tree_at(lambda tree: tree.fermion_mf, self, fermion_mf)
        else:
            return self


class NeuralJastrow(Sequential, RefModel):
    layers: Tuple[eqx.Module, ...]
    holomorphic: bool
    trans_symm: Optional[Symmetry]
    sublattice: Tuple[int, ...]

    def __init__(
        self,
        net: eqx.Module,
        fermion_mf: RefModel,
        trans_symm: Optional[Symmetry] = None,
    ):
        fermion_layer = _JastrowFermionLayer(fermion_mf, trans_symm)
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

    @property
    def net(self) -> Sequential:
        return self[:-1]

    @property
    def fermion_layer(self) -> _JastrowFermionLayer:
        return self.layers[-1]

    @property
    def fermion_mf(self) -> RefModel:
        return self.layers[-1].fermion_mf

    def rescale(self, maximum: jax.Array) -> NeuralJastrow:
        return Sequential.rescale(self, jnp.sqrt(maximum))

    def get_sublattice_spins(self, x: jax.Array) -> jax.Array:
        return self.fermion_layer.get_sublattice_spins(x)

    def sub_symmetrize(
        self, x_net: jax.Array, x_mf: jax.Array, s: jax.Array
    ) -> jax.Array:
        return self.fermion_layer.sub_symmetrize(x_net, x_mf, s)

    def init_internal(self, s: jax.Array) -> PyTree:
        """
        Initialize internal values for given input configurations
        """
        s_symm = self.get_sublattice_spins(s)
        return jax.vmap(self.fermion_mf.init_internal)(s_symm)

    def ref_forward_with_updates(
        self,
        s: jax.Array,
        s_old: jax.Array,
        nflips: int,
        internal: PyTree,
    ) -> Tuple[jax.Array, PyTree]:
        x_net = self.net(s)
        if x_net.size > 1:
            x_net = x_net.reshape(-1, get_lattice().ncells).mean(axis=0)

        s_symm = self.get_sublattice_spins(s)
        s_old = self.get_sublattice_spins(s_old)
        fn_vmap = eqx.filter_vmap(
            self.fermion_mf.ref_forward_with_updates, in_axes=(0, 0, None, 0)
        )
        x_mf, internal = fn_vmap(s_symm, s_old, nflips, internal)
        psi = self.sub_symmetrize(x_net, x_mf, s)
        return psi, internal

    def ref_forward(
        self,
        s: jax.Array,
        s_old: jax.Array,
        nflips: int,
        idx_segment: jax.Array,
        internal: PyTree,
    ) -> jax.Array:
        x_net = self.net(s)
        if x_net.size > 1:
            x_net = x_net.reshape(-1, get_lattice().ncells).mean(axis=0)

        s_symm = self.get_sublattice_spins(s)
        s_old = self.get_sublattice_spins(s_old)
        fn_vmap = eqx.filter_vmap(
            self.fermion_mf.ref_forward, in_axes=(0, 1, None, None, 1)
        )
        x_mf = fn_vmap(s_symm, s_old, nflips, idx_segment, internal)
        psi = self.sub_symmetrize(x_net, x_mf, s)
        return psi


class _ConstantPairing(eqx.Module):
    pairing: jax.Array

    def __init__(self, Nhidden: int, dtype: jnp.dtype = jnp.float64):
        N = get_sites().N
        shape = (2 * Nhidden, 2 * N)
        is_dtype_cpl = jnp.issubdtype(dtype, jnp.complexfloating)
        if is_default_cpl() and not is_dtype_cpl:
            shape = (2,) + shape
        self.pairing = jr.normal(get_subkeys(), shape, dtype)

    def __call__(self, x: jax.Array):
        if self.pairing.ndim == 3:
            return jax.lax.complex(self.pairing[0], self.pairing[1])
        else:
            return self.pairing


class _FullOrbsLayerPfaffian(RawInputLayer):
    F: jax.Array
    F_hidden: jax.Array
    index: jax.Array
    Nhidden: int
    holomorphic: bool
    trans_symm: Symmetry
    pg_symm: Symmetry
    sublattice: Tuple[int, ...]
    scale_layer: Scale
    exp_layer: Exp

    def __init__(
        self,
        Nhidden: int,
        trans_symm: Optional[Symmetry],
        pg_symm: Symmetry,
        sublattice: Tuple[int, ...],
        dtype: jnp.dtype = jnp.float64,
    ):

        sites = get_sites()
        N = sites.N
        self.Nhidden = Nhidden
        Ntotal = sites.Ntotal + Nhidden

        index, nparams = _get_pfaffian_indices(sublattice, 2 * N)
        self.index = index

        #F_hidden = jnp.zeros((Nhidden*(Nhidden-1)//2),dtype=dtype)
        F_hidden = jr.normal(get_subkeys(),(Nhidden*(Nhidden-1)//2),dtype=dtype)

        is_dtype_cpl = jnp.issubdtype(dtype, jnp.complexfloating)
        if is_default_cpl() and not is_dtype_cpl:
            self.F = jr.normal(get_subkeys(), (2, nparams), dtype)
            self.F_hidden = jnp.stack([F_hidden.real, F_hidden.imag], axis=0)
        else:
            self.F = jr.normal(get_subkeys(), (nparams), dtype)
            self.F_hidden = F_hidden

        self.holomorphic = is_default_cpl() and is_dtype_cpl
        self.trans_symm = trans_symm
        self.pg_symm = pg_symm

        self.sublattice = sublattice

        self.scale_layer = Scale(np.sqrt(np.e / Ntotal))
        self.exp_layer = Exp()

    def pairing_and_jastrow(self, x: jax.Array) -> jax.Array:
        N = get_sites().N
        x = x.reshape(-1, 2 * N)
        x_mf = x[: self.Nhidden]
        jastrow = x[self.Nhidden :]
        jastrow = jnp.mean(jastrow.reshape(-1, N), axis=0)
        return self.scale_layer(x_mf), self.exp_layer(jastrow)

    @property
    def F_full(self) -> jax.Array:
        F = self.F if self.F.ndim == 1 else jax.lax.complex(self.F[0], self.F[1])

        F_full = F[self.index]
        F_full = (F_full - F_full.T)/2

        return self.scale_layer(F_full)

    @property
    def F_hidden_full(self) -> jax.Array:
        Nhidden = self.Nhidden
        if self.F_hidden.ndim == 1:
            F_hidden = self.F_hidden
        else:
            F_hidden = jax.lax.complex(self.F_hidden[0], self.F_hidden[1])
        F_full = jnp.zeros((Nhidden, Nhidden), F_hidden.dtype)
        F_full = array_set(F_full, F_hidden, jnp.triu_indices(Nhidden, 1))
        F_full = (F_full - F_full.T)/2
        return self.scale_layer(F_full)

    def get_sublattice_spins(self, x: jax.Array) -> jax.Array:
        return _get_sublattice_spins(x, self.trans_symm, self.sublattice)
    
    def sub_symmetrize(
        self, jastrow: jax.Array, mf: jax.Array, s: jax.Array
    ) -> jax.Array:
        return _jastrow_sub_symmetrize(jastrow, mf, s, self.trans_symm, self.sublattice)

    def __call__(self, x: jax.Array, s: jax.Array) -> jax.Array:
        x, jastrow = jax.vmap(self.pairing_and_jastrow, in_axes=1)(x)

        s_point = self.pg_symm.get_symm_spins(s) 
        s_symm = self.get_sublattice_spins(s_point)
        x_symm = self.get_sublattice_spins(x)
        x_symm = x_symm.swapaxes(1,2)

        psi = jax.vmap(jax.vmap(self.forward))(x_symm, s_symm)

        psi = jax.vmap(self.sub_symmetrize)(jastrow, psi, s_point)

        return self.pg_symm.symmetrize(psi,s)
    

    def forward(self, x: jax.Array, s: jax.Array) -> jax.Array:
        idx = _get_fermion_idx(s, get_sites().Ntotal)

        F_full = self.F_full
        sliced_pfa = F_full[idx, :][:, idx]

        pairing = x[:, idx].T.astype(sliced_pfa.dtype)

        F_hidden_full = self.F_hidden_full
        
        full_orbs = sliced_pfa + pairing @ F_hidden_full @ pairing.T
        
        return pfaffian(full_orbs)

    def rescale(self, maximum: jax.Array) -> _FullOrbsLayerPfaffian:
        Ntotal = get_sites().Ntotal + self.Nhidden

        scale = self.scale_layer.scale
        scale /= maximum.astype(scale.dtype) ** (1 / Ntotal)
        where = lambda tree: tree.scale_layer.scale
        tree = eqx.tree_at(where, self, scale)

        new_exp = self.exp_layer.rescale(jnp.sqrt(maximum))
        where = lambda tree: tree.exp_layer
        tree = eqx.tree_at(where, tree, new_exp)

        return tree


def _get_default_Nhidden(net: eqx.Module) -> int:
    sites = get_sites()
    s = jax.ShapeDtypeStruct((sites.nstates,), jnp.int8)
    x = jax.eval_shape(net, s)
    if x.size % (4 * sites.N) == 0:
        return x.size // (4 * sites.N)
    else:
        raise ValueError("Can't determine the default number of hidden fermions.")


class HiddenPfaffian(Sequential, RefModel):
    Nhidden: int
    layers: Tuple[eqx.Module, ...]
    holomorphic: bool
    trans_symm: Optional[Symmetry]
    pg_symm: Optional[Symmetry]
    sublattice: Optional[Tuple[int, ...]]

    def __init__(
        self,
        pairing_net: Optional[eqx.Module] = None,
        Nhidden: Optional[int] = None,
        trans_symm: Optional[Symmetry] = None,
        pg_symm: Optional[Symmetry] = None, 
        sublattice: Optional[tuple] = None,
        dtype: jnp.dtype = jnp.float64,
    ):
        if pairing_net is None:
            if sublattice is not None:
                raise NotImplementedError(
                    "Constant pairing is not implemented with sublattice symmetry,"
                    "try using a CNN for pairing net."
                )
            if Nhidden is None:
                raise ValueError(
                    "`Nhidden` should be specified if `pairing_net` is not given"
                )
            pairing_net = _ConstantPairing(Nhidden, dtype)

        self.Nhidden = _get_default_Nhidden(pairing_net) if Nhidden is None else Nhidden
        self.trans_symm = trans_symm

        if trans_symm is None:
            self.sublattice = None
        elif sublattice is None:
            self.sublattice = get_lattice().shape[1:]
        else:
            self.sublattice = sublattice

        if pg_symm is None:
            self.pg_symm = Identity()
            reshape_layer = eqx.nn.Lambda(lambda x: x[:,None])
        else:
            self.pg_symm = pg_symm
            reshape_layer = eqx.nn.Lambda(lambda x: x)

        full_orbs_layer = _FullOrbsLayerPfaffian(
            self.Nhidden, self.trans_symm, self.pg_symm, self.sublattice, dtype
        )

        if isinstance(pairing_net, Sequential):
            layers = pairing_net.layers + (reshape_layer, full_orbs_layer,)
        else:
            layers = (pairing_net, reshape_layer, full_orbs_layer)

        if hasattr(pairing_net, "holomorphic"):
            holomorphic = pairing_net.holomorphic and full_orbs_layer.holomorphic
        else:
            holomorphic = False

        Sequential.__init__(self, layers, holomorphic)

    @property
    def pairing_net(self) -> Sequential:
        return self[:-1]

    @property
    def full_orbs_layer(self) -> _FullOrbsLayerPfaffian:
        return self.layers[-1]

    def rescale(self, maximum: jax.Array) -> HiddenPfaffian:
        new_orbs_layer = self.full_orbs_layer.rescale(maximum)
        where = lambda tree: tree.full_orbs_layer
        return eqx.tree_at(where, self, new_orbs_layer)

    def get_sublattice_spins(self, x: jax.Array) -> jax.Array:
        return self.full_orbs_layer.get_sublattice_spins(x)

    def sub_symmetrize(
        self, jastrow: jax.Array, psi: jax.Array, s: jax.Array
    ) -> jax.Array:
        return self.full_orbs_layer.sub_symmetrize(jastrow, psi, s)

    def _init_internal(self, s: jax.Array) -> PyTree:
        """
        Initialize internal values for given input configurations
        """
        F_full = self.full_orbs_layer.F_full
        idx = _get_fermion_idx(s, get_lattice().Ntotal)
        orbs = F_full[idx, :][:, idx]

        inv = jnp.linalg.inv(orbs)
        inv = (inv - inv.T) / 2
        return {"idx": idx, "inv": inv, "psi": pfaffian(orbs)}

    def init_internal(self, s: jax.Array) -> PyTree:

        s_point = self.pg_symm.get_symm_spins(s) 
        s_symm = self.get_sublattice_spins(s_point)

        s_symm = s_symm.reshape(-1,s_symm.shape[-1])

        return jax.vmap(self._init_internal)(s_symm)

    def ref_forward_with_updates(
        self,
        s: jax.Array,
        s_old: jax.Array,
        nflips: int,
        internal: PyTree,
    ) -> Tuple[jax.Array, PyTree]:
        x = self.pairing_net(s)
        
        pairing, jastrow = jax.vmap(self.full_orbs_layer.pairing_and_jastrow, in_axes=1)(x)

        s_point = self.pg_symm.get_symm_spins(s) 
        s_symm = self.get_sublattice_spins(s_point)
        s_old_point = self.pg_symm.get_symm_spins(s_old) 
        s_old_symm = self.get_sublattice_spins(s_old_point)
        pair_symm = self.get_sublattice_spins(pairing)
        pair_symm = pair_symm.swapaxes(1,2)

        n_point = s_symm.shape[0]
        n_trans = s_symm.shape[1]
        n_symm = n_trans*n_point

        s_symm = s_symm.reshape(n_symm,-1)      
        s_old_symm = s_old_symm.reshape(n_symm,-1)      
        pair_symm = pair_symm.reshape(n_symm,*pair_symm.shape[2:])      

        occ_idx = internal["idx"]
        old_inv = internal["inv"]
        old_psi = internal["psi"]

        fn = eqx.filter_vmap(
            self._low_rank_update, in_axes=(0, 0, None, 0, 0, 0, 0, None)
        )
        psi, internal = fn(
            s_symm, s_old_symm, nflips, occ_idx, old_inv, old_psi, pair_symm, True
        )
        
        psi = psi.reshape(n_point,n_trans)

        psi = jax.vmap(self.sub_symmetrize)(jastrow, psi, s_point)

        psi = self.pg_symm.symmetrize(psi,s)
        
        return psi, internal

    def ref_forward(
        self,
        s: jax.Array,
        s_old: jax.Array,
        nflips: int,
        idx_segment: jax.Array,
        internal: PyTree,
    ) -> jax.Array:
        x = self.pairing_net(s)
        pairing, jastrow = jax.vmap(self.full_orbs_layer.pairing_and_jastrow, in_axes=1)(x)
        
        s_old = s_old[idx_segment]
        s_point = self.pg_symm.get_symm_spins(s) 
        s_symm = self.get_sublattice_spins(s_point)
        s_old_point = self.pg_symm.get_symm_spins(s_old) 
        s_old_symm = self.get_sublattice_spins(s_old_point)
        pair_symm = self.get_sublattice_spins(pairing)
        pair_symm = pair_symm.swapaxes(1,2)

        n_point = s_symm.shape[0]
        n_trans = s_symm.shape[1]
        n_symm = n_trans*n_point

        s_symm = s_symm.reshape(n_symm,-1)      
        s_old_symm = s_old_symm.reshape(n_symm,-1)      
        pair_symm = pair_symm.reshape(n_symm,*pair_symm.shape[2:])      

        occ_idx = internal["idx"][idx_segment]
        old_inv = internal["inv"][idx_segment]
        old_psi = internal["psi"][idx_segment]

        fn = eqx.filter_vmap(
            self._low_rank_update, in_axes=(0, 0, None, 0, 0, 0, 0, None)
        )
        psi = fn(s_symm, s_old_symm, nflips, occ_idx, old_inv, old_psi, pair_symm, False)
        
        psi = psi.reshape(n_point,n_trans)

        psi = jax.vmap(self.sub_symmetrize)(jastrow, psi, s_point)

        psi = self.pg_symm.symmetrize(psi,s)
        
        return psi

    def _low_rank_update(
        self,
        s: jax.Array,
        s_old: jax.Array,
        nflips: int,
        occ_idx: jax.Array,
        old_inv: jax.Array,
        old_psi: jax.Array,
        pairing: jax.Array,
        return_internal: bool,
    ) -> Union[jax.Array, Tuple[jax.Array, PyTree]]:
        """
        Accelerated forward pass through local updates and internal quantities.
        This function is designed for sampling.

        :return:
            The evaluated wave function and the updated internal values.
        """
        F_full = self.full_orbs_layer.F_full
        F_hidden_full = self.full_orbs_layer.F_hidden_full
        dtype = F_full.dtype
        pairing = pairing.astype(dtype)
        
        psi_mf, internal = _low_rank_update_pfaffian(F_full,s,s_old,nflips, occ_idx, old_inv, old_psi,True)

        idx = internal['idx']
        inv = internal['inv']
        sliced_orbs = pairing[:, idx]

        inv_full = jnp.linalg.inv(F_hidden_full)
        ratio = pfaffian(inv_full + sliced_orbs@inv@sliced_orbs.T)/pfaffian(inv_full)
        ratio = jnp.where(jnp.allclose(F_hidden_full,0),1,ratio)
        psi = psi_mf*ratio

        if return_internal:
            return psi, internal
        else:
            return psi
