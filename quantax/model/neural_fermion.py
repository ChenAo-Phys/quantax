from __future__ import annotations
from typing import Optional, Tuple, Union
from jaxtyping import PyTree, ArrayLike
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from ..nn import Sequential, RefModel, RawInputLayer, Scale
from ..symmetry.symmetry import Symmetry, _permutation_sign
from ..utils import pfaffian, pfa_update, array_set
from ..global_defs import get_sites, get_lattice, get_subkeys, is_default_cpl
from .fermion_mf import (
    _get_pfaffian_indices,
    _get_fermion_idx,
    _get_changed_inds,
    _parity_pfa,
    _idx_to_canon,
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


def _jastrow_sub_symmetrize(
    x_full: ArrayLike,
    x_sub: jax.Array,
    s: jax.Array,
    trans_symm: Optional[Symmetry],
    sublattice: Optional[tuple],
) -> jax.Array:
    if trans_symm is None:
        return x_full * x_sub[0]

    lattice_shape = get_lattice().shape[1:]
    lattice_mul = [l // subl for l, subl in zip(lattice_shape, sublattice)]
    x_sub = jnp.tile(x_sub.reshape(sublattice), lattice_mul).flatten()
    return trans_symm.symmetrize(x_full * x_sub, s)


class _JastrowFermionLayer(RawInputLayer):
    fermion_mf: RefModel
    trans_symm: Optional[Symmetry] = eqx.field(static=True)
    sublattice: Optional[tuple] = eqx.field(static=True)

    def __init__(self, fermion_mf, trans_symm):
        self.fermion_mf = fermion_mf
        self.trans_symm = trans_symm
        if hasattr(fermion_mf, "sublattice") and fermion_mf.sublattice is not None:
            self.sublattice = fermion_mf.sublattice
        else:
            self.sublattice = get_lattice().shape[1:]

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


def pfa_eye(rank, dtype):
    a = jnp.zeros([rank, rank], dtype=dtype)
    b = jnp.eye(rank, dtype=dtype)

    return jnp.block([[a, b], [-b, a]])


class _FullOrbsLayerPfaffian(RawInputLayer):
    F: jax.Array
    index: jax.Array
    Nhidden: int
    holomorphic: bool
    trans_symm: Symmetry
    sublattice: Tuple[int, ...]
    scale_layer: Scale

    def __init__(
        self,
        Nhidden: int,
        trans_symm: Optional[Symmetry],
        sublattice: Tuple[int, ...],
        dtype: jnp.dtype = jnp.float64,
    ):

        sites = get_sites()
        N = sites.N
        Ntotal = sites.Ntotal + Nhidden
        self.Nhidden = Nhidden

        index, nparams = _get_pfaffian_indices(sublattice, 2 * N)
        self.index = index

        is_dtype_cpl = jnp.issubdtype(dtype, jnp.complexfloating)
        if is_default_cpl() and not is_dtype_cpl:
            self.F = jr.normal(get_subkeys(), (2, nparams), dtype)
        else:
            self.F = jr.normal(get_subkeys(), (nparams), dtype)

        self.holomorphic = is_default_cpl() and is_dtype_cpl
        self.trans_symm = trans_symm
        self.sublattice = sublattice
        self.scale_layer = Scale(np.sqrt(np.e / Ntotal))

    @property
    def F_full(self) -> jax.Array:
        F = self.F if self.F.ndim == 1 else jax.lax.complex(self.F[0], self.F[1])

        F_full = F[self.index]
        F_full = F_full - F_full.T

        return self.scale_layer(F_full)

    def to_hidden_orbs(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
        N = get_sites().N
        x = x.reshape(2 * self.Nhidden, -1, 2 * N)
        x = jnp.sum(x, axis=1) / np.sqrt(x.shape[1], dtype=x.dtype)

        pairing = x[: self.Nhidden, :]

        x_hh_up = x[self.Nhidden :, :N]
        x_hh_down = x[self.Nhidden :, N:]
        hidden = x_hh_up @ x_hh_down.T / np.sqrt(x_hh_up.shape[1], dtype=x_hh_up.dtype)
        hidden = hidden - hidden.T + 2 * pfa_eye(hidden.shape[0] // 2, hidden.dtype)
        return self.scale_layer(pairing), self.scale_layer(hidden)

    def get_sublattice_spins(self, x: jax.Array) -> jax.Array:
        return _get_sublattice_spins(x, self.trans_symm, self.sublattice)

    def sub_symmetrize(self, psi: jax.Array, s: jax.Array) -> jax.Array:
        return _sub_symmetrize(psi, s, self.trans_symm, self.sublattice)

    def sliced_mat(self, s: jax.Array, pairing: jax.Array, hidden: jax.Array):
        idx = _get_fermion_idx(s, get_sites().Ntotal)
        F_full = self.F_full
        sliced_pfa = F_full[idx, :][:, idx]
        pairing = pairing[:, idx]

        pairing = pairing.astype(sliced_pfa.dtype)
        hidden = hidden.astype(sliced_pfa.dtype)

        return jnp.block([[sliced_pfa, pairing.T], [-pairing, hidden]])

    def __call__(self, x: jax.Array, s: jax.Array) -> jax.Array:
        pairing, hidden = self.to_hidden_orbs(x)
        s_symm = self.get_sublattice_spins(s)
        pairing_symm = self.get_sublattice_spins(pairing)
        fn_vmap = jax.vmap(self.sliced_mat, in_axes=(0, 1, None))
        mat = fn_vmap(s_symm, pairing_symm, hidden)
        psi = pfaffian(mat)
        return self.sub_symmetrize(psi, s)


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
    sublattice: Tuple[int, ...]

    def __init__(
        self,
        pairing_net: Optional[eqx.Module] = None,
        Nhidden: Optional[int] = None,
        trans_symm: Optional[Symmetry] = None,
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
        if trans_symm is not None and sublattice is None:
            sublattice = get_lattice().shape[1:]
        self.sublattice = sublattice

        full_orbs_layer = _FullOrbsLayerPfaffian(
            self.Nhidden, self.trans_symm, self.sublattice, dtype
        )

        if isinstance(pairing_net, Sequential):
            layers = pairing_net.layers + (full_orbs_layer,)
        else:
            layers = (pairing_net, full_orbs_layer)

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
        scale = self.full_orbs_layer.scale_layer.scale
        Ntotal = get_lattice().Ntotal + self.Nhidden
        scale /= maximum.astype(scale.dtype) ** (2 / Ntotal)
        where = lambda tree: tree.full_orbs_layer.scale_layer.scale
        return eqx.tree_at(where, self, scale)

    def get_sublattice_spins(self, x: jax.Array) -> jax.Array:
        return self.full_orbs_layer.get_sublattice_spins(x)

    def sub_symmetrize(self, psi: jax.Array, s: jax.Array) -> jax.Array:
        return self.full_orbs_layer.sub_symmetrize(psi, s)

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
        s_symm = self.get_sublattice_spins(s)
        return jax.vmap(self._init_internal)(s_symm)

    def ref_forward_with_updates(
        self,
        s: jax.Array,
        s_old: jax.Array,
        nflips: int,
        internal: PyTree,
    ) -> Tuple[jax.Array, PyTree]:
        x = self.pairing_net(s)
        pairing, hidden = self.full_orbs_layer.to_hidden_orbs(x)

        s_symm = self.get_sublattice_spins(s)
        s_old = self.get_sublattice_spins(s_old)
        pair_symm = self.get_sublattice_spins(pairing)

        occ_idx = internal["idx"]
        old_inv = internal["inv"]
        old_psi = internal["psi"]

        fn = eqx.filter_vmap(
            self._low_rank_update, in_axes=(0, 0, None, 0, 0, 0, 1, None, None)
        )
        psi, internal = fn(
            s_symm, s_old, nflips, occ_idx, old_inv, old_psi, pair_symm, hidden, True
        )
        psi = self.sub_symmetrize(psi, s)
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
        pairing, hidden = self.full_orbs_layer.to_hidden_orbs(x)

        s_symm = self.get_sublattice_spins(s)
        s_old = s_old[idx_segment]
        s_old = self.get_sublattice_spins(s_old)
        pair_symm = self.get_sublattice_spins(pairing)

        occ_idx = internal["idx"][idx_segment]
        old_inv = internal["inv"][idx_segment]
        old_psi = internal["psi"][idx_segment]

        fn = eqx.filter_vmap(
            self._low_rank_update, in_axes=(0, 0, None, 0, 0, 0, 1, None, None)
        )
        psi = fn(
            s_symm, s_old, nflips, occ_idx, old_inv, old_psi, pair_symm, hidden, False
        )
        return self.sub_symmetrize(psi, s)

    def _low_rank_update(
        self,
        s: jax.Array,
        s_old: jax.Array,
        nflips: int,
        occ_idx: jax.Array,
        old_inv: jax.Array,
        old_psi: jax.Array,
        pairing: jax.Array,
        hidden: jax.Array,
        return_internal: bool,
    ) -> Union[jax.Array, Tuple[jax.Array, PyTree]]:
        """
        Accelerated forward pass through local updates and internal quantities.
        This function is designed for sampling.

        :return:
            The evaluated wave function and the updated internal values.
        """
        F_full = self.full_orbs_layer.F_full

        flips = (s - s_old) // 2

        old_idx, new_idx = _get_changed_inds(flips, nflips, len(s))

        old_loc = _idx_to_canon(old_idx, occ_idx)

        update = F_full[new_idx][:, occ_idx] - F_full[old_idx][:, occ_idx]
        mat = F_full[new_idx][:, new_idx] - F_full[old_idx][:, old_idx]
        update = array_set(update.T, mat.T / 2, old_loc).T

        sliced_orbs = pairing[:, occ_idx].astype(F_full.dtype)
        Nvisible = get_sites().Ntotal
        full_old_loc = jnp.concatenate(
            (old_loc, jnp.arange(Nvisible, Nvisible + self.Nhidden))
        )

        b = jnp.zeros([len(occ_idx), self.Nhidden])
        id_inv = pfa_eye(self.Nhidden // 2, F_full.dtype)
        full_inv = jnp.block([[old_inv, b], [b.T, id_inv]])

        full_update = jnp.concatenate((update, -1 * sliced_orbs), axis=0)
        full_update = jnp.concatenate(
            (full_update, jnp.zeros([len(full_update), self.Nhidden])), axis=1
        )

        mat22 = hidden + pfa_eye(self.Nhidden // 2, dtype=F_full.dtype)
        full_mat = jnp.block(
            [[mat, pairing[:, new_idx].T], [-1 * pairing[:, new_idx], mat22]]
        )

        full_update = array_set(full_update.T, full_mat.T / 2, full_old_loc).T

        rat = pfa_update(full_inv, full_update, full_old_loc, False)
        parity_mf = _parity_pfa(new_idx, old_idx, occ_idx)
        parity = parity_mf * jnp.power(-1, self.Nhidden // 4)
        psi = old_psi * rat * parity

        if return_internal:
            rat_mf, inv = pfa_update(old_inv, update, old_loc, True)
            psi_mf = old_psi * rat_mf * parity_mf

            idx = occ_idx.at[old_loc].set(new_idx)
            sort = jnp.argsort(idx)

            return psi, {"idx": idx[sort], "inv": inv[sort][:, sort], "psi": psi_mf}
        else:
            return psi
