from __future__ import annotations
from typing import Optional, Tuple, Union, Sequence
from jaxtyping import PyTree
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from ..nn import Sequential, RefModel, RawInputLayer, Scale
from ..symmetry import Symmetry
from ..symmetry.symmetry import _permutation_sign
from ..utils import det, pfaffian, array_set
from ..global_defs import get_sites, get_lattice, get_subkeys, is_default_cpl
from .fermion_mf import (
    _get_pair_product_indices,
    _get_pfaffian_indices,
    _get_Nparticle,
    _get_fermion_idx,
    _get_changed_inds,
    _parity_pfa,
    pfa_eye,
)


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


def _get_sublattice_perm(
    trans_symm: Symmetry,
    sublattice: Optional[tuple],
):
    lattice_shape = get_lattice().shape

    perm = trans_symm._perm
    dims = []
    for fulldim, subdim in zip(lattice_shape[1:], sublattice):
        if not fulldim % subdim == 0:
            raise ValueError(
                f"lattice dimension of length {fulldim} is not divisible by"
                f"sublattice dimension of length {subdim}"
            )
        dims.append(fulldim // subdim)
        dims.append(subdim)
    dims.append(perm.shape[-1])

    perm = perm.reshape(dims)

    for i in range(len(sublattice)):
        perm = jnp.take(perm, 0, axis=i)

    perm = perm.reshape(-1, perm.shape[-1])

    if not get_sites().is_fermion:
        perm = jnp.concatenate((perm, perm + perm.shape[-1]), -1)

    return perm


class _JastrowFermionLayer(RawInputLayer):
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


class NeuralJastrow(Sequential, RefModel):
    layers: Tuple[eqx.Module]
    holomorphic: bool = eqx.field(static=True)
    trans_symm: Optional[Symmetry] = eqx.field(static=True)
    sublattice: Optional[tuple] = eqx.field(static=True)

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
    def fermion_mf(self) -> RefModel:
        return self.layers[-1].fermion_mf

    def rescale(self, maximum: jax.Array) -> NeuralJastrow:
        return Sequential.rescale(self, jnp.sqrt(maximum))

    def init_internal(self, x: jax.Array) -> PyTree:
        """
        Initialize internal values for given input configurations
        """
        fn = self.fermion_mf.init_internal

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
        x_net = self.net(x)
        fn = self.fermion_mf.ref_forward_with_updates

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
        x_net = self.net(x)
        fn = self.fermion_mf.ref_forward

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


class _ConstantPairing(eqx.Module):
    pairing: jax.Array

    def __init__(self, Nhidden: int, dtype: jnp.dtype = jnp.float64):
        N = get_sites().nsites
        shape = (Nhidden, 2 * N)
        is_dtype_cpl = jnp.issubdtype(dtype, jnp.complexfloating)
        if is_default_cpl() and not is_dtype_cpl:
            shape = (2,) + shape
        self.pairing = jr.normal(get_subkeys(), shape, dtype)

    def __call__(self, x: jax.Array):
        if self.pairing.ndim == 3:
            return jax.lax.complex(self.pairing[0], self.pairing[1])
        else:
            return self.pairing


def _get_default_Nhidden(net: eqx.Module) -> int:
    sites = get_sites()
    s = jax.ShapeDtypeStruct((sites.nstates,), jnp.int8)
    x = jax.eval_shape(net, s)
    if x.size % (2 * sites.nsites) == 0:
        return x.size // (2 * sites.nsites)
    else:
        raise ValueError("Can't determine the default number of hidden fermions.")


class _FullOrbsLayerPairProduct(RawInputLayer):
    F: jax.Array
    F_hidden: jax.Array
    index: jax.Array
    Nvisible: int
    Nhidden: int
    holomorphic: bool

    def __init__(
        self,
        Nvisible: int,
        Nhidden: int,
        sublattice: Optional[tuple],
        dtype: jnp.dtype = jnp.float64,
    ):
        sites = get_sites()
        N = sites.nsites
        self.Nvisible = Nvisible
        self.Nhidden = Nhidden

        is_dtype_cpl = jnp.issubdtype(dtype, jnp.complexfloating)

        index, nparams = _get_pair_product_indices(sublattice, N)
        self.index = index
        shape = (nparams,)

        is_dtype_cpl = jnp.issubdtype(dtype, jnp.complexfloating)
        self.F_hidden = jnp.eye(Nhidden, dtype=dtype)
        if is_default_cpl() and not is_dtype_cpl:
            shape = (2,) + shape
            self.F_hidden = jnp.concatenate(
                (self.F_hidden[None], jnp.zeros_like(self.F_hidden[None])), 0
            )

        self.F = jr.normal(get_subkeys(), shape, dtype)
        self.holomorphic = is_default_cpl() and is_dtype_cpl

    def to_hidden_orbs(self, x: jax.Array) -> jax.Array:
        N = get_sites().nsites
        x = x.reshape(2 * self.Nhidden, -1, N)
        x = jnp.sum(x, axis=1) / np.sqrt(x.shape[1], dtype=x.dtype)
        return jnp.split(x, 2, axis=0)

    @property
    def F_full(self) -> jax.Array:
        F = self.F if self.F.ndim == 1 else jax.lax.complex(self.F[0], self.F[1])
        F_full = F[self.index]
        return F_full

    @property
    def F_hidden_full(self) -> jax.Array:
        if self.F_hidden.ndim == 2:
            F_hidden = self.F_hidden
        else:
            F_hidden = jax.lax.complex(self.F_hidden[0], self.F_hidden[1])

        return F_hidden

    def __call__(self, x: jax.Array, s: jax.Array) -> jax.Array:

        idx = _get_fermion_idx(s, self.Nvisible)

        N = get_sites().nsites

        idx_down, idx_up = jnp.split(idx, 2)
        idx_up = idx_up - N

        F_full = self.F_full

        mat11 = F_full[idx_down, :][:, idx_up]

        xd, xu = self.to_hidden_orbs(x)
        mat21 = xd.T[idx_down, :].astype(mat11.dtype)
        mat12 = xu[:, idx_up].astype(mat11.dtype)

        mat22 = self.F_hidden_full

        full_orbs = jnp.block([[mat11, mat21], [mat12, mat22]])

        return full_orbs


class HiddenPairProduct(Sequential):
    Nvisible: int
    Nhidden: int
    layers: Tuple[eqx.Module]
    holomorphic: bool = eqx.field(static=True)

    def __init__(
        self,
        pairing_net: Union[eqx.Module, int],
        Nvisible: Union[None, int, Sequence[int]] = None,
        Nhidden: Optional[int] = None,
        sublattice: Optional[tuple] = None,
        dtype: jnp.dtype = jnp.float64,
    ):
        if isinstance(pairing_net, int):
            pairing_net = _ConstantPairing(pairing_net, dtype)

        self.Nvisible = _get_Nparticle(Nvisible)
        self.Nhidden = _get_default_Nhidden(pairing_net) if Nhidden is None else Nhidden

        full_orbs_layer = _FullOrbsLayerPairProduct(
            self.Nvisible, self.Nhidden, sublattice, dtype
        )
        scale_layer = Scale(np.sqrt(np.e / (self.Nvisible + self.Nhidden)))
        pfa_layer = eqx.nn.Lambda(lambda x: det(x))

        if isinstance(pairing_net, Sequential):
            layers = pairing_net.layers + (full_orbs_layer, scale_layer, pfa_layer)
        else:
            layers = (pairing_net, full_orbs_layer, scale_layer, pfa_layer)

        if hasattr(pairing_net, "holomorphic"):
            holomorphic = pairing_net.holomorphic and full_orbs_layer.holomorphic
        else:
            holomorphic = False

        Sequential.__init__(self, layers, holomorphic)

    @property
    def pairing_net(self) -> Sequential:
        return self[:-3]

    @property
    def full_orbs_layer(self) -> _FullOrbsLayerPfaffian:
        return self.layers[-3]

    @property
    def scale_layer(self) -> Scale:
        return self.layers[-2]

    def rescale(self, maximum: jax.Array) -> HiddenPfaffian:
        scale = self.scale_layer.scale
        scale /= maximum.astype(scale.dtype) ** (2 / (self.Nvisible + self.Nhidden))
        return eqx.tree_at(lambda tree: tree.layers[-2].scale, self, scale)


class _FullOrbsLayerPfaffian(RawInputLayer):
    F: jax.Array
    F_hidden: jax.Array
    index: jax.Array
    Nvisible: int
    Nhidden: int
    holomorphic: bool
    trans_symm: Optional[Symmetry] = eqx.field(static=True)
    sublattice: Optional[tuple] = eqx.field(static=True)
    scale_layer: Scale

    def __init__(
        self,
        Nvisible: int,
        Nhidden: int,
        trans_symm: Optional[Symmetry],
        sublattice: Optional[tuple],
        dtype: jnp.dtype = jnp.float64,
    ):

        sites = get_sites()
        N = sites.nsites
        self.Nvisible = Nvisible
        self.Nhidden = Nhidden

        index, nparams = _get_pfaffian_indices(sublattice, 2 * N)
        self.index = index
        shape = (nparams,)

        F_hidden = -pfa_eye(Nhidden // 2, dtype)
        F_hidden = F_hidden[jnp.triu_indices(Nhidden, 1)] * 2

        is_dtype_cpl = jnp.issubdtype(dtype, jnp.complexfloating)
        if is_default_cpl() and not is_dtype_cpl:
            shape = (2,) + shape
            F_hidden = jnp.stack([F_hidden.real, F_hidden.imag], axis=0)
        self.F = jr.normal(get_subkeys(), shape, dtype)
        self.F_hidden = F_hidden

        self.holomorphic = is_default_cpl() and is_dtype_cpl
        self.trans_symm = trans_symm
        self.sublattice = sublattice
        self.scale_layer = Scale(np.sqrt(np.e / (self.Nvisible + self.Nhidden)))

    def to_hidden_orbs(self, x: jax.Array) -> jax.Array:
        N = get_sites().nsites
        x = x.reshape(self.Nhidden, -1, 2 * N)
        x = jnp.sum(x, axis=1) / np.sqrt(x.shape[1], dtype=x.dtype)
        return self.scale_layer(x)

    @property
    def F_full(self) -> jax.Array:
        F = self.F if self.F.ndim == 1 else jax.lax.complex(self.F[0], self.F[1])

        F_full = F[self.index]
        F_full = F_full - F_full.T

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
        F_full = F_full - F_full.T
        return self.scale_layer(F_full)

    def __call__(self, x: jax.Array, s: jax.Array) -> jax.Array:
        x = self.to_hidden_orbs(x)

        if self.trans_symm is None:
            return self.forward(x, s)
        else:
            perm = _get_sublattice_perm(self.trans_symm, self.sublattice)
            perm_s = perm[:, : s.shape[-1]]

            x = x[:, perm].transpose(1, 0, 2)
            s = s[perm_s]

            return jax.vmap(self.forward)(x, s)

    def forward(self, x: jax.Array, s: jax.Array) -> jax.Array:
        idx = _get_fermion_idx(s, self.Nvisible)

        F_full = self.F_full
        sliced_pfa = F_full[idx, :][:, idx]

        pairing = x[:, idx].T.astype(sliced_pfa.dtype)

        F_hidden_full = self.F_hidden_full

        full_orbs = jnp.block([[sliced_pfa, pairing], [-pairing.T, F_hidden_full]])
        return pfaffian(full_orbs)


class HiddenPfaffian(Sequential, RefModel):
    Nvisible: int
    Nhidden: int
    layers: Tuple[eqx.Module, ...]
    holomorphic: bool = eqx.field(static=True)
    trans_symm: Optional[Symmetry] = eqx.field(static=True)
    sublattice: Optional[tuple] = eqx.field(static=True)
    perm: Optional[jax.Array] = eqx.field(static=True)

    def __init__(
        self,
        pairing_net: Optional[eqx.Module] = None,
        Nvisible: Union[None, int, Sequence[int]] = None,
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

        self.Nvisible = _get_Nparticle(Nvisible)
        self.Nhidden = _get_default_Nhidden(pairing_net) if Nhidden is None else Nhidden

        if trans_symm is not None:
            if sublattice is None:
                sublattice = get_lattice().shape[1:]
            perm = _get_sublattice_perm(trans_symm, sublattice)
        else:
            perm = None

        self.trans_symm = trans_symm
        self.sublattice = sublattice
        self.perm = perm

        full_orbs_layer = _FullOrbsLayerPfaffian(
            self.Nvisible, self.Nhidden, trans_symm, sublattice, dtype
        )

        class SymmLayer(RawInputLayer):
            def __call__(self, x, s):
                if trans_symm == None:
                    return x
                else:
                    if get_sites().is_fermion:
                        perm_sign = jnp.ones([len(perm)])
                        Nparticle = 2 * get_sites().nsites
                        sign = _permutation_sign(s, perm, perm_sign, Nparticle)
                        return jnp.sum(x * sign)
                    else:
                        return jnp.sum(x)

        symm_layer = SymmLayer()

        additional_layers = (full_orbs_layer, symm_layer)
        if isinstance(pairing_net, Sequential):
            layers = pairing_net.layers + additional_layers
        else:
            layers = (pairing_net,) + additional_layers

        if hasattr(pairing_net, "holomorphic"):
            holomorphic = pairing_net.holomorphic and full_orbs_layer.holomorphic
        else:
            holomorphic = False

        Sequential.__init__(self, layers, holomorphic)

    @property
    def pairing_net(self) -> Sequential:
        return self[:-2]

    @property
    def full_orbs_layer(self) -> _FullOrbsLayerPfaffian:
        return self.layers[-2]

    @property
    def symm_layer(self) -> eqx.Module:
        return self.layers[-1]

    def rescale(self, maximum: jax.Array) -> HiddenPfaffian:
        scale = self.full_orbs_layer.scale_layer.scale
        scale /= maximum.astype(scale.dtype) ** (2 / (self.Nvisible + self.Nhidden))
        where = lambda tree: tree.full_orbs_layer.scale_layer.scale
        return eqx.tree_at(where, self, scale)

    def init_internal(self, x: jax.Array) -> PyTree:

        fn = self._init_internal

        if self.trans_symm is None:
            return fn(x)
        else:
            perm = self.perm[:, : len(x)]

            return jax.vmap(fn)(x[perm])

    def ref_forward_with_updates(
        self,
        x: jax.Array,
        x_old: jax.Array,
        nflips: int,
        internal: PyTree,
    ) -> Tuple[jax.Array, PyTree]:

        fn = self._ref_forward_with_updates

        orbs = self.pairing_net(x)
        orbs = self.full_orbs_layer.to_hidden_orbs(orbs)

        if self.trans_symm is None:
            return fn(x, x_old, nflips, internal, orbs)
        else:
            perm = self.perm
            perm_s = perm[:, : len(x)]

            x_symm = x[perm_s]
            x_old = x_old[perm_s]
            orbs = orbs[:, perm]

            fn_vmap = eqx.filter_vmap(fn, in_axes=(0, 0, None, 0, 1))
            psi, internal = fn_vmap(x_symm, x_old, nflips, internal, orbs)
            psi = self.symm_layer(psi, x)
            return psi, internal

    def ref_forward(
        self,
        x: jax.Array,
        x_old: jax.Array,
        nflips: int,
        idx_segment: jax.Array,
        internal: PyTree,
    ) -> jax.Array:

        fn = self._ref_forward

        orbs = self.pairing_net(x)
        orbs = self.full_orbs_layer.to_hidden_orbs(orbs)

        if self.trans_symm is None:
            return fn(x, x_old, nflips, idx_segment, internal, orbs)
        else:
            perm = self.perm
            perm_s = perm[:, : len(x)]

            x_symm = x[perm_s]
            x_old = x_old[:, perm_s]
            orbs = orbs[:, perm]

            fn_vmap = eqx.filter_vmap(fn, in_axes=(0, 1, None, None, 1, 1))
            psi = fn_vmap(x_symm, x_old, nflips, idx_segment, internal, orbs)
            return self.symm_layer(psi, x)

    def _init_internal(self, x: jax.Array) -> PyTree:
        """
        Initialize internal values for given input configurations
        """
        F_full = self.full_orbs_layer.F_full
        idx = _get_fermion_idx(x, self.Nvisible)
        orbs = F_full[idx, :][:, idx]

        inv = jnp.linalg.inv(orbs)
        inv = (inv - inv.T) / 2
        return {"idx": idx, "inv": inv, "psi": pfaffian(orbs)}

    def _ref_forward_with_updates(
        self,
        x: jax.Array,
        x_old: jax.Array,
        nflips: int,
        internal: PyTree,
        orbs: jax.Array,
    ) -> Tuple[jax.Array, PyTree]:
        """
        Accelerated forward pass through local updates and internal quantities.
        This function is designed for sampling.

        :return:
            The evaluated wave function and the updated internal values.
        """
        F_full = self.full_orbs_layer.F_full
        F_hidden_full = self.full_orbs_layer.F_hidden_full

        occ_idx = internal["idx"]
        old_inv = internal["inv"]
        old_psi = internal["psi"]

        flips = (x - x_old) // 2

        old_idx, new_idx = _get_changed_inds(flips, nflips, len(x))

        @jax.vmap
        def idx_to_canon(old_idx):
            return jnp.argwhere(old_idx == occ_idx, size=1)

        old_loc = jnp.ravel(idx_to_canon(old_idx))

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

        inv_times_update = jnp.concatenate((update @ old_inv, old_inv[old_loc]), 0)

        solve = jnp.linalg.solve(low_rank_matrix, inv_times_update)
        inv = old_inv + inv_times_update.T @ solve
        inv = (inv - inv.T) / 2

        sliced_orbs = orbs[:, occ_idx].astype(F_full.dtype)
        full_old_loc = jnp.concatenate(
            (old_loc, jnp.arange(self.Nvisible, self.Nvisible + self.Nhidden))
        )

        b = jnp.zeros([len(occ_idx), self.Nhidden])
        id_inv = -1 * pfa_eye(self.Nhidden // 2, F_full.dtype)
        full_inv = jnp.block([[old_inv, b], [b.T, id_inv]])

        update = jnp.concatenate((update, -1 * sliced_orbs), axis=0)
        update = jnp.concatenate((update, jnp.zeros([len(update), self.Nhidden])), 1)

        mat22 = F_hidden_full - pfa_eye(self.Nhidden // 2, dtype=F_full.dtype)
        mat = jnp.block([[mat, orbs[:, new_idx].T], [-1 * orbs[:, new_idx], mat22]])
        mat = jnp.tril(mat)

        update = array_set(update.T, mat.T, full_old_loc).T

        mat11 = update @ full_inv @ update.T
        mat21 = update @ full_inv[:, full_old_loc]
        mat22 = full_inv[full_old_loc][:, full_old_loc]

        elrm = jnp.block([[mat11, mat21], [-1 * mat21.T, mat22]])
        elrm = elrm - pfa_eye(len(elrm) // 2, F_full.dtype)

        parity = _parity_pfa(new_idx, old_idx, occ_idx)
        psi_mf = old_psi * pfaffian(low_rank_matrix) * parity
        psi = old_psi * pfaffian(elrm) * parity * jnp.power(-1, self.Nhidden // 4)

        idx = occ_idx.at[old_loc].set(new_idx)
        sort = jnp.argsort(idx)

        return psi, {"idx": idx[sort], "inv": inv[sort][:, sort], "psi": psi_mf}

    def _ref_forward(
        self,
        x: jax.Array,
        x_old: jax.Array,
        nflips: int,
        idx_segment: jax.Array,
        internal: jax.Array,
        orbs: jax.Array,
    ) -> jax.Array:
        """
        Accelerated forward pass through local updates and internal quantities.
        This function is designed for local observables.
        """
        F_full = self.full_orbs_layer.F_full
        F_hidden_full = self.full_orbs_layer.F_hidden_full

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

        update = F_full[new_idx][:, occ_idx] - F_full[old_idx][:, occ_idx]
        sliced_orbs = orbs[:, occ_idx].astype(F_full.dtype)
        full_old_loc = jnp.concatenate(
            (old_loc, jnp.arange(self.Nvisible, self.Nvisible + self.Nhidden))
        )

        b = jnp.zeros([len(occ_idx), self.Nhidden])
        id_inv = -1 * pfa_eye(self.Nhidden // 2, F_full.dtype)
        full_inv = jnp.block([[old_inv, b], [b.T, id_inv]])

        update = jnp.concatenate((update, -1 * sliced_orbs), axis=0)
        update = jnp.concatenate((update, jnp.zeros([len(update), self.Nhidden])), 1)

        mat11 = F_full[new_idx][:, new_idx] - F_full[old_idx][:, old_idx]
        mat22 = F_hidden_full - pfa_eye(self.Nhidden // 2, dtype=F_full.dtype)
        mat = jnp.block([[mat11, orbs[:, new_idx].T], [-1 * orbs[:, new_idx], mat22]])
        mat = jnp.tril(mat)

        update = array_set(update.T, mat.T, full_old_loc).T

        mat11 = update @ full_inv @ update.T
        mat21 = update @ full_inv[:, full_old_loc]
        mat22 = full_inv[full_old_loc][:, full_old_loc]

        elrm = jnp.block([[mat11, mat21], [-1 * mat21.T, mat22]])
        elrm = elrm - pfa_eye(len(elrm) // 2, F_full.dtype)

        parity = _parity_pfa(new_idx, old_idx, occ_idx)
        psi = old_psi * pfaffian(elrm) * parity * jnp.power(-1, self.Nhidden // 4)

        return psi
