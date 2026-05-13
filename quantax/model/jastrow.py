from __future__ import annotations
from typing import Any, Callable, Literal, overload
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from ..nn import RefModel, RawInputLayer
from ..symmetry import Translation
from ..symmetry.symmetry import _permutation_sign
from ..global_defs import get_lattice
from ..utils import PsiArray
from .fermion_mf import MF_Internal


def _to_sub_term(x: jax.Array, sublattice: tuple[int, ...]) -> jax.Array:
    remaining_dims = x.shape[1:]
    x = x.reshape(get_lattice().shape[1:] + remaining_dims)
    for axis, subl in enumerate(sublattice):
        x = x.take(np.arange(subl), axis)
    x = x.reshape(-1, *remaining_dims)
    return x


def _get_sublattice_spins(
    s: jax.Array, trans_symm: Translation | None, sublattice: tuple[int, ...] | None
) -> jax.Array:
    if trans_symm is None or sublattice is None:
        return s[..., None, :]

    perm = _to_sub_term(trans_symm._perm, sublattice)

    Nmodes = trans_symm.Nmodes
    batch = s.shape[:-1]
    s = s.reshape(-1, Nmodes)
    s_symm = s[:, perm]
    s_symm = s_symm.reshape(*batch, -1, perm.shape[0], Nmodes)
    s_symm = jnp.swapaxes(s_symm, -3, -2)
    return s_symm.reshape(*batch, perm.shape[0], -1)


def _sub_symmetrize(
    x_sub: PsiArray,
    s: jax.Array,
    trans_symm: Translation | None,
    sublattice: tuple[int, ...] | None,
) -> PsiArray:
    if trans_symm is None or sublattice is None:
        return x_sub[0]

    eigval = _to_sub_term(trans_symm._character, sublattice) / trans_symm.nsymm

    if trans_symm.is_fermion:
        perm = _to_sub_term(trans_symm._perm, sublattice)
        perm_sign = _to_sub_term(trans_symm._perm_sign, sublattice)
        sign = _permutation_sign(s, perm, perm_sign)
        eigval *= sign

    eigval = eigval.astype(x_sub.dtype)
    return (x_sub * eigval).sum()


class _JastrowFermionLayer(RawInputLayer):
    fermion_mf: RefModel
    trans_symm: Translation | None
    sublattice: tuple[int, ...] | None

    def __init__(self, fermion_mf, trans_symm):
        self.fermion_mf = fermion_mf
        self.trans_symm = trans_symm
        if hasattr(fermion_mf, "sublattice"):
            if isinstance(fermion_mf.sublattice, Translation):
                vectors = fermion_mf.sublattice.vectors
                sublattice = [1] * get_lattice().ndim
                for vec in vectors:
                    if np.sum(vec != 0) != 1:
                        raise ValueError(
                            "Sublattice vectors must be along one axis for Jastrow layer."
                        )
                    else:
                        axis = np.flatnonzero(vec)[0]
                        sublattice[axis] = vec[axis].item()
                sublattice = tuple(sublattice)
            else:
                sublattice = fermion_mf.sublattice
        else:
            sublattice = None

        if sublattice is None and trans_symm is not None:
            sublattice = get_lattice().shape[1:]
        self.sublattice = sublattice

    def get_sublattice_spins(self, s: jax.Array) -> jax.Array:
        return _get_sublattice_spins(s, self.trans_symm, self.sublattice)

    def sub_symmetrize(self, x_net: PsiArray, x_mf: PsiArray, s: jax.Array) -> PsiArray:
        if self.trans_symm is None or self.sublattice is None:
            return x_mf[0] * x_net.mean()

        x_net = x_net.reshape(-1, *get_lattice().shape[1:]).mean(axis=0)
        for axis, subl in enumerate(self.sublattice):
            new_shape = x_net.shape[:axis] + (-1, subl) + x_net.shape[axis + 1 :]
            x_net = x_net.reshape(new_shape)
            x_net = x_net.mean(axis=axis)
        return _sub_symmetrize(
            x_mf * x_net.flatten(), s, self.trans_symm, self.sublattice
        )

    def __call__(self, x: PsiArray, s: jax.Array) -> PsiArray:
        if x.size > 1:
            x = x.reshape(-1, get_lattice().ncells).mean(axis=0)

        s_symm = self.get_sublattice_spins(s)
        x_mf = jax.vmap(self.fermion_mf)(s_symm)
        return self.sub_symmetrize(x, x_mf, s)


class NeuralJastrow(RefModel):
    net: Callable[[jax.Array], PsiArray]
    fermion_layer: _JastrowFermionLayer
    holomorphic: bool
    trans_symm: Translation | None
    sublattice: tuple[int, ...] | None

    def __init__(
        self,
        net: Callable[[jax.Array], PsiArray],
        fermion_mf: RefModel,
        trans_symm: Translation | None = None,
    ):
        self.net = net
        self.fermion_layer = _JastrowFermionLayer(fermion_mf, trans_symm)
        self.trans_symm = trans_symm
        self.sublattice = self.fermion_layer.sublattice
        net_holomorphic = getattr(net, "holomorphic", False)
        mf_holomorphic = getattr(fermion_mf, "holomorphic", False)
        self.holomorphic = net_holomorphic and mf_holomorphic

    def __call__(self, s: jax.Array) -> PsiArray:
        x = self.net(s)
        return self.fermion_layer(x, s)

    @property
    def fermion_mf(self) -> RefModel:
        return self.fermion_layer.fermion_mf

    def get_sublattice_spins(self, x: jax.Array) -> jax.Array:
        return self.fermion_layer.get_sublattice_spins(x)

    def sub_symmetrize(self, x_net: PsiArray, x_mf: PsiArray, s: jax.Array) -> PsiArray:
        return self.fermion_layer.sub_symmetrize(x_net, x_mf, s)

    def init_internal(self, s: jax.Array) -> tuple[PsiArray, MF_Internal]:
        """
        Initialize internal values for given input configurations
        """
        x_net = self.net(s)
        s_symm = self.get_sublattice_spins(s)
        x_mf, internal = jax.vmap(self.fermion_mf.init_internal)(s_symm)
        return self.sub_symmetrize(x_net, x_mf, s), internal

    @property
    def required_update_modes(self) -> tuple[str, ...]:
        """
        The required update modes for accelerated ref_forward pass.
        """
        return self.fermion_mf.required_update_modes

    @overload
    def ref_forward(
        self,
        s: jax.Array,
        s_old: jax.Array,
        update_mode: dict[str, Any],
        internal: MF_Internal,
        return_update: Literal[False] = False,
    ) -> PsiArray: ...

    @overload
    def ref_forward(
        self,
        s: jax.Array,
        s_old: jax.Array,
        update_mode: dict[str, Any],
        internal: MF_Internal,
        return_update: Literal[True],
    ) -> tuple[PsiArray, MF_Internal]: ...

    def ref_forward(
        self,
        s: jax.Array,
        s_old: jax.Array,
        update_mode: dict[str, Any],
        internal: MF_Internal,
        return_update: bool = False,
    ) -> PsiArray | tuple[PsiArray, MF_Internal]:
        x_net = self.net(s)
        if x_net.size > 1:
            x_net = x_net.reshape(-1, get_lattice().ncells).mean(axis=0)

        s_symm = self.get_sublattice_spins(s)
        s_old = self.get_sublattice_spins(s_old)
        fn_vmap = eqx.filter_vmap(
            self.fermion_mf.ref_forward, in_axes=(0, 0, None, 0, None)
        )
        if return_update:
            x_mf, internal = fn_vmap(s_symm, s_old, update_mode, internal, True)
            psi = self.sub_symmetrize(x_net, x_mf, s)
            return psi, internal
        else:
            x_mf = fn_vmap(s_symm, s_old, update_mode, internal, False)
            psi = self.sub_symmetrize(x_net, x_mf, s)
            return psi
