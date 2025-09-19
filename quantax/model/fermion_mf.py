from __future__ import annotations
from typing import Optional, Tuple, NamedTuple, Union
from jaxtyping import PyTree
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import lrux
from ..global_defs import (
    get_sites,
    get_subkeys,
    get_default_dtype,
    PARTICLE_TYPE,
)
from ..sites import Lattice
from ..nn import RefModel, fermion_idx, changed_inds, permute_sign
from ..utils import array_set, LogArray, PsiArray


class MF_Internal(NamedTuple):
    idx: Union[jax.Array, Tuple[jax.Array, jax.Array]]
    inv: Union[jax.Array, lrux.DetCarrier, lrux.PfCarrier]
    psi: PsiArray


def _check_dtype(
    x: Union[None, jax.Array, Tuple[jax.Array, ...]],
    dtype: Optional[jnp.dtype],
    preferred_element_type: Optional[jnp.dtype],
) -> Tuple[jnp.dtype, jnp.dtype, bool, bool]:
    if isinstance(x, tuple):
        x = x[0]
    if dtype is None:
        dtype = get_default_dtype() if x is None else x.dtype
    if preferred_element_type is None:
        preferred_element_type = dtype
    is_dtype_cpl = jnp.issubdtype(dtype, jnp.complexfloating)
    is_comp_cpl = jnp.issubdtype(preferred_element_type, jnp.complexfloating)
    holomorphic = is_dtype_cpl and is_comp_cpl
    real_to_cpl = is_comp_cpl and not is_dtype_cpl
    return dtype, preferred_element_type, holomorphic, real_to_cpl


def _init_spinless_orbs(preferred_element_type: jnp.dtype) -> jax.Array:
    sites = get_sites()
    if isinstance(sites, Lattice):
        is_comp_cpl = jnp.issubdtype(preferred_element_type, jnp.complexfloating)
        orbitals = sites.orbitals(return_real=not is_comp_cpl)
        orbitals = jnp.asarray(orbitals)
    else:
        shape = (sites.Nsites, sites.Nsites)
        orbitals = jr.normal(get_subkeys(), shape, preferred_element_type)
        orbitals, R = jnp.linalg.qr(orbitals)
    return orbitals


def _to_comp_mat(x: jax.Array, preferred_element_type: jnp.dtype) -> jax.Array:
    is_dtype_cpl = jnp.issubdtype(x.dtype, jnp.complexfloating)
    is_comp_cpl = jnp.issubdtype(preferred_element_type, jnp.complexfloating)
    if is_comp_cpl and not is_dtype_cpl:
        x = jax.lax.complex(x[0], x[1])
    return x.astype(preferred_element_type)


class GeneralDet(RefModel):
    U: jax.Array
    dtype: jnp.dtype
    preferred_element_type: jnp.dtype
    holomorphic: bool

    def __init__(
        self,
        U: Optional[jax.Array] = None,
        dtype: Optional[jnp.dtype] = None,
        preferred_element_type: Optional[jnp.dtype] = None,
    ):
        sites = get_sites()
        if sites.Ntotal is None:
            raise ValueError("Determinant should have a fixed amount of particles.")

        self.dtype, self.preferred_element_type, self.holomorphic, real_to_cpl = (
            _check_dtype(U, dtype, preferred_element_type)
        )

        if U is None:
            U = _init_spinless_orbs(self.preferred_element_type)
            if sites.is_spinful:
                Nparticles = sites.Nparticles
                if isinstance(Nparticles, int):
                    Nhalf = Nparticles // 2
                    Nup, Ndn = Nhalf, Nhalf
                else:
                    Nup, Ndn = sites.Nparticles
                Uup = U[:, :Nup]
                Udn = U[:, :Ndn]
                zeros_up = jnp.zeros((Uup.shape[0], Udn.shape[1]), dtype=U.dtype)
                zeros_dn = jnp.zeros((Udn.shape[0], Uup.shape[1]), dtype=U.dtype)
                U = jnp.block([[Uup, zeros_up], [zeros_dn, Udn]])
            else:
                U = U[:, : sites.Ntotal]
        else:
            shape = (sites.Nfmodes, sites.Ntotal)
            if U.shape != shape:
                raise ValueError(f"Expected U to have shape {shape}, but got {U.shape}")

        if real_to_cpl:
            U = jnp.stack([jnp.real(U), jnp.imag(U)], axis=0)
        self.U = U.astype(self.dtype)

    @property
    def U_full(self) -> jax.Array:
        """
        Returns the full orbital matrix U.
        """
        return _to_comp_mat(self.U, self.preferred_element_type)

    def normalize(self) -> GeneralDet:
        """Return a new GeneralDet instance with orthonormalized orbitals."""
        U_full = self.U_full
        Q, R = jnp.linalg.qr(U_full)
        return GeneralDet(Q, self.dtype, self.preferred_element_type)

    def __call__(self, s: jax.Array) -> PsiArray:
        idx = fermion_idx(s)
        sign, logabs = jnp.linalg.slogdet(self.U_full[idx, :])
        return LogArray(sign, logabs)

    def init_internal(self, s: jax.Array) -> MF_Internal:
        """
        Initialize internal values for given input configurations
        """
        idx = fermion_idx(s)
        orbs = self.U_full[idx, :]
        inv = jnp.linalg.inv(orbs)
        sign, logabs = jnp.linalg.slogdet(orbs)
        psi = LogArray(sign, logabs)
        return MF_Internal(idx, inv, psi)

    def ref_forward(
        self,
        s: jax.Array,
        s_old: jax.Array,
        nflips: int,
        internal: MF_Internal,
        return_update: bool = False,
    ) -> Union[LogArray, Tuple[LogArray, MF_Internal]]:
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
            psi = internal.psi * (ratio * sign)
            internal = MF_Internal(idx, inv, psi)
            return psi, internal
        else:
            ratio = lrux.det_lru(internal.inv, row_update.T, row_update_idx, False)
            psi = internal.psi * (ratio * sign)
            return psi


class RestrictedDet(eqx.Module):
    U: jax.Array
    dtype: jnp.dtype
    preferred_element_type: jnp.dtype
    holomorphic: bool

    def __init__(
        self,
        U: Optional[jax.Array] = None,
        dtype: Optional[jnp.dtype] = None,
        preferred_element_type: Optional[jnp.dtype] = None,
    ):
        sites = get_sites()
        if not sites.is_spinful:
            raise ValueError("RestrictedDet only works for spinful systems.")

        Nparticles = sites.Nparticles
        if (not isinstance(Nparticles, tuple)) or Nparticles[0] != Nparticles[1]:
            raise ValueError(
                "RestrictedDet only works for equal number of spin-up and spin-down particles."
                f"Got Nparticles={Nparticles}."
            )
        Nup = Nparticles[0]

        self.dtype, self.preferred_element_type, self.holomorphic, real_to_cpl = (
            _check_dtype(U, dtype, preferred_element_type)
        )

        if U is None:
            U = _init_spinless_orbs(self.preferred_element_type)
            U = U[:, :Nup]
        else:
            shape = (sites.Nsites, Nup)
            if U.shape != shape:
                raise ValueError(f"Expected U to have shape {shape}, but got {U.shape}")

        if real_to_cpl:
            U = jnp.stack([jnp.real(U), jnp.imag(U)], axis=0)
        self.U = U.astype(self.dtype)

    @property
    def U_full(self) -> jax.Array:
        """
        Returns the full orbital matrix U.
        """
        return _to_comp_mat(self.U, self.preferred_element_type)

    def normalize(self) -> RestrictedDet:
        """Return a new RestrictedDet instance with orthonormalized orbitals."""
        U_full = self.U_full
        Q, R = jnp.linalg.qr(U_full)
        return RestrictedDet(Q, self.dtype, self.preferred_element_type)

    def __call__(self, s: jax.Array) -> LogArray:
        idx_up, idx_dn = fermion_idx(s, separate_spins=True)
        U_full = self.U_full
        sign_up, logabs_up = jnp.linalg.slogdet(U_full[idx_up, :])
        sign_dn, logabs_dn = jnp.linalg.slogdet(U_full[idx_dn, :])
        return LogArray(sign_up, logabs_up) * LogArray(sign_dn, logabs_dn)


class UnrestrictedDet(eqx.Module):
    U: Tuple[jax.Array, jax.Array]
    dtype: jnp.dtype
    preferred_element_type: jnp.dtype
    holomorphic: bool

    def __init__(
        self,
        U: Optional[Tuple[jax.Array, jax.Array]] = None,
        dtype: Optional[jnp.dtype] = None,
        preferred_element_type: Optional[jnp.dtype] = None,
    ):
        sites = get_sites()
        if not sites.is_spinful:
            raise ValueError("UnrestrictedDet only works for spinful systems.")

        if not isinstance(sites.Nparticles, tuple):
            raise ValueError(
                "RestrictedDet requires specified spin-up and spin-down particle numbers."
            )
        Nup, Ndn = sites.Nparticles

        self.dtype, self.preferred_element_type, self.holomorphic, real_to_cpl = (
            _check_dtype(U, dtype, preferred_element_type)
        )

        shape_up = (sites.Nsites, Nup)
        shape_dn = (sites.Nsites, Ndn)
        if U is None:
            U = _init_spinless_orbs(self.preferred_element_type)
            Uup = U[:, :Nup]
            Udn = U[:, :Ndn]
        else:
            Uup, Udn = U
            if Uup.shape != shape_up:
                raise ValueError(f"Expected Uup shape {shape_up}, but got {Uup.shape}")
            if Udn.shape != shape_dn:
                raise ValueError(f"Expected Udn shape {shape_dn}, but got {Udn.shape}")

        if real_to_cpl:
            Uup = jnp.stack([jnp.real(Uup), jnp.imag(Uup)], axis=0)
            Udn = jnp.stack([jnp.real(Udn), jnp.imag(Udn)], axis=0)
        Uup = Uup.astype(self.dtype)
        Udn = Udn.astype(self.dtype)
        self.U = (Uup, Udn)

    @property
    def U_full(self) -> jax.Array:
        """
        Returns the full orbital matrix U.
        """
        Uup = _to_comp_mat(self.U[0], self.preferred_element_type)
        Udn = _to_comp_mat(self.U[1], self.preferred_element_type)
        return Uup, Udn

    def normalize(self) -> UnrestrictedDet:
        """Return a new UnrestrictedDet instance with orthonormalized orbitals."""
        Uup, Udn = self.U_full
        Qup, R = jnp.linalg.qr(Uup)
        Qdn, R = jnp.linalg.qr(Udn)
        return UnrestrictedDet((Qup, Qdn), self.dtype, self.preferred_element_type)

    def __call__(self, s: jax.Array) -> LogArray:
        idx_up, idx_dn = fermion_idx(s, separate_spins=True)
        Uup, Udn = self.U_full
        sign_up, logabs_up = jnp.linalg.slogdet(Uup[idx_up, :])
        sign_dn, logabs_dn = jnp.linalg.slogdet(Udn[idx_dn, :])
        return LogArray(sign_up, logabs_up) * LogArray(sign_dn, logabs_dn)


class MultiDet(eqx.Module):
    """
    The multi-determinant wavefunction
    """

    ndets: int
    U: jax.Array
    coeffs: jax.Array
    dtype: jnp.dtype
    preferred_element_type: jnp.dtype
    holomorphic: bool

    def __init__(
        self,
        ndets: int = 4,
        U: Optional[jax.Array] = None,
        coeffs: Optional[jax.Array] = None,
        dtype: Optional[jnp.dtype] = None,
        preferred_element_type: Optional[jnp.dtype] = None,
    ):
        sites = get_sites()
        if sites.Ntotal is None:
            raise ValueError("Determinant should have a fixed amount of particles.")

        self.ndets = ndets
        self.dtype, self.preferred_element_type, self.holomorphic, real_to_cpl = (
            _check_dtype(U, dtype, preferred_element_type)
        )

        shape = (ndets, sites.Nfmodes, sites.Ntotal)
        if U is None:
            U = _init_spinless_orbs(self.preferred_element_type)
            if sites.is_spinful:
                Nparticles = sites.Nparticles
                if isinstance(Nparticles, int):
                    Nhalf = Nparticles // 2
                    Nup, Ndn = Nhalf, Nhalf
                else:
                    Nup, Ndn = sites.Nparticles
                Uup = U[:, :Nup]
                Udn = U[:, :Ndn]
                zeros_up = jnp.zeros((Uup.shape[0], Udn.shape[1]), dtype=U.dtype)
                zeros_dn = jnp.zeros((Udn.shape[0], Uup.shape[1]), dtype=U.dtype)
                U = jnp.block([[Uup, zeros_up], [zeros_dn, Udn]])
            else:
                U = U[:, : sites.Ntotal]
            U = jnp.tile(U, (ndets, 1, 1))
        else:
            if U.ndim == 2:
                U = jnp.tile(U, (ndets, 1, 1))
            if U.shape != shape:
                raise ValueError(f"Expected U to have shape {shape}, but got {U.shape}")

        if real_to_cpl:
            U = jnp.stack([jnp.real(U), jnp.imag(U)], axis=0)
        self.U = U.astype(self.dtype)

        if coeffs is None:
            coeffs = jnp.ones((ndets,), dtype=self.dtype) / ndets
        self.coeffs = coeffs.astype(self.dtype)

    @property
    def U_full(self) -> jax.Array:
        """
        Returns the full orbital matrix U.
        """
        return _to_comp_mat(self.U, self.preferred_element_type)

    def normalize(self) -> MultiDet:
        """Return a new MultiDet instance with orthonormalized orbitals."""
        U_full = self.U_full
        Q, R = jnp.linalg.qr(U_full)
        coeffs = self.coeffs * jnp.prod(jnp.diagonal(R, axis1=1, axis2=2), axis=1)
        return MultiDet(self.ndets, Q, coeffs, self.dtype, self.preferred_element_type)

    def __call__(self, s: jax.Array) -> LogArray:
        idx = fermion_idx(s)
        sign, logabs = jnp.linalg.slogdet(self.U_full[:, idx, :])
        psi = LogArray(sign, logabs)
        return (psi * self.coeffs).sum()


def _init_paired_orbs(preferred_element_type: jnp.dtype) -> jax.Array:
    U1 = _init_spinless_orbs(preferred_element_type)
    if jnp.issubdtype(preferred_element_type, jnp.complexfloating):
        U2 = U1.conj()
    else:
        U2 = U1
    return U1 @ U2.T


class GeneralPf(RefModel):
    F: jax.Array
    sublattice: Optional[tuple]
    dtype: jnp.dtype
    preferred_element_type: jnp.dtype
    holomorphic: bool

    def __init__(
        self,
        F: Optional[jax.Array] = None,
        sublattice: Optional[tuple] = None,
        dtype: Optional[jnp.dtype] = None,
        preferred_element_type: Optional[jnp.dtype] = None,
    ):
        sites = get_sites()
        self.sublattice = sublattice
        self.dtype, self.preferred_element_type, self.holomorphic, real_to_cpl = (
            _check_dtype(F, dtype, preferred_element_type)
        )

        shape = (sites.Nfmodes, sites.Nfmodes)
        if F is None:
            if sites.is_spinful:
                F = _init_paired_orbs(self.preferred_element_type)
                zeros = jnp.zeros_like(F)
                F = jnp.block([[zeros, F], [-F.T, zeros]])
            else:
                F = jr.normal(get_subkeys(), shape, self.preferred_element_type)
                F = (F - F.T) / 2
        else:
            if F.shape != shape:
                raise ValueError(f"Expected F to have shape {shape}, but got {F.shape}")

        if real_to_cpl:
            F = jnp.stack([jnp.real(F), jnp.imag(F)], axis=0)
        self.F = F.astype(self.dtype)

    @property
    def F_full(self) -> jax.Array:
        F = _to_comp_mat(self.F, self.preferred_element_type)
        return (F - F.T) / 2

    def __call__(self, x: jax.Array) -> LogArray:
        idx = fermion_idx(x)
        sign, logabs = lrux.slogpf(self.F_full[idx, :][:, idx])
        return LogArray(sign, logabs)

    def init_internal(self, x: jax.Array) -> MF_Internal:
        """
        Initialize internal values for given input configurations
        """
        idx = fermion_idx(x)
        orbs = self.F_full[idx, :][:, idx]
        inv = jnp.linalg.inv(orbs)
        inv = (inv - inv.T) / 2  # Ensure antisymmetry
        sign, logabs = lrux.slogpf(orbs)
        psi = LogArray(sign, logabs)
        return MF_Internal(idx, inv, psi)

    def ref_forward(
        self,
        s: jax.Array,
        s_old: jax.Array,
        nflips: int,
        internal: MF_Internal,
        return_update: bool = False,
    ) -> Union[LogArray, Tuple[LogArray, MF_Internal]]:
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
            psi = internal.psi * (ratio * sign)
            internal = MF_Internal(idx, inv, psi)
            return psi, internal
        else:
            ratio = lrux.pf_lru(internal.inv, u, False)
            psi = internal.psi * (ratio * sign)
            return psi


class SingletPair(RefModel):
    F: jax.Array
    sublattice: Optional[tuple]
    dtype: jnp.dtype
    preferred_element_type: jnp.dtype
    holomorphic: bool

    def __init__(
        self,
        F: Optional[jax.Array] = None,
        sublattice: Optional[tuple] = None,
        dtype: Optional[jnp.dtype] = None,
        preferred_element_type: Optional[jnp.dtype] = None,
    ):
        sites = get_sites()
        if not sites.is_spinful:
            raise ValueError("SingletPair only works for spinful systems.")

        self.sublattice = sublattice
        self.dtype, self.preferred_element_type, self.holomorphic, real_to_cpl = (
            _check_dtype(F, dtype, preferred_element_type)
        )

        shape = (sites.Nsites, sites.Nsites)
        if F is None:
            F = _init_paired_orbs(self.preferred_element_type)
        else:
            if F.shape != shape:
                raise ValueError(f"Expected F to have shape {shape}, but got {F.shape}")

        if real_to_cpl:
            F = jnp.stack([jnp.real(F), jnp.imag(F)], axis=0)
        self.F = F.astype(self.dtype)

    @property
    def F_full(self) -> jax.Array:
        return _to_comp_mat(self.F, self.preferred_element_type)

    def __call__(self, s: jax.Array) -> LogArray:
        idx_up, idx_dn = fermion_idx(s, separate_spins=True)
        if idx_up.size != idx_dn.size:
            sign = jnp.array(0.0, dtype=self.preferred_element_type)
            logabs = jnp.array(-jnp.inf, dtype=self.preferred_element_type)
        else:
            F_full = self.F_full[idx_up, :][:, idx_dn]
            sign, logabs = jnp.linalg.slogdet(F_full)
            n = F_full.shape[0]
            sign *= (-1) ** (n * (n - 1) // 2)
        return LogArray(sign, logabs)

    def init_internal(self, s: jax.Array) -> MF_Internal:
        """
        Initialize internal values for given input configurations
        """
        sites = get_sites()

        if sites.particle_type == PARTICLE_TYPE.spinful_fermion:
            raise NotImplementedError(
                "Low-rank update is not implemented for `SingletPair` with spinful fermions,"
                "because the number of spin-up and spin-down hoppings is not fixed."
            )

        idx_up, idx_dn = fermion_idx(s, separate_spins=True)
        F_full = self.F_full[idx_up, :][:, idx_dn]
        inv = jnp.linalg.inv(F_full)
        sign, logabs = jnp.linalg.slogdet(F_full)
        psi = LogArray(sign, logabs)
        return MF_Internal((idx_up, idx_dn), inv, psi)

    def ref_forward(
        self,
        s: jax.Array,
        s_old: jax.Array,
        nflips: int,
        internal: MF_Internal,
        return_update: bool = False,
    ) -> Union[LogArray, Tuple[LogArray, MF_Internal]]:
        """
        Accelerated forward pass through local updates and internal quantities.
        """
        sites = get_sites()
        if sites.particle_type == PARTICLE_TYPE.spinful_fermion:
            raise NotImplementedError(
                "Low-rank update is not implemented for `SingletPair` with spinful fermions,"
                "because the number of spin-up and spin-down hoppings is not fixed."
            )

        idx_flip_dn, idx_flip_up = changed_inds(s, s_old, nflips)
        idx_flip_dn -= sites.Nparticles[0]  # Convert to site index
        idx_up, idx_dn = internal.idx
        sign_up = permute_sign(idx_up, idx_flip_dn, idx_flip_up)
        sign_dn = permute_sign(idx_dn, idx_flip_up, idx_flip_dn)
        F = self.F_full

        updated = jnp.isin(idx_up, idx_flip_dn)
        row_update_idx = jnp.flatnonzero(updated, size=nflips, fill_value=idx_up.size)
        updated = jnp.isin(idx_dn, idx_flip_up)
        col_update_idx = jnp.flatnonzero(updated, size=nflips, fill_value=idx_dn.size)

        row_update = F[idx_flip_up][:, idx_dn] - F[idx_flip_dn][:, idx_dn]
        idx_up = idx_up.at[row_update_idx].set(idx_flip_up)
        idx_dn = idx_dn.at[col_update_idx].set(idx_flip_dn)
        col_update = F[idx_up][:, idx_flip_dn] - F[idx_up][:, idx_flip_up]

        # See https://chenao-phys.github.io/lrux/lrux.det_lru.html#lrux.det_lru
        u = (row_update.T, col_update_idx)
        v = (col_update, row_update_idx)

        if return_update:
            ratio, inv = lrux.det_lru(internal.inv, u, v, True)
            psi = internal.psi * (ratio * sign_up * sign_dn)
            internal = MF_Internal((idx_up, idx_dn), inv, psi)
            return psi, internal
        else:
            ratio = lrux.det_lru(internal.inv, u, v, False)
            psi = internal.psi * (ratio * sign_up * sign_dn)
            return psi


class MultiPf(eqx.Module):
    npfs: int
    F: jax.Array
    sublattice: Optional[tuple]
    dtype: jnp.dtype
    preferred_element_type: jnp.dtype
    holomorphic: bool

    def __init__(
        self,
        npfs: int = 4,
        F: Optional[jax.Array] = None,
        sublattice: Optional[tuple] = None,
        dtype: Optional[jnp.dtype] = None,
        preferred_element_type: Optional[jnp.dtype] = None,
    ):
        sites = get_sites()
        self.npfs = npfs
        self.sublattice = sublattice
        self.dtype, self.preferred_element_type, self.holomorphic, real_to_cpl = (
            _check_dtype(F, dtype, preferred_element_type)
        )

        shape = (npfs, sites.Nfmodes, sites.Nfmodes)
        if F is None:
            if sites.is_spinful:
                F = _init_paired_orbs(self.preferred_element_type)
                zeros = jnp.zeros_like(F)
                F = jnp.block([[zeros, F], [-F.mT, zeros]])
                F = jnp.tile(F, (npfs, 1, 1))
            else:
                F = jr.normal(get_subkeys(), shape, self.preferred_element_type)
                F = (F - F.mT) / 2
        else:
            if F.shape != shape:
                raise ValueError(f"Expected F to have shape {shape}, but got {F.shape}")

        if real_to_cpl:
            F = jnp.stack([jnp.real(F), jnp.imag(F)], axis=0)
        self.F = F.astype(self.dtype)

    @property
    def F_full(self) -> jax.Array:
        F = _to_comp_mat(self.F, self.preferred_element_type)
        return (F - F.mT) / 2

    def __call__(self, x: jax.Array) -> LogArray:
        idx = fermion_idx(x)
        sign, logabs = lrux.slogpf(self.F_full[:, idx, :][:, :, idx])
        return LogArray(sign, logabs).sum()
