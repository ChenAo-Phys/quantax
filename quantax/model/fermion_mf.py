from __future__ import annotations
from typing import Optional, Tuple, NamedTuple, Union
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import lrux
from ..global_defs import (
    get_sites,
    get_lattice,
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
    out_dtype: Optional[jnp.dtype],
) -> Tuple[jnp.dtype, jnp.dtype, bool, bool]:
    if isinstance(x, tuple):
        x = x[0]
    if dtype is None:
        dtype = get_default_dtype() if x is None else x.dtype
    if out_dtype is None:
        out_dtype = dtype
    is_dtype_cpl = jnp.issubdtype(dtype, jnp.complexfloating)
    is_comp_cpl = jnp.issubdtype(out_dtype, jnp.complexfloating)
    holomorphic = is_dtype_cpl and is_comp_cpl
    real_to_cpl = is_comp_cpl and not is_dtype_cpl
    return dtype, out_dtype, holomorphic, real_to_cpl


def _init_spinless_orbs(
    out_dtype: jnp.dtype,
    sublattice: Optional[Tuple[int, ...]] = None,
    return_kpts: bool = False,
) -> jax.Array:
    sites = get_sites()
    if isinstance(sites, Lattice):
        is_comp_cpl = jnp.issubdtype(out_dtype, jnp.complexfloating)
        use_real = not is_comp_cpl
        if return_kpts:
            orbitals, kpts = sites.orbitals(sublattice, use_real, return_kpts)
            kpts = jnp.asarray(kpts, dtype=jnp.int32)
        else:
            orbitals = sites.orbitals(sublattice, use_real, return_kpts)
        orbitals = jnp.asarray(orbitals, dtype=out_dtype)
        scale = jnp.std(orbitals) * 0.1
        perturb = jr.normal(get_subkeys(), orbitals.shape, out_dtype) * scale
        orbitals = orbitals + perturb
        if return_kpts:
            return orbitals, kpts
        else:
            return orbitals
    else:
        if sublattice is not None or return_kpts:
            raise ValueError(
                "Sublattice and k-points are only defined for lattice systems."
            )
        shape = (sites.Nsites, sites.Nsites)
        orbitals = jr.normal(get_subkeys(), shape, out_dtype)
        orbitals, R = jnp.linalg.qr(orbitals)
        return orbitals


def _to_comp_mat(x: jax.Array, out_dtype: jnp.dtype) -> jax.Array:
    is_dtype_cpl = jnp.issubdtype(x.dtype, jnp.complexfloating)
    is_comp_cpl = jnp.issubdtype(out_dtype, jnp.complexfloating)
    if is_comp_cpl and not is_dtype_cpl:
        x = jax.lax.complex(x[0], x[1])
    return x.astype(out_dtype)


class GeneralDet(RefModel):
    r"""
    A general determinant wavefunction :math:`\psi(n) = \mathrm{det}(n \star U)`.
    """

    U: jax.Array
    sublattice: Optional[Tuple[int, ...]]
    kpts: Optional[jax.Array]
    dtype: jnp.dtype
    out_dtype: jnp.dtype
    holomorphic: bool

    def __init__(
        self,
        U: Optional[jax.Array] = None,
        sublattice: Optional[Tuple[int, ...]] = None,
        kpts: Optional[jax.Array] = None,
        dtype: Optional[jnp.dtype] = None,
        out_dtype: Optional[jnp.dtype] = None,
    ):
        """
        Initialize the GeneralDet model.

        :param U:
            The orbital matrix. If None, it will be initialized as a Fermi sea.

        :param sublattice:
            The sublattice shape.

        :param kpts:
            The k-points for each orbital in the sublattice structure.

        :param dtype:
            The data type for orbital parameters.

        :param out_dtype:
            The data type for computations and outputs. When dtype is real and out_dtype is complex,
            U stores the real and imaginary parts using real numbers.
        """
        sites = get_sites()
        if sites.Ntotal is None:
            raise ValueError("Determinant should have a fixed amount of particles.")

        self.dtype, self.out_dtype, self.holomorphic, real_to_cpl = _check_dtype(
            U, dtype, out_dtype
        )

        if U is None:
            if kpts is not None:
                raise NotImplementedError(
                    "Unspecified orbitals with k-points is not implemented."
                )

            if sublattice is None:
                U = _init_spinless_orbs(self.out_dtype)
            else:
                U, kpts = _init_spinless_orbs(self.out_dtype, sublattice, True)

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
                if kpts is not None:
                    kpts = jnp.concatenate([kpts[:Nup], kpts[:Ndn]], axis=0)
            else:
                U = U[:, : sites.Ntotal]
                if kpts is not None:
                    kpts = kpts[: sites.Ntotal]
            if real_to_cpl:
                U = jnp.stack([jnp.real(U), jnp.imag(U)], axis=0)
        else:
            if sublattice is None:
                Nfmodes = sites.Nfmodes
            else:
                Nfmodes = np.prod(sublattice).item()
                if sites.is_spinful:
                    Nfmodes *= 2
            shape = (Nfmodes, sites.Ntotal)
            if real_to_cpl:
                shape = (2,) + shape
                if jnp.issubdtype(U.dtype, jnp.complexfloating):
                    U = jnp.stack([jnp.real(U), jnp.imag(U)], axis=0)
            if U.shape != shape:
                raise ValueError(f"Expected U to have shape {shape}, but got {U.shape}")

        self.sublattice = sublattice
        self.kpts = kpts
        self.U = U.astype(self.dtype)

    @property
    def U_full(self) -> jax.Array:
        """
        Returns the full orbital matrix U.
        """
        U = _to_comp_mat(self.U, self.out_dtype)
        if self.sublattice is not None:
            lattice = get_lattice()
            if lattice.is_spinful:
                Uup, Udn = jnp.split(U, 2, axis=0)
                Uup = lattice.restore_full_orbitals(Uup, self.sublattice, self.kpts)
                Udn = lattice.restore_full_orbitals(Udn, self.sublattice, self.kpts)
                U = jnp.concatenate([Uup, Udn], axis=0)
            else:
                U = lattice.restore_full_orbitals(U, self.sublattice, self.kpts)
        return U

    def __call__(self, s: jax.Array) -> PsiArray:
        """
        Evaluate the wavefunction on given input configurations.
        """
        idx = fermion_idx(s)
        sign, logabs = jnp.linalg.slogdet(self.U_full[idx, :])
        return LogArray(sign, logabs)

    def init_internal(self, s: jax.Array) -> MF_Internal:
        """
        Initialize internal values for given input configurations.
        See `~quantax.nn.RefModel` for details.
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
        See `~quantax.nn.RefModel` for details.
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
    r"""
    Restricted determinant wavefunction
    :math:`\psi(n) = \mathrm{det}(n_\uparrow \star U) \mathrm{det}(n_\downarrow \star U)`.
    Only works for spinful systems with equal number of spin-up and spin-down particles.
    """

    U: jax.Array
    dtype: jnp.dtype
    out_dtype: jnp.dtype
    holomorphic: bool

    def __init__(
        self,
        U: Optional[jax.Array] = None,
        dtype: Optional[jnp.dtype] = None,
        out_dtype: Optional[jnp.dtype] = None,
    ):
        """
        Initialize the RestrictedDet model.

        :param U:
            The orbital matrix. If None, it will be initialized as a Fermi sea.

        :param dtype:
            The data type for orbital parameters.

        :param out_dtype:
            The data type for computations and outputs. When dtype is real and out_dtype is complex,
            U stores the real and imaginary parts using real numbers.
        """
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

        self.dtype, self.out_dtype, self.holomorphic, real_to_cpl = _check_dtype(
            U, dtype, out_dtype
        )

        if U is None:
            U = _init_spinless_orbs(self.out_dtype)
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
        return _to_comp_mat(self.U, self.out_dtype)

    def normalize(self) -> RestrictedDet:
        """Return a new RestrictedDet instance with orthonormalized orbitals."""
        U_full = self.U_full
        Q, R = jnp.linalg.qr(U_full)
        return RestrictedDet(Q, self.dtype, self.out_dtype)

    def __call__(self, s: jax.Array) -> LogArray:
        """
        Evaluate the wavefunction on given input configurations.
        """
        idx_up, idx_dn = fermion_idx(s, separate_spins=True)
        U_full = self.U_full
        sign_up, logabs_up = jnp.linalg.slogdet(U_full[idx_up, :])
        sign_dn, logabs_dn = jnp.linalg.slogdet(U_full[idx_dn, :])
        return LogArray(sign_up, logabs_up) * LogArray(sign_dn, logabs_dn)


class UnrestrictedDet(eqx.Module):
    r"""
    Unrestricted determinant wavefunction
    :math:`\psi(n) = \mathrm{det}(n_\uparrow \star U_\uparrow)
    \mathrm{det}(n_\downarrow \star U_\downarrow)`.
    Only works for spinful systems with specified number of spin-up and spin-down particles
    """

    U: Tuple[jax.Array, jax.Array]
    dtype: jnp.dtype
    out_dtype: jnp.dtype
    holomorphic: bool

    def __init__(
        self,
        U: Optional[Tuple[jax.Array, jax.Array]] = None,
        dtype: Optional[jnp.dtype] = None,
        out_dtype: Optional[jnp.dtype] = None,
    ):
        sites = get_sites()
        if not sites.is_spinful:
            raise ValueError("UnrestrictedDet only works for spinful systems.")

        if not isinstance(sites.Nparticles, tuple):
            raise ValueError(
                "RestrictedDet requires specified spin-up and spin-down particle numbers."
            )
        Nup, Ndn = sites.Nparticles

        self.dtype, self.out_dtype, self.holomorphic, real_to_cpl = _check_dtype(
            U, dtype, out_dtype
        )

        shape_up = (sites.Nsites, Nup)
        shape_dn = (sites.Nsites, Ndn)
        if U is None:
            U = _init_spinless_orbs(self.out_dtype)
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
        Uup = _to_comp_mat(self.U[0], self.out_dtype)
        Udn = _to_comp_mat(self.U[1], self.out_dtype)
        return Uup, Udn

    def normalize(self) -> UnrestrictedDet:
        """Return a new UnrestrictedDet instance with orthonormalized orbitals."""
        Uup, Udn = self.U_full
        Qup, R = jnp.linalg.qr(Uup)
        Qdn, R = jnp.linalg.qr(Udn)
        return UnrestrictedDet((Qup, Qdn), self.dtype, self.out_dtype)

    def __call__(self, s: jax.Array) -> LogArray:
        idx_up, idx_dn = fermion_idx(s, separate_spins=True)
        Uup, Udn = self.U_full
        sign_up, logabs_up = jnp.linalg.slogdet(Uup[idx_up, :])
        sign_dn, logabs_dn = jnp.linalg.slogdet(Udn[idx_dn, :])
        return LogArray(sign_up, logabs_up) * LogArray(sign_dn, logabs_dn)


class MultiDet(eqx.Module):
    r"""
    The multi-determinant wavefunction
    :math:`\psi(n) = \sum_i c_i \mathrm{det}(n \star U_i)`.
    :math:`U_i` is the orbital matrix for the i-th determinant and :math:`c_i` is its coefficient.
    """

    ndets: int
    U: jax.Array
    coeffs: jax.Array
    dtype: jnp.dtype
    out_dtype: jnp.dtype
    holomorphic: bool

    def __init__(
        self,
        ndets: int = 4,
        U: Optional[jax.Array] = None,
        coeffs: Optional[jax.Array] = None,
        dtype: Optional[jnp.dtype] = None,
        out_dtype: Optional[jnp.dtype] = None,
    ):
        r"""
        Initialize the MultiDet model.

        :param ndets:
            The number of determinants.

        :param U:
            The orbital matrices with shape (ndets, Nfmodes, Nparticles).
            If None, it will be initialized as Fermi seas.

        :param coeffs:
            The coefficients of each determinant. If None, it will be initialized as uniform.

        :param dtype:
            The data type for orbital parameters and coefficients.

        :param out_dtype:
            The data type for computations and outputs. When dtype is real and out_dtype is complex,
            U stores the real and imaginary parts using real numbers.
        """
        sites = get_sites()
        if sites.Ntotal is None:
            raise ValueError("Determinant should have a fixed amount of particles.")

        self.ndets = ndets
        self.dtype, self.out_dtype, self.holomorphic, real_to_cpl = _check_dtype(
            U, dtype, out_dtype
        )

        shape = (ndets, sites.Nfmodes, sites.Ntotal)
        if U is None:
            U = _init_spinless_orbs(self.out_dtype)
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
        return _to_comp_mat(self.U, self.out_dtype)

    def normalize(self) -> MultiDet:
        """Return a new MultiDet instance with orthonormalized orbitals."""
        U_full = self.U_full
        Q, R = jnp.linalg.qr(U_full)
        coeffs = self.coeffs * jnp.prod(jnp.diagonal(R, axis1=1, axis2=2), axis=1)
        return MultiDet(self.ndets, Q, coeffs, self.dtype, self.out_dtype)

    def __call__(self, s: jax.Array) -> LogArray:
        idx = fermion_idx(s)
        sign, logabs = jnp.linalg.slogdet(self.U_full[:, idx, :])
        psi = LogArray(sign, logabs)
        return (psi * self.coeffs).sum()


def _init_paired_orbs(out_dtype: jnp.dtype, f: Optional[jax.Array] = None) -> jax.Array:
    U1 = _init_spinless_orbs(out_dtype)
    if jnp.issubdtype(out_dtype, jnp.complexfloating):
        U2 = U1.conj()
    else:
        U2 = U1

    if f is None:
        f = jnp.ones(U1.shape[1], dtype=out_dtype)
    return jnp.einsum("ia,a,ja->ij", U1, f, U2)


class GeneralPf(RefModel):
    r"""
    A general Pfaffian wavefunction :math:`\psi(n) = \mathrm{pf}(n \star F \star n)`.
    """

    F: jax.Array
    sublattice: Optional[tuple]
    dtype: jnp.dtype
    out_dtype: jnp.dtype
    holomorphic: bool

    def __init__(
        self,
        F: Optional[jax.Array] = None,
        sublattice: Optional[tuple] = None,
        dtype: Optional[jnp.dtype] = None,
        out_dtype: Optional[jnp.dtype] = None,
    ):
        """
        Initialize the GeneralPf model.

        :param F:
            The antisymmetric matrix. If None, it will be initialized as paired Fermi sea orbitals.

        :param sublattice:
            The sublattice structure (not yet implemented).

        :param dtype:
            The data type for orbital parameters.

        :param out_dtype:
            The data type for computations and outputs. When dtype is real and out_dtype is complex,
            F stores the real and imaginary parts using real numbers.
        """
        sites = get_sites()
        self.sublattice = sublattice
        self.dtype, self.out_dtype, self.holomorphic, real_to_cpl = _check_dtype(
            F, dtype, out_dtype
        )

        shape = (sites.Nfmodes, sites.Nfmodes)
        if F is None:
            if sites.is_spinful:
                F = _init_paired_orbs(self.out_dtype)
                zeros = jnp.zeros_like(F)
                F = jnp.block([[zeros, F], [-F.T, zeros]])
            else:
                F = jr.normal(get_subkeys(), shape, self.out_dtype)
                F = (F - F.T) / 2
        else:
            if F.shape != shape:
                raise ValueError(f"Expected F to have shape {shape}, but got {F.shape}")

        if real_to_cpl:
            F = jnp.stack([jnp.real(F), jnp.imag(F)], axis=0)
        self.F = F.astype(self.dtype)

    @property
    def F_full(self) -> jax.Array:
        """
        Returns the full antisymmetric matrix F.
        """
        F = _to_comp_mat(self.F, self.out_dtype)
        return (F - F.T) / 2

    def __call__(self, x: jax.Array) -> LogArray:
        """
        Evaluates the wavefunction at a given configuration.
        """
        idx = fermion_idx(x)
        sign, logabs = lrux.slogpf(self.F_full[idx, :][:, idx])
        return LogArray(sign, logabs)

    def init_internal(self, x: jax.Array) -> MF_Internal:
        """
        Initialize internal values for given input configurations.
        See `~quantax.nn.RefModel` for details.
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
        See `~quantax.nn.RefModel` for details.
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


class GeneralPf_UJUt(eqx.Module):
    r"""
    A general determinant wavefunction :math:`\psi(n) = \mathrm{pf}(n \star F \star n)`,
    where :math:`F = U J U^T`.
    """

    U: jax.Array
    J: jax.Array
    sublattice: Optional[Tuple[int, ...]]
    kpts: Optional[jax.Array]
    dtype: jnp.dtype
    out_dtype: jnp.dtype
    holomorphic: bool

    def __init__(
        self,
        U: Optional[jax.Array] = None,
        J: Optional[jax.Array] = None,
        sublattice: Optional[Tuple[int, ...]] = None,
        kpts: Optional[jax.Array] = None,
        dtype: Optional[jnp.dtype] = None,
        out_dtype: Optional[jnp.dtype] = None,
    ):
        """
        Initialize the GeneralDet model.

        :param U:
            The orbital matrix. If None, it will be initialized as a Fermi sea.

        :param J:
            The antisymmetric matrix. If None, it will be initialized as the standard symplectic form.

        :param sublattice:
            The sublattice shape.

        :param kpts:
            The k-points for each orbital in the sublattice structure.

        :param dtype:
            The data type for orbital parameters.

        :param out_dtype:
            The data type for computations and outputs. When dtype is real and out_dtype is complex,
            U stores the real and imaginary parts using real numbers.
        """
        sites = get_sites()
        Nfmodes = sites.Nfmodes
        if Nfmodes % 2 != 0:
            raise ValueError("Pfaffian requires even number of modes.")

        self.dtype, self.out_dtype, self.holomorphic, real_to_cpl = _check_dtype(
            U, dtype, out_dtype
        )

        if U is None:
            if kpts is not None:
                raise NotImplementedError(
                    "Unspecified orbitals with k-points is not implemented."
                )

            if sublattice is None:
                U = _init_spinless_orbs(self.out_dtype)
            else:
                U, kpts = _init_spinless_orbs(self.out_dtype, sublattice, True)

            if sites.is_spinful:
                Uup = U
                if jnp.issubdtype(self.out_dtype, jnp.complexfloating):
                    Udn = U.conj()
                else:
                    Udn = U
                zeros = jnp.zeros_like(U)
                U = jnp.block([[Uup, zeros], [zeros, Udn]])
                if kpts is not None:
                    kpts = jnp.concatenate([kpts, -kpts], axis=0)
            if real_to_cpl:
                U = jnp.stack([jnp.real(U), jnp.imag(U)], axis=0)
        else:
            shape = (Nfmodes, Nfmodes)
            if real_to_cpl:
                shape = (2,) + shape
                if jnp.issubdtype(U.dtype, jnp.complexfloating):
                    U = jnp.stack([jnp.real(U), jnp.imag(U)], axis=0)
            if U.shape != shape:
                raise ValueError(f"Expected U to have shape {shape}, but got {U.shape}")

        if J is None:
            J = lrux.skew_eye(Nfmodes // 2, self.dtype)
        else:
            if J.shape != (Nfmodes, Nfmodes):
                raise ValueError(
                    f"Expected J to have shape {(Nfmodes, Nfmodes)}, but got {J.shape}"
                )

        self.sublattice = sublattice
        self.kpts = kpts
        self.U = U.astype(self.dtype)
        self.J = J.astype(self.dtype)

    @property
    def U_full(self) -> jax.Array:
        """
        Returns the full orbital matrix U.
        """
        U = _to_comp_mat(self.U, self.out_dtype)
        if self.sublattice is not None:
            lattice = get_lattice()
            if lattice.is_spinful:
                Uup, Udn = jnp.split(U, 2, axis=0)
                Uup = lattice.restore_full_orbitals(Uup, self.sublattice, self.kpts)
                Udn = lattice.restore_full_orbitals(Udn, self.sublattice, self.kpts)
                U = jnp.concatenate([Uup, Udn], axis=0)
            else:
                U = lattice.restore_full_orbitals(U, self.sublattice, self.kpts)
        return U
    
    @property
    def F_full(self) -> jax.Array:
        """
        Returns the full antisymmetric matrix F = U J U^T.
        """
        U = self.U_full
        J = (self.J - self.J.T) / 2
        return U @ J @ U.T

    def __call__(self, s: jax.Array) -> PsiArray:
        """
        Evaluate the wavefunction on given input configurations.
        """
        idx = fermion_idx(s)
        F = self.F_full
        sign, logabs = lrux.slogpf(F[idx, :][:, idx])
        return LogArray(sign, logabs)


class SingletPair(RefModel):
    r"""
    The singlet paired wavefunction
    :math:`\psi(n) = \mathrm{det}(n_\uparrow \star F \star n_\downarrow)`.
    Only works for spinful systems with equal number of spin-up and spin-down particles.
    """

    F: jax.Array
    sublattice: Optional[tuple]
    dtype: jnp.dtype
    out_dtype: jnp.dtype
    holomorphic: bool

    def __init__(
        self,
        F: Optional[jax.Array] = None,
        sublattice: Optional[tuple] = None,
        dtype: Optional[jnp.dtype] = None,
        out_dtype: Optional[jnp.dtype] = None,
    ):
        """
        Initialize the SingletPair model.

        :param F:
            The pairing matrix. If None, it will be initialized as paired Fermi sea orbitals

        :param sublattice:
            The sublattice structure (not yet implemented).

        :param dtype:
            The data type for orbital parameters.

        :param out_dtype:
            The data type for computations and outputs. When dtype is real and out_dtype is complex,
            F stores the real and imaginary parts using real numbers.
        """
        sites = get_sites()
        if not sites.is_spinful:
            raise ValueError("SingletPair only works for spinful systems.")

        self.sublattice = sublattice
        self.dtype, self.out_dtype, self.holomorphic, real_to_cpl = _check_dtype(
            F, dtype, out_dtype
        )

        shape = (sites.Nsites, sites.Nsites)
        if F is None:
            F = _init_paired_orbs(self.out_dtype)
        else:
            if F.shape != shape:
                raise ValueError(f"Expected F to have shape {shape}, but got {F.shape}")

        if real_to_cpl:
            F = jnp.stack([jnp.real(F), jnp.imag(F)], axis=0)
        self.F = F.astype(self.dtype)

    @property
    def F_full(self) -> jax.Array:
        """
        Return the full pairing matrix.
        """
        return _to_comp_mat(self.F, self.out_dtype)

    def __call__(self, s: jax.Array) -> LogArray:
        idx_up, idx_dn = fermion_idx(s, separate_spins=True)
        if idx_up.size != idx_dn.size:
            sign = jnp.array(0.0, dtype=self.out_dtype)
            logabs = jnp.array(-jnp.inf, dtype=self.out_dtype)
        else:
            F_full = self.F_full[idx_up, :][:, idx_dn]
            sign, logabs = jnp.linalg.slogdet(F_full)
            n = F_full.shape[0]
            sign *= (-1) ** (n * (n - 1) // 2)
        return LogArray(sign, logabs)

    def init_internal(self, s: jax.Array) -> MF_Internal:
        """
        Initialize internal values for given input configurations.
        See `~quantax.nn.RefModel` for details.
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
        See `~quantax.nn.RefModel` for details.
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
    r"""
    Multi-particle fermionic wavefunction
    :math:`\psi(n) = \sum_i \mathrm{pf}(n \star F_i \star n)`.
    """

    npfs: int
    F: jax.Array
    sublattice: Optional[tuple]
    dtype: jnp.dtype
    out_dtype: jnp.dtype
    holomorphic: bool

    def __init__(
        self,
        npfs: int = 4,
        F: Optional[jax.Array] = None,
        sublattice: Optional[tuple] = None,
        dtype: Optional[jnp.dtype] = None,
        out_dtype: Optional[jnp.dtype] = None,
    ):
        """
        Initialize the MultiPf model.

        :param npfs:
            The number of Pfaffians.

        :param F:
            The antisymmetric matrices with shape (npfs, Nfmodes, Nfmodes).
            If None, it will be initialized as paired Fermi sea orbitals.

        :param sublattice:
            The sublattice structure (not yet implemented).

        :param dtype:
            The data type for orbital parameters.

        :param out_dtype:
            The data type for computations and outputs. When dtype is real and out_dtype is complex,
            F stores the real and imaginary parts using real numbers.
        """
        sites = get_sites()
        self.npfs = npfs
        self.sublattice = sublattice
        self.dtype, self.out_dtype, self.holomorphic, real_to_cpl = _check_dtype(
            F, dtype, out_dtype
        )

        shape = (npfs, sites.Nfmodes, sites.Nfmodes)
        if F is None:
            if sites.is_spinful:
                F = _init_paired_orbs(self.out_dtype)
                zeros = jnp.zeros_like(F)
                F = jnp.block([[zeros, F], [-F.mT, zeros]])
                F = jnp.tile(F, (npfs, 1, 1))
            else:
                F = jr.normal(get_subkeys(), shape, self.out_dtype)
                F = (F - F.mT) / 2
        else:
            if F.shape != shape:
                raise ValueError(f"Expected F to have shape {shape}, but got {F.shape}")

        if real_to_cpl:
            F = jnp.stack([jnp.real(F), jnp.imag(F)], axis=0)
        self.F = F.astype(self.dtype)

    @property
    def F_full(self) -> jax.Array:
        F = _to_comp_mat(self.F, self.out_dtype)
        return (F - F.mT) / 2

    def __call__(self, x: jax.Array) -> LogArray:
        idx = fermion_idx(x)
        sign, logabs = lrux.slogpf(self.F_full[:, idx, :][:, :, idx])
        return LogArray(sign, logabs).sum()


class PartialPair(eqx.Module):
    r"""
    A partially paired wavefunction.
    """

    Nunpaired: int
    U: jax.Array
    J: jax.Array
    dtype: jnp.dtype
    out_dtype: jnp.dtype
    holomorphic: bool

    def __init__(
        self,
        Nunpaired: int,
        U: Optional[jax.Array] = None,
        J: Optional[jax.Array] = None,
        dtype: Optional[jnp.dtype] = None,
        out_dtype: Optional[jnp.dtype] = None,
    ):
        """
        Initialize the GeneralPf model.

        :param Nunpaired:
            The number of unpaired orbitals.

        :param U:
            The orbital matrix. If None, it will be initialized as a Fermi sea.

        :param J:
            The antisymmetric pairing matrix. If None, it will be initialized as
            equal pairing of Fermi sea orbitals.

        :param dtype:
            The data type for orbital parameters.

        :param out_dtype:
            The data type for computations and outputs. When dtype is real and out_dtype is complex,
            F stores the real and imaginary parts using real numbers.
        """
        sites = get_sites()
        Nfmodes = sites.Nfmodes
        self.dtype, self.out_dtype, self.holomorphic, real_to_cpl = _check_dtype(
            J, dtype, out_dtype
        )

        if Nfmodes % 2 == 1 or Nunpaired % 2 == 1:
            raise NotImplementedError

        self.Nunpaired = Nunpaired

        shapeU = (Nfmodes, Nfmodes)
        if U is None:
            U = _init_spinless_orbs(self.out_dtype)
            if sites.is_spinful:
                Uup = U
                if jnp.issubdtype(out_dtype, jnp.complexfloating):
                    Udn = Uup.conj()
                else:
                    Udn = Uup
                Uup = jnp.concatenate([Uup, jnp.zeros_like(Uup)], axis=0)
                Udn = jnp.concatenate([jnp.zeros_like(Udn), Udn], axis=0)
                U = jnp.stack([Uup, Udn], axis=2).reshape(Nfmodes, Nfmodes)
        else:
            if U.shape != shapeU:
                raise ValueError(
                    f"Expected U to have shape {shapeU}, but got {U.shape}"
                )

        Nmodes = Nfmodes - Nunpaired
        shapeJ = (Nmodes, Nmodes)
        if J is None:
            J = jnp.array([1, 0] * (Nmodes // 2), dtype=self.dtype)[:-1]
            J = jnp.diag(J, k=1)
            J = J - J.T
        else:
            if J.shape != shapeJ:
                raise ValueError(
                    f"Expected F to have shape {shapeJ}, but got {J.shape}"
                )

        if real_to_cpl:
            U = jnp.stack([jnp.real(U), jnp.imag(U)], axis=0)
        self.U = U.astype(self.dtype)
        self.J = J.astype(self.dtype)

    @property
    def U_full(self) -> jax.Array:
        """
        Returns the full orbital matrix U.
        """
        return _to_comp_mat(self.U, self.out_dtype)

    @property
    def F_full(self) -> jax.Array:
        """
        Returns the full antisymmetric matrix F.
        """
        U = self.U_full[:, self.Nunpaired :]
        J = (self.J - self.J.T) / 2
        return U @ J @ U.T

    def __call__(self, x: jax.Array) -> LogArray:
        """
        Evaluates the wavefunction at a given configuration.
        """
        idx = fermion_idx(x)
        U_full = self.U_full
        U = U_full[idx, : self.Nunpaired]
        Up = U_full[idx, self.Nunpaired :]
        J = (self.J - self.J.T) / 2
        F = Up @ J @ Up.T
        O = jnp.zeros((U.shape[1], U.shape[1]), dtype=U.dtype)
        F_full = jnp.block([[F, U], [-U.T, O]])
        sign, logabs = lrux.slogpf(F_full)
        return LogArray(sign, logabs)
