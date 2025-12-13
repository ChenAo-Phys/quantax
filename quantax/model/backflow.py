from typing import Optional, Tuple, Union
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import lrux
from .fermion_mf import GeneralDet, MF_Internal, _init_spinless_orbs
from ..global_defs import get_sites, get_subkeys
from ..nn import (
    RefModel,
    lecun_normal,
    fermion_idx,
    changed_inds,
    permute_sign,
    fermion_inverse_sign,
)
from ..utils import LogArray


class DetBackflow(RefModel):
    net: eqx.Module
    U0: jax.Array
    W: jax.Array
    dtype: jnp.dtype

    r"""
    Determinant backflow model.
    :math:`\psi(n) = \mathrm{det}(n \star (U_0 + U_1(n)))`,
    where :math:`\star` denotes the operation slicing the rows of the matrix.
    """

    def __init__(
        self,
        net: eqx.Module,
        d: int,
        U0: Optional[jax.Array] = None,
        dtype: jnp.dtype = jnp.float64,
    ):
        r"""
        Initialize the determinant backflow model.

        :param net:
            The backflow network that outputs the correction to the mean-field orbitals.

        :param d:
            Channel (hidden) dimension of the network output.

        :param U0:
            The mean-field orbitals. If None, it's initialized close to
            non-interacting fermions.

        :param dtype:
            Data type of the parameters.
        """
        self.net = net
        self.dtype = dtype
        sites = get_sites()

        if U0 is None:
            det = GeneralDet(dtype=dtype)
            U0 = det.U_full
        else:
            if U0.shape != (sites.Nfmodes, sites.Ntotal):
                raise ValueError(
                    f"U0 must have shape {(sites.Nfmodes, sites.Ntotal)}, got {U0.shape}"
                )
        U0 /= jnp.std(U0)
        self.U0 = U0.astype(dtype)

        if sites.is_spinful:
            d //= 2
        self.W = lecun_normal(get_subkeys(), (sites.Ntotal, d), dtype=dtype) / 10

    def __call__(self, s: jax.Array) -> jax.Array:
        x = self.net(s)

        idx = fermion_idx(s)
        x = x.reshape(-1, get_sites().Nfmodes).astype(self.dtype)
        x = x.T[idx]
        U = self.U0[idx, :] + x @ self.W.T
        sign, logabs = jnp.linalg.slogdet(U)
        psi = LogArray(sign, logabs)
        return psi * fermion_inverse_sign(s)

    @property
    def use_ref(self) -> bool:
        """
        Whether to use reference low-rank updates. It's not used if the rank of the
        backflow correction is larger than the total number of particles.
        """
        Ntotal = get_sites().Ntotal
        rank = self.W.shape[1]
        return rank < Ntotal

    def init_internal(self, s) -> Optional[MF_Internal]:
        """
        Initialize internal values for given input configurations.
        See `~quantax.nn.RefModel` for details.
        """
        if not self.use_ref:
            return None

        idx = fermion_idx(s)
        orbs = self.U0[idx, :]
        inv = jnp.linalg.inv(orbs)
        sign, logabs = jnp.linalg.slogdet(orbs)
        psi = LogArray(sign, logabs) * fermion_inverse_sign(s)
        return MF_Internal(idx, inv, psi)

    def ref_forward(
        self,
        s: jax.Array,
        s_old: jax.Array,
        nflips: int,
        internal: Optional[MF_Internal],
        return_update: bool = False,
    ) -> Union[LogArray, Tuple[LogArray, MF_Internal]]:
        """
        Accelerated forward pass through local updates and internal quantities.
        See `~quantax.nn.RefModel` for details.
        """
        if not self.use_ref:
            psi = self(s)
            if return_update:
                return psi, internal
            else:
                return psi

        nhops = nflips // 2
        idx_annihilate, idx_create = changed_inds(s, s_old, nhops)
        idx = internal.idx
        sign = permute_sign(idx, idx_annihilate, idx_create)
        is_updated = jnp.isin(idx, idx_annihilate)
        row_update_idx = jnp.flatnonzero(is_updated, size=nhops, fill_value=idx.size)
        new_idx = idx.at[row_update_idx].set(idx_create)
        row_update = self.U0[idx_create] - self.U0[idx_annihilate]

        x = self.net(s)
        x = x.reshape(-1, get_sites().Nfmodes).astype(self.dtype)
        x = x.T[new_idx]

        if return_update:
            r1, inv = lrux.det_lru(
                internal.inv, row_update.T, row_update_idx, return_update=True
            )
            psi = internal.psi * (r1 * sign)
            internal = MF_Internal(new_idx, inv, psi)
            r2 = lrux.det_lru(inv, self.W, x)
            return psi * r2, internal
        else:
            u = jnp.concatenate((row_update.T, self.W), axis=1)
            v = (x, row_update_idx)
            ratio = lrux.det_lru(internal.inv, u, v)
            psi = internal.psi * (ratio * sign)
            return psi


class PfBackflow(RefModel):
    net: eqx.Module
    U0: jax.Array
    J0: jax.Array
    W: jax.Array
    dtype: jnp.dtype

    r"""
    Pfaffian backflow model.
    :math:`\psi(n) = \mathrm{pf}(n \star (U_0 + U_1(n)) J_0 (U_0 + U_1(n))^T)`,
    where :math:`\star` denotes the operation slicing the rows and columns of the matrix.
    """

    def __init__(
        self,
        net: eqx.Module,
        d: int,
        U0: Optional[jax.Array] = None,
        J0: Optional[jax.Array] = None,
        dtype: jnp.dtype = jnp.float64,
    ):
        r"""
        Initialize the Pfaffian backflow model.

        :param net:
            The backflow network that outputs the correction to the mean-field orbitals.

        :param d:
            Channel (hidden) dimension of the network output.

        :param U0:
            The mean-field orbitals. If None, it's initialized close to
            non-interacting fermions.

        :param J0:
            The mean-field pairing matrix.

        :param dtype:
            Data type of the parameters.
        """
        self.net = net
        self.dtype = dtype
        sites = get_sites()
        M = sites.Nfmodes

        if U0 is None:
            U0 = _init_spinless_orbs(dtype)
            if sites.is_spinful:
                U1 = U0.conj() if jnp.issubdtype(dtype, jnp.complexfloating) else U0
                O = jnp.zeros_like(U0)
                U0 = jnp.block([[U0, O], [O, U1]])
        elif U0.shape != (M, M):
            raise ValueError(f"U0 must have shape {(M, M)}, got {U0.shape}")
        U0 /= jnp.std(U0)
        self.U0 = U0.astype(dtype)

        if J0 is None:
            if sites.is_spinful:
                J0 = lrux.skew_eye(M // 2, dtype)
            else:
                J0 = jr.normal(get_subkeys(), (M, M), dtype=dtype)
                J0 = (J0 - J0.T) / 2
        elif J0.shape != (M, M):
            raise ValueError(f"J0 must have shape {(M, M)}, got {J0.shape}")
        self.J0 = J0[jnp.triu_indices(M, k=1)].astype(dtype)

        if sites.is_spinful:
            d //= 2
        self.W = lecun_normal(get_subkeys(), (M, d), dtype=dtype) / 10

    @property
    def J0_full(self) -> jax.Array:
        M = self.U0.shape[0]
        J_full = jnp.zeros((M, M), dtype=self.dtype)
        triu_indices = jnp.triu_indices(M, k=1)
        J_full = J_full.at[triu_indices].set(self.J0)
        J_full = J_full - J_full.T
        return J_full

    def __call__(self, s: jax.Array) -> jax.Array:
        x = self.net(s)

        idx = fermion_idx(s)
        x = x.reshape(-1, get_sites().Nfmodes).astype(self.dtype)
        x = x.T[idx]
        U = self.U0[idx, :] + x @ self.W.T
        F = U @ self.J0_full @ U.T
        sign, logabs = lrux.slogpf(F)
        psi = LogArray(sign, logabs)
        return psi * fermion_inverse_sign(s)

    @property
    def use_ref(self) -> bool:
        """
        Whether to use reference low-rank updates. It's not used if the rank of the
        backflow correction is larger than the total number of particles.
        """
        Ntotal = get_sites().Ntotal
        rank = self.W.shape[1] * 2
        return rank < Ntotal

    def init_internal(self, s) -> Optional[MF_Internal]:
        """
        Initialize internal values for given input configurations.
        See `~quantax.nn.RefModel` for details.
        """
        if not self.use_ref:
            return None

        idx = fermion_idx(s)
        U = self.U0[idx, :]
        F = U @ self.J0_full @ U.T
        inv = jnp.linalg.inv(F)
        sign, logabs = lrux.slogpf(F)
        psi = LogArray(sign, logabs) * fermion_inverse_sign(s)
        return MF_Internal(idx, inv, psi)

    def ref_forward(
        self,
        s: jax.Array,
        s_old: jax.Array,
        nflips: int,
        internal: Optional[MF_Internal],
        return_update: bool = False,
    ) -> Union[LogArray, Tuple[LogArray, MF_Internal]]:
        """
        Accelerated forward pass through local updates and internal quantities.
        See `~quantax.nn.RefModel` for details.
        """
        if not self.use_ref:
            psi = self(s)
            if return_update:
                return psi, internal
            else:
                return psi

        nhops = nflips // 2
        idx_annihilate, idx_create = changed_inds(s, s_old, nhops)
        idx = internal.idx
        sign = permute_sign(idx, idx_annihilate, idx_create)
        is_updated = jnp.isin(idx, idx_annihilate)
        row_update_idx = jnp.flatnonzero(is_updated, size=nhops, fill_value=idx.size)
        new_idx = idx.at[row_update_idx].set(idx_create)

        U0 = self.U0
        J0 = self.J0_full
        U_diff = U0[idx_create, :] - U0[idx_annihilate, :]
        U_mean = (U0[new_idx, :] + U0[idx, :]) / 2
        x = jnp.einsum("im,mn,jn->ij", U_diff, J0, U_mean).T

        U1 = self.net(s)
        U1 = U1.reshape(-1, get_sites().Nfmodes).astype(self.dtype)
        U1 = U1.T[new_idx, :]
        W = self.W
        U2 = U0[new_idx, :] + jnp.einsum("nd,id->in", W, U1) / 2
        V = jnp.einsum("im,mn,nd->id", U2, J0, W)

        if return_update:
            r1, inv = lrux.pf_lru(internal.inv, (x, row_update_idx), return_update=True)
            psi = internal.psi * (r1 * sign)
            internal = MF_Internal(new_idx, inv, psi)
            r2 = lrux.pf_lru(inv, jnp.concatenate([U1, V], axis=1))
            return psi * r2, internal
        else:
            e = jnp.zeros_like(x)
            e = e.at[row_update_idx, jnp.arange(nhops)].set(1)
            u = jnp.concatenate([x, U1, e, V], axis=1)
            ratio = lrux.pf_lru(internal.inv, u)
            psi = internal.psi * (ratio * sign)
            return psi
