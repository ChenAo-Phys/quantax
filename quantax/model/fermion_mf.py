from __future__ import annotations
from typing import Optional, Callable
from jaxtyping import Key
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from ..symmetry import Symmetry
from ..nn import Sequential, RawInputLayer, NoGradLayer
from ..utils import det, pfaffian
from ..global_defs import get_sites, get_subkeys, is_default_cpl


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


class Determinant(eqx.Module):
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
    holomorphic: bool

    def __init__(self, dtype: jnp.dtype = jnp.float32):
        if get_sites().is_fermion:
            raise RuntimeError("`PairProductSpin` only works in spin systems.")

        N = get_sites().nsites
        if N % 2 > 0:
            raise RuntimeError("`PairProductSpin` only supports even sites.")

        shape = (N * N - N,)
        is_dtype_cpl = jnp.issubdtype(dtype, jnp.complexfloating)
        if is_default_cpl() and not is_dtype_cpl:
            shape = (2,) + shape
        scale = np.sqrt(2 * np.e / N, dtype=dtype)
        self.F = jr.normal(get_subkeys(), shape, dtype) * scale
        self.holomorphic = is_default_cpl() and is_dtype_cpl

    def __call__(self, x: jax.Array) -> jax.Array:
        F = self.F if self.F.ndim == 1 else jax.lax.complex(self.F[0], self.F[1])
        N = get_sites().nsites
        F_full = jnp.zeros((N, N), F.dtype)
        F_full = F_full.at[~np.eye(N, dtype=np.bool_)].set(F)
        idx = _get_fermion_idx(x, N)
        F_full = F_full[idx[: N // 2], :][:, idx[N // 2 :] - N]
        return det(F_full)

    def rescale(self, maximum: jax.Array) -> PairProductSpin:
        N = get_sites().nsites
        F = self.F / maximum.astype(self.F.dtype) ** (2 / N)
        return eqx.tree_at(lambda tree: tree.F, self, F)


class NeuralFermionLayer(RawInputLayer):
    fermion_mf: eqx.Module
    symm: Symmetry = eqx.field(static=True)
    output_fn: Callable = eqx.field(static=True)

    def __init__(self, fermion_mf: eqx.Module, symm: Symmetry, output_fn: Callable):
        self.fermion_mf = fermion_mf
        self.symm = symm
        self.output_fn = output_fn

    def __call__(self, x: jax.Array, s: jax.Array) -> jax.Array:
        s_symm = self.symm.get_symm_spins(s)
        x_mf = jax.vmap(self.fermion_mf)(s_symm)
        return self.output_fn(x, x_mf, s)

    def rescale(self, maximum: jax.Array) -> eqx.Module:
        if hasattr(self.fermion_mf, "rescale"):
            fermion_mf = self.fermion_mf.rescale(maximum)
            return eqx.tree_at(lambda tree: tree.fermion_mf, self, fermion_mf)
        else:
            return self


class NeuralJastrow(Sequential):
    layers: tuple
    holomorphic: bool = eqx.field(static=True)

    def __init__(self, net: eqx.Module, fermion_mf: eqx.Module, symm: Symmetry):
        def output_fn(x_net: jax.Array, x_mf: jax.Array, s: jax.Array) -> jax.Array:
            x_net = x_net.reshape(-1, symm.nsymm)
            x_net = jnp.mean(x_net, axis=0)
            return symm.symmetrize(x_net * x_mf, s)

        fermion_layer = NeuralFermionLayer(fermion_mf, symm, output_fn)
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


class HiddenDet(eqx.Module):
    net: eqx.Module
    U: jax.Array
    Nvisible: int
    Nhidden: int

    def __init__(
        self,
        net: eqx.Module,
        Nvisible: Optional[int] = None,
        Nhidden: Optional[int] = None,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.net = net
        N = get_sites().nsites
        self.Nvisible = N if Nvisible is None else Nvisible
        self.Nhidden = self.Nvisible if Nhidden is None else Nhidden
        Ntotal = self.Nvisible + self.Nhidden
        # scale = np.sqrt((np.e / Ntotal) ** (Ntotal / self.Nvisible), dtype=dtype)
        is_dtype_cpl = jnp.issubdtype(dtype, jnp.complexfloating)
        shape = (2 * N, Ntotal)
        if is_default_cpl() and not is_dtype_cpl:
            shape = (2,) + shape
        self.U = jr.normal(get_subkeys(), shape, dtype)  # * scale

    def get_Uvisible(self, x: jax.Array):
        U = self.U if self.U.ndim == 2 else jax.lax.complex(self.U[0], self.U[1])
        idx = _get_fermion_idx(x, self.Nvisible)
        return U[idx, :]

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        Ntotal = self.Nvisible + self.Nhidden
        shape = (self.Nhidden, Ntotal)
        Uhidden = self.net(x, key=key).reshape(shape)
        Uvisible = self.get_Uvisible(x)
        U = jnp.concatenate([Uvisible, Uhidden], axis=0)
        U *= np.sqrt(np.e / Ntotal, dtype=U.dtype)
        return det(U)

    # def rescale(self, maximum: jax.Array) -> Determinant:
    #     U = self.U / maximum.astype(self.U.dtype) ** (1 / self.Nvisible)
    #     return eqx.tree_at(lambda tree: tree.U, self, U)


class HiddenPf(eqx.Module):
    net1: eqx.Module
    net2: eqx.Module
    F: jax.Array
    Nvisible: int
    Nhidden: int

    def __init__(
        self,
        net1: eqx.Module,
        net2: eqx.Module,
        Nvisible: Optional[int] = None,
        Nhidden: Optional[int] = None,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.net1 = net1
        self.net2 = net2
        N = get_sites().nsites
        self.Nvisible = N if Nvisible is None else Nvisible
        self.Nhidden = self.Nvisible if Nhidden is None else Nhidden
        Ntotal = self.Nvisible + self.Nhidden
        # scale = np.sqrt((np.e / Ntotal) ** (Ntotal / self.Nvisible), dtype=dtype)
        is_dtype_cpl = jnp.issubdtype(dtype, jnp.complexfloating)
        shape = (N * (2 * N - 1),)
        if is_default_cpl() and not is_dtype_cpl:
            shape = (2,) + shape
        self.F = jr.normal(get_subkeys(), shape, dtype)  # * scale

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        Ntotal = self.Nvisible + self.Nhidden
        fermion_idx = _get_fermion_idx(x, self.Nvisible)

        F0 = self.F if self.F.ndim == 1 else jax.lax.complex(self.F[0], self.F[1])
        N = get_sites().nsites
        F0_full = jnp.zeros((2 * N, 2 * N), dtype=F0.dtype)
        F0_full = F0_full.at[jnp.tril_indices(2 * N, -1)].set(F0)
        F = F0_full[fermion_idx, :][:, fermion_idx]
        F = jnp.pad(F, ((0, self.Nhidden), (0, self.Nhidden)))

        F1 = self.net1(x, key=key).reshape(self.Nhidden, self.Nvisible)
        F = F.at[self.Nvisible :, : self.Nvisible].set(F1[:, fermion_idx])

        F2 = self.net2(x, key=key)
        idx = jnp.tril_indices(self.Nhidden, -1)
        idx = tuple(i + self.Nvisible for i in idx)
        F = F.at[idx].set(F2)

        F *= np.sqrt(np.e / Ntotal, dtype=F.dtype)
        F = F - F.T
        return pfaffian(F)
