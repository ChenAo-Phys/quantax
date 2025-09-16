from __future__ import annotations
from typing import Optional, Tuple, Union, BinaryIO
from pathlib import Path
from functools import partial
from warnings import warn
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.flatten_util as jfu
import equinox as eqx
import lrux
from .state import State
from .variational import Variational
from ..sites import Grid
from ..nn import fermion_idx
from ..model import GeneralDet, RestrictedDet, UnrestrictedDet, MultiDet
from ..global_defs import get_subkeys, get_sites, get_default_dtype, get_real_dtype


_Array = Union[np.ndarray, jax.Array]


@eqx.filter_jit
def _reformat_fermion_op(jax_op_list: list) -> list:
    reformatted_list = []

    for opstr, J_array, index_array in jax_op_list:
        new_op = ""
        new_idx = []
        for op, idx in zip(opstr, index_array.T):
            if op == "I":
                continue
            elif op == "n":
                new_op += "+-"
                new_idx.append(idx)
                new_idx.append(idx)
            else:
                new_op += op
                new_idx.append(idx)

        new_idx = jnp.stack(new_idx, axis=1)
        reformatted_list.append([new_op, J_array, new_idx])

    return reformatted_list


@eqx.filter_jit
def _reformat_jax_op_list(jax_op_list: list) -> list:
    if get_sites().is_fermion:
        return _reformat_fermion_op(jax_op_list)
    else:
        raise NotImplementedError


class MeanFieldFermionState(Variational):
    """Base class for fermionic mean-field states"""

    def __init__(
        self,
        model: Optional[eqx.Module] = None,
        param_file: Optional[Union[str, Path, BinaryIO]] = None,
        max_parallel: Union[None, int, Tuple[int, int]] = None,
        use_refmodel: bool = True,
    ):
        model = self._check_model(model)
        super().__init__(
            model, param_file, max_parallel=max_parallel, use_refmodel=use_refmodel
        )
        self._energy = None
        self._spectrum = None

        exp_real = lambda model, jax_op_list: self._expectation_from_model(
            model, jax_op_list
        ).real
        self._val_grad_model = eqx.filter_jit(eqx.filter_value_and_grad(exp_real))

        exp_real = lambda rho, jax_op_list: self._expectation_from_rho(
            rho, jax_op_list
        ).real
        self._val_grad_rho = eqx.filter_jit(eqx.filter_value_and_grad(exp_real))

    def _check_model(self, model: Optional[eqx.Module]) -> eqx.Module:
        """Check the input model and initialize if None"""
        return model

    @property
    def is_paired(self) -> bool:
        """Whether the state is a paired state (pfaffian) or not (determinant)"""
        return False

    @property
    def energy(self) -> Optional[float]:
        return self._energy

    @eqx.filter_jit
    def rho_from_model(self, model: eqx.Module) -> jax.Array:
        r"""
        Get the one-body density matrix $\left< c_i^\dagger c_j \right>$ from the
        mean-field parameters.

        Args:
            full_params: Full mean-field orbitals, for instance,
            U for determinant state and F for pfaffian state

        Returns:
            One-body density matrix
        """
        return NotImplemented

    @eqx.filter_jit
    def kappa_from_model(self, model: eqx.Module) -> jax.Array:
        r"""
        Get the pairing matrix $\left< c_i c_j \right>$ from the
        mean-field parameters.

        Args:
            full_params: Full mean-field orbitals, for instance,
            U for determinant state and F for pfaffian state

        Returns:
            Pairing matrix
        """
        if self.is_paired:
            return NotImplemented
        else:
            M = get_sites().Nfmodes
            return jnp.zeros((M, M), self.dtype)

    def expectation(self, operator) -> jax.Array:
        op_list = _reformat_fermion_op(operator.jax_op_list)
        return self._expectation_from_model(self.model, op_list)

    @eqx.filter_jit
    def _expectation_from_model(
        self, model: eqx.Module, jax_op_list: list
    ) -> jax.Array:
        rho = self.rho_from_model(model)

        if self.is_paired:
            kappa = self.kappa_from_model(model)
            return self._expectation_from_rho((rho, kappa), jax_op_list)
        else:
            return self._expectation_from_rho(rho, jax_op_list)

    @eqx.filter_jit
    def _expectation_from_rho(
        self, rho: Union[jax.Array, Tuple[jax.Array, jax.Array]], jax_op_list: list
    ) -> jax.Array:
        if self.is_paired:
            rho, kappa = rho
            kappa_ = -kappa.conj()
        I = jnp.eye(get_sites().Nfmodes, dtype=rho.dtype)
        rho_ = I - rho.T

        def get_contract(opstr, index_array):
            if len(opstr) % 2 == 1:
                return 0.0

            if len(opstr) == 2:
                idx0 = index_array[:, 0]
                idx1 = index_array[:, 1]
                if opstr == "+-":
                    return rho[idx0, idx1]
                elif opstr == "-+":
                    return rho_[idx0, idx1]
                elif opstr == "++":
                    return kappa[idx0, idx1] if self.is_paired else 0.0
                elif opstr == "--":
                    return kappa_[idx0, idx1] if self.is_paired else 0.0
                else:
                    raise NotImplementedError

            output = 0.0
            for i, c in enumerate(opstr[1:]):
                i += 1
                current_op = opstr[0] + c
                current_idx = index_array[:, [0, i]]
                current_contract = get_contract(current_op, current_idx)

                remain_op = opstr[1:i] + opstr[i + 1 :]
                idx = list(range(1, i)) + list(range(i + 1, len(opstr)))
                remain_idx = index_array[:, idx]
                remain_contract = get_contract(remain_op, remain_idx)
                sign = 1 if i % 2 == 1 else -1
                output += sign * current_contract * remain_contract

            return output

        output = jnp.array(0.0, rho.dtype)
        for opstr, J_array, index_array in jax_op_list:
            output += jnp.sum(J_array * get_contract(opstr, index_array))
        return output

    def get_step(self, hamiltonian) -> jax.Array:
        """Get the gradient of the energy with respect to the mean-field parameters."""
        op_list = _reformat_fermion_op(hamiltonian.jax_op_list)
        self._energy, g = self._val_grad_model(self.model, op_list)
        g, _ = jfu.ravel_pytree(g)
        return g.conj()

    def HF_update(self, hamiltonian) -> None:
        """Update the mean-field parameters with one step of Hartree-Fock iteration"""
        raise NotImplementedError

    def update(self, step: jax.Array) -> None:
        """Update the mean-field parameters with a given step"""
        super().update(step)
        if hasattr(self.model, "normalize"):
            self._model = self.model.normalize()


class GeneralDetState(MeanFieldFermionState):
    """Determinant mean-field state"""

    def _check_model(self, model):
        if model is None:
            model = GeneralDet()
        elif not isinstance(model, GeneralDet):
            raise ValueError("Input model must be an instance of GeneralDet.")
        else:
            U = model.U_full
            I = jnp.eye(U.shape[1], dtype=U.dtype)
            if not jnp.allclose(U.conj().T @ U, I):
                warn("Input orbitals aren't orthonormal. They will be orthonormalized.")
                model = model.normalize()
        return model

    @eqx.filter_jit
    def rho_from_model(self, model: GeneralDet) -> jax.Array:
        r"""
        Get the one-body density matrix $\left< c_i^\dagger c_j \right>$ from the
        mean-field orbitals.

        Args:
            full_params: Full mean-field orbitals

        Returns:
            One-body density matrix
        """
        U = model.U
        inv = jnp.linalg.inv(U.conj().T @ U)
        return (U @ inv @ U.conj().T).T

    def HF_update(self, hamiltonian) -> None:
        """Update the mean-field parameters with one step of Hartree-Fock iteration"""
        op_list = _reformat_fermion_op(hamiltonian.jax_op_list)
        rho = self.rho_from_model(self.model)
        self._energy, H = self._val_grad_rho(rho, op_list)
        self._spectrum, U = jnp.linalg.eigh(H)
        Ntotal = get_sites().Ntotal
        U = U[:, :Ntotal]
        self._model = GeneralDet(U, self.model.dtype, self.model.preferred_element_type)


class RestrictedDetState(MeanFieldFermionState):
    """Restricted determinant mean-field state"""

    def _check_model(self, model):
        if model is None:
            model = RestrictedDet()
        elif not isinstance(model, RestrictedDet):
            raise ValueError("Input model must be an instance of RestrictedDet.")
        else:
            U = model.U_full
            I = jnp.eye(U.shape[1], dtype=U.dtype)
            if not jnp.allclose(U.conj().T @ U, I):
                warn("Input orbitals aren't orthonormal. They will be orthonormalized.")
                model = model.normalize()
        return model

    @eqx.filter_jit
    def rho_from_model(self, model: RestrictedDet) -> jax.Array:
        r"""
        Get the one-body density matrix $\left< c_i^\dagger c_j \right>$ from the
        mean-field orbitals.

        Args:
            full_params: Full mean-field orbitals

        Returns:
            One-body density matrix
        """
        U = model.U
        inv = jnp.linalg.inv(U.conj().T @ U)
        rho0 = (U @ inv @ U.conj().T).T
        zeros = jnp.zeros_like(rho0)
        rho = jnp.block([[rho0, zeros], [zeros, rho0]])
        return rho

    def HF_update(self, hamiltonian) -> None:
        """Update the mean-field parameters with one step of Hartree-Fock iteration"""
        op_list = _reformat_fermion_op(hamiltonian.jax_op_list)
        rho = self.rho_from_model(self.model)
        self._energy, H = self._val_grad_rho(rho, op_list)
        N = get_sites().Nsites
        H = H[:N, :N]
        self._spectrum, U = jnp.linalg.eigh(H)
        Nup = get_sites().Ntotal // 2
        U = U[:, :Nup]
        self._model = RestrictedDet(
            U, self.model.dtype, self.model.preferred_element_type
        )


class UnrestrictedDetState(MeanFieldFermionState):
    """Unrestricted determinant mean-field state"""

    def _check_model(self, model):
        if model is None:
            model = UnrestrictedDet()
        elif not isinstance(model, UnrestrictedDet):
            raise ValueError("Input model must be an instance of UnrestrictedDet.")
        else:
            Utuple = model.U_full
            for U in Utuple:
                I = jnp.eye(U.shape[1], dtype=U.dtype)
                if not jnp.allclose(U.conj().T @ U, I):
                    warn(
                        "Input orbitals aren't orthonormal. They will be orthonormalized."
                    )
                    model = model.normalize()
                    return model
        return model

    @eqx.filter_jit
    def rho_from_model(self, model: UnrestrictedDet) -> jax.Array:
        r"""
        Get the one-body density matrix $\left< c_i^\dagger c_j \right>$ from the
        mean-field orbitals.

        Args:
            full_params: Full mean-field orbitals

        Returns:
            One-body density matrix
        """
        Uup, Udn = model.U
        inv_up = jnp.linalg.inv(Uup.conj().T @ Uup)
        rho_up = (Uup @ inv_up @ Uup.conj().T).T
        inv_dn = jnp.linalg.inv(Udn.conj().T @ Udn)
        rho_dn = (Udn @ inv_dn @ Udn.conj().T).T
        zeros_up = jnp.zeros((rho_up.shape[0], rho_dn.shape[1]), dtype=rho_up.dtype)
        zeros_dn = jnp.zeros((rho_dn.shape[0], rho_up.shape[1]), dtype=rho_dn.dtype)
        rho = jnp.block([[rho_up, zeros_up], [zeros_dn, rho_dn]])
        return rho

    def HF_update(self, hamiltonian) -> None:
        """Update the mean-field parameters with one step of Hartree-Fock iteration"""
        op_list = _reformat_fermion_op(hamiltonian.jax_op_list)
        rho = self.rho_from_model(self.model)
        self._energy, H = self._val_grad_rho(rho, op_list)
        N = get_sites().Nsites
        Hup = H[:N, :N]
        spectrum_up, Uup = jnp.linalg.eigh(Hup)
        Hdn = H[N:, N:]
        spectrum_dn, Udn = jnp.linalg.eigh(Hdn)
        self._spectrum = jnp.concatenate([spectrum_up, spectrum_dn])
        Nup, Ndn = get_sites().Nparticles
        Uup = Uup[:, :Nup]
        Udn = Udn[:, :Ndn]
        self._model = UnrestrictedDet(
            (Uup, Udn), self.model.dtype, self.model.preferred_element_type
        )


class MultiDetState(MeanFieldFermionState):
    """Multi-determinant mean-field state"""

    def _check_model(self, model):
        if model is None:
            model = MultiDet()
        elif not isinstance(model, MultiDet):
            raise ValueError("Input model must be an instance of MultiDet.")
        else:
            U = model.U_full
            I = jnp.eye(U.shape[-1], dtype=U.dtype)[None]
            if not jnp.allclose(U.conj().mT @ U, I):
                warn("Input orbitals aren't orthonormal. They will be orthonormalized.")
                model = model.normalize()
        return model

    @eqx.filter_jit
    def _expectation_from_model(self, model: MultiDet, jax_op_list: list) -> jax.Array:
        # model = model.normalize()  # might give better gradients
        ndets = model.ndets
        idxu0, idxu1 = jnp.triu_indices(ndets)
        idxl0, idxl1 = jnp.tril_indices(ndets, k=-1)
        c = model.coeffs
        U = model.U_full
        U0 = U[idxu0]
        U1 = U[idxu1]
        X = U0.conj().mT @ U1
        S = jnp.linalg.det(X)
        zeros = jnp.zeros((ndets, ndets), dtype=S.dtype)
        S = zeros.at[idxu0, idxu1].set(S)
        S = S.at[idxl0, idxl1].set(S.conj().T[idxl0, idxl1])

        T = U1 @ jnp.linalg.inv(X) @ U0.conj().mT
        M = T.shape[-1]
        zeros = jnp.zeros((ndets, ndets, M, M), dtype=T.dtype)
        T = zeros.at[idxu0, idxu1].set(T)
        T = T.at[idxl0, idxl1].set(jnp.transpose(T.conj(), (1, 0, 3, 2))[idxl0, idxl1])
        I = jnp.eye(M, dtype=T.dtype)[None, None]
        T_ = I - T.mT

        def get_contract(T, T_, opstr, index_array):
            if len(opstr) % 2 == 1:
                return 0.0

            if len(opstr) == 2:
                idx0 = index_array[:, 0]
                idx1 = index_array[:, 1]
                if opstr == "+-":
                    return T[idx0, idx1]
                elif opstr == "-+":
                    return T_[idx0, idx1]
                elif opstr == "++":
                    return 0.0
                elif opstr == "--":
                    return 0.0
                else:
                    raise NotImplementedError

            output = 0.0
            for i, c in enumerate(opstr[1:]):
                i += 1
                current_op = opstr[0] + c
                current_idx = index_array[:, [0, i]]
                current_contract = get_contract(T, T_, current_op, current_idx)

                remain_op = opstr[1:i] + opstr[i + 1 :]
                idx = list(range(1, i)) + list(range(i + 1, len(opstr)))
                remain_idx = index_array[:, idx]
                remain_contract = get_contract(T, T_, remain_op, remain_idx)
                sign = 1 if i % 2 == 1 else -1
                output += sign * current_contract * remain_contract

            return output

        contract_vmap = jax.vmap(get_contract, in_axes=(0, 0, None, None))
        contract_vmap = jax.vmap(contract_vmap, in_axes=(0, 0, None, None))

        output = jnp.array(0.0, T.dtype)
        for opstr, J_array, index_array in jax_op_list:
            contract = contract_vmap(T, T_, opstr, index_array)
            contract = jnp.sum(J_array[None, None, :] * contract, axis=2)
            contract *= S
            contract = (c.conj() @ contract @ c) / (c.conj() @ S @ c)
            output += contract
        return output


class GeneralPfState(State):
    """Mean-field pfaffian state"""

    def __init__(self, F: Optional[jax.Array] = None):
        sites = get_sites()

        if sites.Nparticles is not None:
            warn(
                "The mean-field pfaffian state doesn't have a conserved number of particles"
            )

        M = sites.Nmodes
        shape = (M, M)
        dtype = get_default_dtype()
        if F is None:
            scale = np.sqrt(np.e / M, dtype=dtype)
            F = jr.normal(get_subkeys(), shape, dtype) * scale
        elif F.shape != shape:
            raise ValueError("Input orbital size incompatible with the system size.")
        F = (F - F.T) / 2
        self._F = F

        super().__init__()

    @property
    def F(self) -> jax.Array:
        return self._F

    @staticmethod
    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, 0))
    def _forward(F, s: jax.Array) -> jax.Array:
        idx = fermion_idx(s)
        return lrux.pf(F[idx, :][:, idx])

    def __call__(self, fock_states: _Array) -> jax.Array:
        fock_states = jnp.asarray(fock_states)

        Ntotal = get_sites().Ntotal
        if Ntotal is None:
            psi = []
            for s in fock_states:
                idx = fermion_idx(s)
                psi.append(lrux.pf(self.F[idx, :][:, idx]))
            return jnp.asarray(psi).flatten()
        else:
            return self._forward(self.F, fock_states)

    def expectation(self, operator) -> jax.Array:
        return self._expectation(self.F, operator.jax_op_list)

    @staticmethod
    @eqx.filter_jit
    def _expectation(F: jax.Array, jax_op_list: list) -> jax.Array:
        jax_op_list = _reformat_jax_op_list(jax_op_list)
        F = (F - F.T) / 2

        I = jnp.eye(F.shape[0])
        # < cc† >
        rho_ = jnp.linalg.inv(I + F.conj().T @ F)
        # < c†c >
        rho = I - rho_.T
        # < c†c† >
        Delta = (F @ rho_).T
        # < cc >
        Delta_ = -Delta.conj()

        def get_contract(opstr, index_array):
            if len(opstr) % 2 == 1:
                return 0.0

            if len(opstr) == 2:
                if opstr == "+-":
                    return rho[index_array[:, 0], index_array[:, 1]]
                elif opstr == "-+":
                    return rho_[index_array[:, 0], index_array[:, 1]]
                elif opstr == "++":
                    return Delta[index_array[:, 0], index_array[:, 1]]
                elif opstr == "--":
                    return Delta_[index_array[:, 0], index_array[:, 1]]
                else:
                    raise NotImplementedError

            output = 0.0
            for i, c in enumerate(opstr[1:]):
                i += 1
                current_op = opstr[0] + c
                current_idx = index_array[:, [0, i]]
                current_contract = get_contract(current_op, current_idx)

                remain_op = opstr[1:i] + opstr[i + 1 :]
                idx = list(range(1, i)) + list(range(i + 1, len(opstr)))
                remain_idx = index_array[:, idx]
                remain_contract = get_contract(remain_op, remain_idx)
                sign = 1 if i % 2 == 1 else -1
                output += sign * current_contract * remain_contract

            return output

        output = jnp.array(0.0, F.dtype)
        for opstr, J_array, index_array in jax_op_list:
            output += jnp.sum(J_array * get_contract(opstr, index_array))
        return output

    @eqx.filter_jit
    def _get_grad(self, F: jax.Array, jax_op_list: list) -> jax.Array:
        F = (F - F.T) / 2
        fn_energy = lambda F, jax_op_list: self._expectation(F, jax_op_list).real
        E, g = jax.value_and_grad(fn_energy)(F, jax_op_list)

        I = jnp.eye(F.shape[0])
        inv = jnp.linalg.inv(I + F.conj().T @ F)
        inv_diag = jnp.diag(inv)
        S_diag = jnp.outer(inv_diag, inv_diag) - inv * inv.conj()
        S_diag = jnp.where(jnp.isclose(S_diag, 0), 1, S_diag)
        g /= S_diag

        return E, (g - g.T) / 2

    def exact_reconfig(self, hamiltonian, step_size: float) -> jax.Array:
        E, g = self._get_grad(self._F, hamiltonian.jax_op_list)
        self._F -= step_size * g
        return E


class BCS_State(State):
    """BCS mean-field state"""

    def __init__(
        self, theta: Optional[jax.Array] = None, U: Optional[jax.Array] = None
    ):
        sites = get_sites()
        if not isinstance(sites, Grid):
            raise NotImplementedError(
                "MeanFieldBCS is only implemented for Grid lattice."
            )

        if sites.Nparticles is not None:
            warn(
                "The mean-field BCS state doesn't have a conserved number of particles"
            )

        N = sites.Nsites
        shape = (N,)
        if theta is None:
            theta = jnp.ones(N, get_real_dtype()) * jnp.pi / 4
        elif theta.shape != shape:
            raise ValueError("Input theta size incompatible with the system size.")
        self._theta = theta

        shape = (2 * N, 2 * N)
        if U is None:
            L = jnp.asarray(sites.shape[1:]).reshape(-1, 1)
            k0 = jnp.eye(sites.ndim) * 2 * jnp.pi / L
            n = sites.coord
            k = jnp.einsum("ij,ki->kj", k0, n)
            arg = jnp.argsort(jnp.einsum("kj,kj->k", k, k))
            k = k[arg]
            kr = jnp.einsum("kj,nj->nk", k, sites.coord)
            U = jnp.exp(1j * kr) / jnp.sqrt(N)

            zeros = jnp.zeros_like(U)
            U_up = jnp.concatenate([U, zeros], axis=0)
            U_down = jnp.concatenate([zeros, U.conj()], axis=0)
            U = jnp.stack([U_up, U_down], axis=2).reshape(2 * N, 2 * N)
        elif U.shape != shape:
            raise ValueError("Input orbital size incompatible with the system size.")
        self._U = U

        def get_expectation(theta, U, jax_op_list):
            return self._expectation(theta, U, jax_op_list).real

        self._val_grad = eqx.filter_jit(jax.value_and_grad(get_expectation))

        super().__init__()

    @property
    def theta(self) -> jax.Array:
        return self._theta

    @property
    def U(self) -> jax.Array:
        return self._U

    @property
    def F(self) -> jax.Array:
        f = jnp.tan(self.theta)
        f = jnp.stack([f, jnp.zeros_like(f)], axis=1).flatten()[:-1]
        D = jnp.diag(f, k=1)
        U = self.U
        F = U @ D @ U.T
        return F - F.T

    @staticmethod
    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, 0))
    def _forward(F, s: jax.Array) -> jax.Array:
        idx = fermion_idx(s)
        return lrux.pf(F[idx, :][:, idx])

    def __call__(self, fock_states: _Array) -> jax.Array:
        fock_states = jnp.asarray(fock_states)

        Ntotal = get_sites().Ntotal
        if Ntotal is None:
            psi = []
            for s in fock_states:
                idx = fermion_idx(s)
                psi.append(lrux.pf(self.F[idx, :][:, idx]))
            return jnp.asarray(psi).flatten()
        else:
            return self._forward(self.F, fock_states)

    def normalize(self) -> BCS_State:
        self._U, R = jnp.linalg.qr(self._U)
        return self

    def expectation(self, operator) -> jax.Array:
        return self._expectation(self._theta, self._U, operator.jax_op_list)

    @staticmethod
    @eqx.filter_jit
    def _expectation(theta: jax.Array, U: jax.Array, jax_op_list: list) -> jax.Array:
        jax_op_list = _reformat_jax_op_list(jax_op_list)

        N = U.shape[0]
        u = jnp.cos(theta)
        u_repeat = jnp.repeat(u, 2)
        v = jnp.sin(theta)
        v_repeat = jnp.repeat(v, 2)
        s = jnp.where(jnp.arange(N) % 2 == 0, 1, -1)
        suv = s * u_repeat * v_repeat
        Uflip = U.reshape(N, N // 2, 2)[:, :, ::-1].reshape(N, N)

        # < c†c >
        rho = jnp.einsum("a,ia,ja->ij", v_repeat**2, U.conj(), U)
        # < cc† >
        rho_ = jnp.einsum("a,ia,ja->ij", u_repeat**2, U, U.conj())
        # < c†c† >
        Delta = jnp.einsum("a,ia,ja->ij", suv, U.conj(), Uflip.conj())
        # < cc >
        Delta_ = -Delta.conj()

        def get_contract(opstr, index_array):
            if len(opstr) % 2 == 1:
                return 0.0

            if len(opstr) == 2:
                if opstr == "+-":
                    return rho[index_array[:, 0], index_array[:, 1]]
                elif opstr == "-+":
                    return rho_[index_array[:, 0], index_array[:, 1]]
                elif opstr == "++":
                    return Delta[index_array[:, 0], index_array[:, 1]]
                elif opstr == "--":
                    return Delta_[index_array[:, 0], index_array[:, 1]]
                else:
                    raise NotImplementedError

            output = 0.0
            for i, c in enumerate(opstr[1:]):
                i += 1
                current_op = opstr[0] + c
                current_idx = index_array[:, [0, i]]
                current_contract = get_contract(current_op, current_idx)

                remain_op = opstr[1:i] + opstr[i + 1 :]
                idx = list(range(1, i)) + list(range(i + 1, len(opstr)))
                remain_idx = index_array[:, idx]
                remain_contract = get_contract(remain_op, remain_idx)
                sign = 1 if i % 2 == 1 else -1
                output += sign * current_contract * remain_contract

            return output

        output = jnp.array(0.0, U.dtype)
        for opstr, J_array, index_array in jax_op_list:
            output += jnp.sum(J_array * get_contract(opstr, index_array))
        return output

    def exact_reconfig(self, hamiltonian, step_size: float) -> jax.Array:
        E, g = self._val_grad(self._theta, self._U, hamiltonian.jax_op_list)
        self._theta -= step_size * g
        return E
