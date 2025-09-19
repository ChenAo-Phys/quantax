from __future__ import annotations
from typing import Optional, Tuple, Union, BinaryIO
from pathlib import Path
from warnings import warn
import jax
import jax.numpy as jnp
import jax.flatten_util as jfu
import equinox as eqx
import lrux
from .variational import Variational
from ..model import (
    GeneralDet,
    RestrictedDet,
    UnrestrictedDet,
    MultiDet,
    GeneralPf,
    SingletPair,
    MultiPf,
)
from ..global_defs import get_sites


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


def _get_op_list(operator) -> list:
    if isinstance(operator, list):
        return operator

    if get_sites().is_fermion:
        return _reformat_fermion_op(operator.jax_op_list)
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

        exp_real = lambda model, op_list: self.expectation(op_list, model).real
        self._val_grad_model = eqx.filter_jit(eqx.filter_value_and_grad(exp_real))
        exp_real = lambda rho, op_list: self._expectation_from_rho(rho, op_list).real
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
    def rho_from_model(
        self, model: eqx.Module
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        r"""
        Get the one-body density matrix $\rho_{ij} = \left< c_i^\dagger c_j \right>$
        from the mean-field parameters. If the state is paired, return a tuple of
        $\rho = \left< c_i^\dagger c_j \right>$ and $\kappa = \left< c_i^\dagger c_j^\dagger \right>$.

        Args:
            full_params: Full mean-field orbitals, for instance,
            U for determinant state and F for pfaffian state

        Returns:
            One-body density matrix
        """
        return NotImplemented

    def expectation(self, operator, model: Optional[eqx.Module] = None) -> jax.Array:
        if model is None:
            model = self.model
        jax_op_list = _get_op_list(operator)
        rho = self.rho_from_model(model)
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
        jax_op_list = _get_op_list(hamiltonian)
        self._energy, g = self._val_grad_model(self.model, jax_op_list)
        g, _ = jfu.ravel_pytree(g)
        return g.conj()

    def HF_update(self, hamiltonian) -> None:
        """Update the mean-field parameters with one step of Hartree-Fock iteration"""
        raise NotImplementedError


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
        rho = self.rho_from_model(self.model)
        op_list = _get_op_list(hamiltonian)
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
        rho = self.rho_from_model(self.model)
        op_list = _get_op_list(hamiltonian)
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
        rho = self.rho_from_model(self.model)
        op_list = _get_op_list(hamiltonian)
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

    def expectation(self, operator, model: Optional[MultiDet] = None) -> jax.Array:
        if model is None:
            model = self.model
        jax_op_list = _get_op_list(operator)
        return self._expectation_from_model(model, jax_op_list)

    @eqx.filter_jit
    def _expectation_from_model(self, model: MultiDet, jax_op_list: list) -> jax.Array:
        model = model.normalize()
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
                    return jnp.zeros(idx0.size, T.dtype)
                elif opstr == "--":
                    return jnp.zeros(idx0.size, T.dtype)
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


class GeneralPfState(MeanFieldFermionState):
    """Mean-field pfaffian state"""

    def _check_model(self, model):
        if model is None:
            model = GeneralPf()
        elif not isinstance(model, GeneralPf):
            raise ValueError("Input model must be an instance of GeneralPf.")
        return model

    @property
    def is_paired(self) -> bool:
        """Whether the state is a paired state (pfaffian) or not (determinant)"""
        return True

    @eqx.filter_jit
    def rho_from_model(self, model: GeneralPf) -> Tuple[jax.Array, jax.Array]:
        r"""
        Get a tuple of
        $\rho = \left< c_i^\dagger c_j \right>$ and $\kappa = \left< c_i c_j \right>$.

        Args:
            full_params: Full mean-field orbitals, for instance,
            U for determinant state and F for pfaffian state

        Returns:
            One-body density matrix
        """
        F = model.F_full
        I = jnp.eye(F.shape[0], dtype=F.dtype)
        O = jnp.zeros_like(I)
        rho_1 = jnp.linalg.inv(I + F.conj() @ F.T)
        rho_2 = jnp.linalg.inv(I + F.conj().T @ F)

        rho1 = I - rho_1.T
        rho2 = I - rho_2.T
        rho = jnp.block([[rho1, O], [O, rho2]])
        kappa1 = -(F.T @ rho_1).T
        kappa2 = (F @ rho_2).T
        kappa = jnp.block([[O, kappa1], [kappa2, O]])
        return rho, kappa


class SingletPairState(MeanFieldFermionState):
    """Singlet-paired mean-field state"""

    def _check_model(self, model):
        if model is None:
            model = SingletPair()
        elif not isinstance(model, SingletPair):
            raise ValueError("Input model must be an instance of SingletPair.")
        return model

    @property
    def is_paired(self) -> bool:
        """Whether the state is a paired state (pfaffian) or not (determinant)"""
        return True

    @eqx.filter_jit
    def rho_from_model(self, model: SingletPair) -> Tuple[jax.Array, jax.Array]:
        r"""
        Get a tuple of
        $\rho = \left< c_i^\dagger c_j \right>$ and $\kappa = \left< c_i c_j \right>$.

        Args:
            full_params: Full mean-field orbitals, for instance,
            U for determinant state and F for pfaffian state

        Returns:
            One-body density matrix
        """
        F = model.F_full
        O = jnp.zeros_like(F)
        F = jnp.block([[O, F], [-F.T, O]])
        I = jnp.eye(F.shape[0])
        rho_ = jnp.linalg.inv(I + F.conj().T @ F)
        rho = I - rho_.T
        kappa = (F @ rho_).conj().T
        return rho, kappa


class MultiPfState(MeanFieldFermionState):
    """Multi-Pfaffian mean-field state"""

    def _check_model(self, model):
        if model is None:
            model = MultiPf()
        elif not isinstance(model, MultiPf):
            raise ValueError("Input model must be an instance of MultiPf.")
        return model

    def expectation(self, operator, model: Optional[MultiPf] = None) -> jax.Array:
        if model is None:
            model = self.model
        jax_op_list = _get_op_list(operator)
        return self._expectation_from_model(model, jax_op_list)

    @eqx.filter_jit
    def _expectation_from_model(self, model: MultiPf, jax_op_list: list) -> jax.Array:
        npfs = model.npfs
        idxu0, idxu1 = jnp.triu_indices(npfs)
        idxl0, idxl1 = jnp.tril_indices(npfs, k=-1)
        F = model.F_full
        F0 = F[idxu0]
        F1 = F[idxu1]
        I = jnp.eye(F.shape[-1], dtype=F.dtype)
        I = jnp.tile(I, (F0.shape[0], 1, 1))
        mat = jax.vmap(jnp.block)([[F1, -I], [I, F0.conj().mT]])
        S = lrux.pf(mat)

        O = jnp.zeros((npfs, npfs), dtype=S.dtype)
        S = O.at[idxu0, idxu1].set(S)
        S = S.at[idxl0, idxl1].set(S.conj().mT[idxl0, idxl1])

        T_ = jnp.linalg.inv(I + F0.conj().mT @ F1)
        T = I - T_.mT
        K = -F1 @ T_
        K_ = T_ @ F0.conj()
        Gamma = jax.vmap(jnp.block)([[T_, K_], [K, T]])

        n = Gamma.shape[-1]
        O = jnp.zeros((npfs, npfs, n, n), dtype=Gamma.dtype)
        Gamma = O.at[idxu0, idxu1].set(Gamma)
        Gamma_T = jnp.transpose(Gamma.conj(), (1, 0, 3, 2))
        Gamma = Gamma.at[idxl0, idxl1].set(Gamma_T[idxl0, idxl1])
        # [[T_, K_], [K, T]] -> [[K_, T_], [T, K]]
        Gamma = jnp.roll(Gamma, shift=n // 2, axis=-1)

        def get_contract(Gamma, opstr, index_array):
            if len(opstr) % 2 == 1:
                return 0.0

            if len(opstr) == 2:
                is_create = jnp.array([c == "+" for c in opstr])
                index_array += n // 2 * is_create[None]
                mat = jax.vmap(lambda idx: Gamma[idx, :][:, idx])(index_array)
                idxl = jnp.tril_indices_from(mat[0], k=-1)
                mat = mat.at[:, idxl[0], idxl[1]].set(-mat.mT[:, idxl[0], idxl[1]])
                return lrux.pf(mat)

            output = 0.0
            for i, c in enumerate(opstr[1:]):
                i += 1
                current_op = opstr[0] + c
                current_idx = index_array[:, [0, i]]
                current_contract = get_contract(Gamma, current_op, current_idx)

                remain_op = opstr[1:i] + opstr[i + 1 :]
                idx = list(range(1, i)) + list(range(i + 1, len(opstr)))
                remain_idx = index_array[:, idx]
                remain_contract = get_contract(Gamma, remain_op, remain_idx)
                sign = 1 if i % 2 == 1 else -1
                output += sign * current_contract * remain_contract

            return output

        contract_vmap = jax.vmap(get_contract, in_axes=(0, None, None))
        contract_vmap = jax.vmap(contract_vmap, in_axes=(0, None, None))

        output = jnp.array(0.0, Gamma.dtype)
        for opstr, J_array, index_array in jax_op_list:
            contract = contract_vmap(Gamma, opstr, index_array)
            contract = jnp.sum(J_array[None, None, :] * contract, axis=2)
            contract = jnp.sum(contract * S) / jnp.sum(S)
            output += contract
        return output
