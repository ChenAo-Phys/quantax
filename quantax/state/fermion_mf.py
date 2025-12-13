from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple, Union, BinaryIO
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

if TYPE_CHECKING:
    from ..operator import Operator


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
def _reformat_spin_op(jax_op_list: list) -> list:
    reformatted_list = []
    N = get_sites().Nsites

    for opstr, J_array, index_array in jax_op_list:
        new_op = ""
        new_idx = []
        for op, idx in zip(opstr, index_array.T):
            if op == "I":
                continue
            elif op == "+":
                new_op += "+-"
                new_idx.append(idx)
                new_idx.append(idx + N)
            elif op == "-":
                new_op += "+-"
                new_idx.append(idx + N)
                new_idx.append(idx)
            elif op == "z":
                new_op += "+-"
                J_array = jnp.concatenate([J_array / 2, -J_array / 2])
                idx = idx[:, None]
                new_idx.append(jnp.block([[idx, idx], [idx + N, idx + N]]))
            elif op == "x":
                new_op += "+-"
                J_array = jnp.concatenate([J_array / 2, J_array / 2])
                idx = idx[:, None]
                new_idx.append(jnp.block([[idx, idx + N], [idx + N, idx]]))
            elif op == "y":
                new_op += "+-"
                J_array = jnp.concatenate([-1j * J_array / 2, 1j * J_array / 2])
                idx = idx[:, None]
                new_idx.append(jnp.block([[idx, idx + N], [idx + N, idx]]))

        n_terms = index_array.shape[0]
        expanded_idx = jnp.empty((n_terms, 0), dtype=index_array.dtype)
        for idx in new_idx:
            reps = expanded_idx.shape[0] // n_terms
            if idx.ndim == 1:
                idx = jnp.tile(idx[:, None], (reps, 1))
            else:
                expanded_idx = jnp.tile(expanded_idx, (2, 1))
                idx = jnp.tile(idx.reshape(2, -1, 2), (1, reps, 1)).reshape(-1, 2)
            expanded_idx = jnp.concatenate([expanded_idx, idx], axis=1)

        reformatted_list.append([new_op, J_array, expanded_idx])

    return reformatted_list


def _get_op_list(operator: Union[Operator, list]) -> list:
    if isinstance(operator, list):
        return operator

    if get_sites().is_fermion:
        return _reformat_fermion_op(operator.jax_op_list)
    else:
        return _reformat_spin_op(operator.jax_op_list)


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
            model, param_file, max_parallel=max_parallel, use_ref=use_refmodel
        )
        self._energy = None

        loss_model = lambda model, op: self._expectation_from_model(model, op).real
        self._val_grad_model = eqx.filter_jit(eqx.filter_value_and_grad(loss_model))
        loss_rho = lambda rho, op: self._expectation_from_rho(rho, op).real
        self._val_grad_rho = eqx.filter_jit(eqx.filter_value_and_grad(loss_rho))

    def _check_model(self, model: Optional[eqx.Module]) -> eqx.Module:
        """Check the input model and initialize if None"""
        if model is None:
            raise NotImplementedError
        return model

    @classmethod
    def is_paired(cls) -> bool:
        """Whether the state is a paired state (pfaffian) or not (determinant), default to False"""
        return False

    @property
    def energy(self) -> Optional[float]:
        """The energy in the previous optimization step."""
        return self._energy

    @classmethod
    @eqx.filter_jit
    def rho_from_model(
        cls, model: eqx.Module
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        r"""
        Get the one-body density matrix $\rho_{ij} = \left< c_i^\dagger c_j \right>$
        from the mean-field parameters. If the state is paired, return a tuple of
        $\rho = \left< c_i^\dagger c_j \right>$ and $\kappa = \left< c_i^\dagger c_j^\dagger \right>$.

        :param model:
            The mean-field model.

        :return:
            One-body density matrix.
        """
        return NotImplemented

    def expectation(self, operator: Operator) -> jax.Array:
        """
        Compute the expectation value of an operator.

        :param operator:
            The operator to compute the expectation value of. It should be an instance of
            `~quantax.operator.Operator`.

        :param model:
            The mean-field model to use. If None, use the current model.
        """
        jax_op_list = _get_op_list(operator)
        return self._expectation_from_model(self.model, jax_op_list)

    def get_loss_fn(self, hamiltonian: Operator):
        """
        Get the loss function for optimization.

        :param hamiltonian:
            The Hamiltonian to compute the gradient of.
        """
        jax_op_list = _get_op_list(hamiltonian)
        params, static = eqx.partition(self.model, eqx.is_inexact_array)

        def loss_fn(params, *args):
            model = eqx.combine(params, static)
            return self._expectation_from_model(model, jax_op_list).real

        return jax.jit(loss_fn)

    @classmethod
    def _expectation_from_model(cls, model: MultiDet, jax_op_list: list) -> jax.Array:
        """
        Compute the expectation value of an operator from the mean-field model.
        """
        rho = cls.rho_from_model(model)
        return cls._expectation_from_rho(rho, jax_op_list)

    @classmethod
    @eqx.filter_jit
    def _expectation_from_rho(
        cls, rho: Union[jax.Array, Tuple[jax.Array, jax.Array]], jax_op_list: list
    ) -> jax.Array:
        """
        Compute the expectation value of an operator from the one-body density matrix.
        """
        if cls.is_paired():
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
                    return kappa[idx0, idx1] if cls.is_paired() else 0.0
                elif opstr == "--":
                    return kappa_[idx0, idx1] if cls.is_paired() else 0.0
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

    def get_step(self, hamiltonian: Operator) -> jax.Array:
        """Get the gradient of the energy with respect to the mean-field parameters.

        :param hamiltonian:
            The Hamiltonian to compute the gradient of.
        """
        jax_op_list = _get_op_list(hamiltonian)
        self._energy, g = self._val_grad_model(self.model, jax_op_list)
        g, _ = jfu.ravel_pytree(g)
        return g.conj()


class GeneralDetState(MeanFieldFermionState):
    """Determinant mean-field state, a wrapper of `~quantax.model.GeneralDet`."""

    def _check_model(self, model) -> GeneralDet:
        if model is None:
            model = GeneralDet()
        elif not isinstance(model, GeneralDet):
            raise ValueError("Input model must be an instance of GeneralDet.")
        return model

    @classmethod
    @eqx.filter_jit
    def rho_from_model(cls, model: GeneralDet) -> jax.Array:
        r"""
        Get the one-body density matrix $\rho_{ij} = \left< c_i^\dagger c_j \right>$
        from the mean-field parameters.

        :param model:
            The mean-field model.

        :return:
            One-body density matrix.
        """
        U = model.U_full
        inv = jnp.linalg.inv(U.conj().T @ U)
        return U @ inv @ U.conj().T


class RestrictedDetState(MeanFieldFermionState):
    """Restricted determinant mean-field state, a wrapper of `~quantax.model.RestrictedDet`."""

    def _check_model(self, model):
        if model is None:
            model = RestrictedDet()
        elif not isinstance(model, RestrictedDet):
            raise ValueError("Input model must be an instance of RestrictedDet.")
        return model

    @classmethod
    @eqx.filter_jit
    def rho_from_model(cls, model: RestrictedDet) -> jax.Array:
        r"""
        Get the one-body density matrix $\left< c_i^\dagger c_j \right>$ from the
        mean-field orbitals.

        :param model:
            The mean-field model.
        """
        U = model.U_full
        inv = jnp.linalg.inv(U.conj().T @ U)
        rho0 = U @ inv @ U.conj().T
        zeros = jnp.zeros_like(rho0)
        rho = jnp.block([[rho0, zeros], [zeros, rho0]])
        return rho


class UnrestrictedDetState(MeanFieldFermionState):
    """Unrestricted determinant mean-field state, a wrapper of `~quantax.model.UnrestrictedDet`."""

    def _check_model(self, model):
        if model is None:
            model = UnrestrictedDet()
        elif not isinstance(model, UnrestrictedDet):
            raise ValueError("Input model must be an instance of UnrestrictedDet.")
        return model

    @classmethod
    @eqx.filter_jit
    def rho_from_model(cls, model: UnrestrictedDet) -> jax.Array:
        r"""
        Get the one-body density matrix $\left< c_i^\dagger c_j \right>$ from the
        mean-field orbitals.

        :param model:
            The mean-field model.
        """
        Uup, Udn = model.U_full
        inv_up = jnp.linalg.inv(Uup.conj().T @ Uup)
        rho_up = Uup @ inv_up @ Uup.conj().T
        inv_dn = jnp.linalg.inv(Udn.conj().T @ Udn)
        rho_dn = Udn @ inv_dn @ Udn.conj().T
        zeros_up = jnp.zeros((rho_up.shape[0], rho_dn.shape[1]), dtype=rho_up.dtype)
        zeros_dn = jnp.zeros((rho_dn.shape[0], rho_up.shape[1]), dtype=rho_dn.dtype)
        rho = jnp.block([[rho_up, zeros_up], [zeros_dn, rho_dn]])
        return rho


class MultiDetState(MeanFieldFermionState):
    """Multi-determinant mean-field state, a wrapper of `~quantax.model.MultiDet`."""

    def _check_model(self, model):
        if model is None:
            model = MultiDet()
        elif not isinstance(model, MultiDet):
            raise ValueError("Input model must be an instance of MultiDet.")
        else:
            U = model.U_full
            I = jnp.eye(U.shape[-1], dtype=U.dtype)[None]
            if not jnp.allclose(U.conj().mT @ U, I):
                if jax.process_index() == 0:
                    warn(
                        "Input orbitals aren't orthonormal. They will be orthonormalized."
                    )
                model = model.normalize()
        return model

    def expectation(
        self, operator: Operator, model: Optional[MultiDet] = None
    ) -> jax.Array:
        if model is None:
            model = self.model
        jax_op_list = _get_op_list(operator)
        return self._expectation_from_model(model, jax_op_list)

    @classmethod
    @eqx.filter_jit
    def _expectation_from_model(cls, model: MultiDet, jax_op_list: list) -> jax.Array:
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
    """Mean-field pfaffian state, a wrapper of `~quantax.model.GeneralPf`."""

    def _check_model(self, model):
        if model is None:
            model = GeneralPf()
        elif not isinstance(model, GeneralPf):
            raise ValueError("Input model must be an instance of GeneralPf.")
        return model

    @classmethod
    def is_paired(cls) -> bool:
        """Whether the state is a paired state (pfaffian) or not (determinant)"""
        return True

    @classmethod
    @eqx.filter_jit
    def rho_from_model(cls, model: GeneralPf) -> Tuple[jax.Array, jax.Array]:
        r"""
        Get a tuple of
        $\rho = \left< c_i^\dagger c_j \right>$ and
        $\kappa = \left< c_i^\dagger c_j^\dagger \right>$.

        :param model:
            The mean-field model.
        """
        F = model.F_full
        I = jnp.eye(F.shape[0])
        rho_ = jnp.linalg.inv(I + F.conj().T @ F)
        rho = I - rho_.T
        kappa = (F @ rho_).conj().T
        return rho, kappa


class SingletPairState(MeanFieldFermionState):
    """Singlet-paired mean-field state, a wrapper of `~quantax.model.SingletPair`."""

    def _check_model(self, model):
        if model is None:
            model = SingletPair()
        elif not isinstance(model, SingletPair):
            raise ValueError("Input model must be an instance of SingletPair.")
        return model

    @classmethod
    def is_paired(cls) -> bool:
        """Whether the state is a paired state (pfaffian) or not (determinant)"""
        return True

    @classmethod
    @eqx.filter_jit
    def rho_from_model(cls, model: SingletPair) -> Tuple[jax.Array, jax.Array]:
        r"""
        Get a tuple of
        $\rho = \left< c_i^\dagger c_j \right>$ and
        $\kappa = \left< c_i^\dagger c_j^\dagger \right>$.

        :param model:
            The mean-field model.
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


class MultiPfState(MeanFieldFermionState):
    """Multi-Pfaffian mean-field state, a wrapper of `~quantax.model.MultiPf`."""

    def _check_model(self, model):
        if model is None:
            model = MultiPf()
        elif not isinstance(model, MultiPf):
            raise ValueError("Input model must be an instance of MultiPf.")
        return model

    @classmethod
    def is_paired(cls) -> bool:
        """Whether the state is a paired state (pfaffian) or not (determinant)"""
        return True

    def expectation(
        self, operator: Operator, model: Optional[MultiPf] = None
    ) -> jax.Array:
        if model is None:
            model = self.model
        jax_op_list = _get_op_list(operator)
        return self._expectation_from_model(model, jax_op_list)

    @classmethod
    @eqx.filter_jit
    def _expectation_from_model(cls, model: MultiPf, jax_op_list: list) -> jax.Array:
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
