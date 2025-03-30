from __future__ import annotations
from typing import Optional, Union
from functools import partial
from warnings import warn
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from .state import State
from ..model.fermion_mf import _get_fermion_idx, Determinant
from ..global_defs import get_subkeys, get_sites, get_default_dtype
from ..utils import det


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


class MeanFieldDet(State):
    """Determinant mean-field state"""

    def __init__(self, U: Optional[jax.Array] = None):
        sites = get_sites()
        if sites.Nparticle is None:
            raise ValueError(
                "The total number of particles should be specified for the "
                "mean-field determinant state."
            )
        elif isinstance(sites.Nparticle, tuple):
            warn(
                "The mean-field determinant state doesn't have a conserved amount of "
                "spin-up and spin-down particles."
            )

        dtype = get_default_dtype()
        shape = (2 * sites.N, sites.Ntotal)
        if U is None:
            U = jr.normal(get_subkeys(), shape, dtype)
            U, R = jnp.linalg.qr(U)
        elif U.shape != shape:
            raise ValueError("Input orbital size incompatible with the system size.")
        self._U = U

        get_expectation = lambda U, jax_op_list: self._expectation(U, jax_op_list).real
        self._val_grad = eqx.filter_jit(jax.value_and_grad(get_expectation))

        super().__init__()

    @property
    def U(self) -> jax.Array:
        return self._U
    
    @property
    def model(self) -> Determinant:
        model = Determinant(self._U.dtype)
        return eqx.tree_at(lambda tree: tree.U, model, self._U)

    @staticmethod
    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, 0))
    def _forward(U, s: jax.Array) -> jax.Array:
        idx = _get_fermion_idx(s, get_sites().Ntotal)
        return det(U[idx, :])

    def __call__(self, fock_states: jax.Array) -> jax.Array:
        fock_states = jnp.asarray(fock_states)
        return self._forward(self._U, fock_states)

    def normalize(self) -> MeanFieldDet:
        self._U, R = jnp.linalg.qr(self._U)
        self._model_updated = False
        return self

    def expectation(self, operator, samples=None) -> jax.Array:
        if samples is not None:
            warn("Expectation value of mean-field state doesn't rely on samples.")

        return self._expectation(self._U, operator.jax_op_list)

    @staticmethod
    @eqx.filter_jit
    def _expectation(U: jax.Array, jax_op_list: list) -> jax.Array:
        U, R = jnp.linalg.qr(U)
        jax_op_list = _reformat_jax_op_list(jax_op_list)
        # < c†c >
        rho = (U @ U.conj().T).T
        # < cc† >
        rho_ = jnp.eye(rho.shape[0], dtype=rho.dtype) - rho.T

        def get_contract(opstr, index_array):
            if len(opstr) % 2 == 1:
                return 0.0

            if len(opstr) == 2:
                if opstr == "+-":
                    return rho[index_array[:, 0], index_array[:, 1]]
                elif opstr == "-+":
                    return rho_[index_array[:, 0], index_array[:, 1]]
                else:
                    return 0.0

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
        E, g = self._val_grad(self._U, hamiltonian.jax_op_list)
        self._U -= step_size * g.conj()
        return E
