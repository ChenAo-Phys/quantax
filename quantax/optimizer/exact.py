from typing import Union, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm
from numbers import Number
from ..state import DenseState
from ..operator import Operator


class ExactTimeEvol:
    def __init__(self, init_state: DenseState, hamiltonian: Operator) -> None:
        self._init_state = init_state
        self._symm = self._init_state.symm
        self._eigs, self._U = hamiltonian.diagonalize(self._symm, "full")
        self._eigs = jnp.asarray(self._eigs)
        self._U = jnp.asarray(self._U)

    def get_evolved_wf(self, time: Union[float, jax.Array]) -> jax.Array:
        is_float = isinstance(time, float)
        if is_float:
            time = jnp.array([time])
        exp_eigs = jnp.exp(-1j * jnp.einsum("t,d->td", time, self._eigs))
        wf0 = self._init_state.wave_function
        wf = jnp.einsum("ij,tj,kj,k->ti", self._U, exp_eigs, self._U.conj(), wf0)
        if is_float:
            wf = wf[0]
        return wf

    def expectation(
        self, operator: Operator, time: Union[float, jax.Array]
    ) -> Union[Number, jax.Array]:
        wf = self.get_evolved_wf(time)
        wf /= norm(wf, axis=1, keepdims=True)
        op = operator.get_quspin_op(self._symm)

        # this hasn't been tested
        out = jnp.einsum("ti,it->t", wf.conj(), op.dot(np.ascontiguousarray(wf.T)))
        if isinstance(time, float):
            out = out.item()
        return out
