from __future__ import annotations
from typing import Optional, Tuple, Union
from numbers import Number
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax.ops import segment_sum
from jax.experimental.multihost_utils import process_allgather
import equinox as eqx
from quspin.operators import hamiltonian

from ..state import State, DenseState
from ..sampler import Samples
from ..symmetry import Symmetry, Identity
from ..utils import (
    get_global_sharding,
    local_to_replicate,
    to_global_array,
    global_to_local,
    local_to_global,
    to_replicate_numpy,
)
from ..global_defs import get_sites, get_default_dtype


def _apply_site_operator(
    x: jax.Array, opstr: str, J: jax.Array, idx: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    # diagonal
    if opstr == "I":
        return x, J

    if opstr == "n":
        return x, jnp.where(x[idx] > 0, J, 0)

    if opstr == "z":
        return x, J * x[idx] / 2

    # off-diagonal
    if get_sites().is_fermion:
        num_fermion = jnp.cumulative_sum(x[::-1] > 0, include_initial=True)[::-1]
        J = jnp.where(num_fermion[idx + 1] % 2 == 0, J, -J)
    elif opstr in ("x", "y"):
        J /= 2

    if opstr == "+":
        J = jnp.where(x[idx] < 0, J, jnp.nan)
        return x.at[idx].set(1), J

    if opstr == "-":
        J = jnp.where(x[idx] > 0, J, jnp.nan)
        return x.at[idx].set(-1), J

    if opstr == "x":
        return x.at[idx].mul(-1), J

    if opstr == "y":
        J *= 1j * x[idx]
        if get_sites().is_fermion:
            J *= -1  # conventional sign difference
        return x.at[idx].mul(-1), J


@eqx.filter_jit
@partial(jax.vmap, in_axes=(0, None))
def _apply_diag(x, jax_op_list):
    Hz = 0
    apply_fn = jax.vmap(_apply_site_operator, in_axes=(None, None, 0, 0))

    for opstr, J, index in jax_op_list:
        if all(op in ("I", "n", "z") for op in opstr):
            for op, idx in zip(opstr, index.T):
                _, J = apply_fn(x, op, J, idx)
            Hz += jnp.sum(J)

    return Hz


@eqx.filter_jit
@partial(jax.vmap, in_axes=(0, None))
def _apply_off_diag(x, jax_op_list):
    out = dict()
    apply_fn = jax.vmap(_apply_site_operator, in_axes=(0, None, 0, 0))

    for opstr, J, index in jax_op_list:
        nflips = sum(1 for s in opstr if s not in ("I", "n", "z"))
        if nflips > 0:
            x_conn = jnp.repeat(x[None, :], J.size, axis=0)
            for op, idx in zip(reversed(opstr), reversed(index.T)):
                x_conn, J = apply_fn(x_conn, op, J, idx)

            if nflips in out:
                out[nflips][0].append(x_conn)
                out[nflips][1].append(J)
            else:
                out[nflips] = [[x_conn], [J]]

    for nflips, (x_conn, J) in out.items():
        out[nflips] = [jnp.concatenate(x_conn), jnp.concatenate(J)]

    return out


class Operator:
    """Quantum operator"""

    def __init__(self, op_list: list):
        """
        :param op_list:
            The operator represented as a list in the
            `QuSpin format <https://quspin.github.io/QuSpin/generated/quspin.operators.hamiltonian.html#quspin.operators.hamiltonian.__init__>`_

            ``[[opstr1, [strength1, index11, index12, ...]], [opstr2, [strength2, index21, index22, ...]], ...]``

                opstr:
                    a `string <https://quspin.github.io/QuSpin/basis.html>`_ representing the operator type.
                    The convention is chosen the same as ``pauli=0`` in
                    `QuSpin <https://quspin.github.io/QuSpin/generated/quspin.basis.spin_basis_general.html#quspin.basis.spin_basis_general.__init__>`_

                strength:
                    interaction strength

                index:
                    the site index that operators act on
        """
        self._op_list = op_list
        self._jax_op_list = None
        self._quspin_op = dict()
        self._connectivity = None

    @property
    def op_list(self) -> list:
        """Operator represented as a list in the QuSpin format"""
        return self._op_list

    @property
    def jax_op_list(self) -> list:
        """
        Operator list with jax arrays, made easy for applying operator to fock states
        """
        if self._jax_op_list is None:
            self._jax_op_list = []
            for opstr, interaction in self.op_list:
                J_array = []
                index_array = []
                for J, *index in interaction:
                    J_array.append(J)
                    index_array.append(index)
                J_array = local_to_replicate(J_array).astype(get_default_dtype())
                index_array = local_to_replicate(index_array).astype(jnp.uint16)
                self._jax_op_list.append([opstr, J_array, index_array])

        return self._jax_op_list

    @property
    def expression(self) -> str:
        """The operator as a human-readable expression"""
        SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        OP = str.maketrans({"x": "Sˣ", "y": "Sʸ", "z": "Sᶻ", "+": "S⁺", "-": "S⁻"})
        expression = []
        for opstr, interaction in self.op_list:
            for J, *index in interaction:
                expression.append(f"{J:+}")
                for op, i in zip(opstr, index):
                    expression.append(f"{op.translate(OP)}{str(i).translate(SUB)}")
        return " ".join(expression)

    def __repr__(self) -> str:
        return self.expression

    def get_quspin_op(self, symm: Optional[Symmetry] = None) -> hamiltonian:
        """
        Obtain the corresponding
        `QuSpin operator <https://quspin.github.io/QuSpin/generated/quspin.operators.hamiltonian.html#quspin.operators.hamiltonian.__init__>`_

        :param symm:
            The symmetry used for generate the operator basis, by default the basis
            without symmetry
        """
        if symm is None:
            symm = Identity()
        symm.basis_make()
        if symm not in self._quspin_op:
            self._quspin_op[symm] = hamiltonian(
                static_list=self.op_list,
                dynamic_list=[],
                basis=symm.basis,
                check_symm=False,
                check_herm=False,
                check_pcon=False,
                dtype=get_default_dtype(),
            )
        return self._quspin_op[symm]

    def todense(self, symm: Optional[Symmetry] = None) -> np.ndarray:
        """
        Obtain the dense matrix representing the operator

        :param symm:
            The symmetry used for generate the operator basis, by default the basis
            without symmetry
        """
        quspin_op = self.get_quspin_op(symm)
        return quspin_op.as_dense_format()

    def __array__(self) -> np.ndarray:
        return self.todense()

    def __jax_array__(self) -> jax.Array:
        return jnp.asarray(self.todense())

    def __matmul__(self, state: State) -> DenseState:
        r"""
        Apply the operator on a ket state by ``H @ state`` to get :math:`H \left| \psi \right>`.
        The exact expectation value :math:`\left<\psi|H|\psi \right>` can be computed by
        ``state @ H @ state``.
        """
        quspin_op = self.get_quspin_op(state.symm)
        wf = state.todense().wave_function
        if isinstance(wf, jax.Array):
            wf = to_replicate_numpy(wf)
        wf = quspin_op.dot(np.asarray(wf, order="C"))
        return DenseState(wf, state.symm)

    def __rmatmul__(self, state: State) -> DenseState:
        r"""
        Apply the operator on a bra state by ``state @ H`` to get :math:`\left< \psi \right| H`.
        The exact expectation value :math:`\left<\psi|H|\psi \right>` can be computed by
        ``state @ H @ state``.
        """
        return self.__matmul__(state)

    def diagonalize(
        self,
        symm: Optional[Symmetry] = None,
        k: Union[int, str] = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalize the hamiltonian :math:`H = V D V^†`

        :param symm:
            Symmetry for generating basis.
        :param k:
            A number specifying how many lowest states to obtain.
        :return:
            w:
                Array of k eigenvalues.

            v:
                An array of k eigenvectors. ``v[:, i]`` is the eigenvector corresponding to
                the eigenvalue ``w[i]``.
        """
        quspin_op = self.get_quspin_op(symm)
        return quspin_op.eigsh(k=k, which="SA")

    def __add__(self, other: Union[Number, Operator]) -> Operator:
        """Add two operators"""
        if isinstance(other, Number):
            if not np.isclose(other, 0.0):
                raise ValueError("Constant shift is not implemented for Operator.")
            return self

        elif isinstance(other, Operator):
            op_list = self.op_list.copy()
            opstr1 = tuple(op for op, _ in op_list)
            for opstr2, interaction in other.op_list:
                try:
                    index = opstr1.index(opstr2)
                    op_list[index][1] += interaction
                except ValueError:
                    op_list.append([opstr2, interaction])
            return Operator(op_list)

        return NotImplemented

    def __radd__(self, other: Number) -> Operator:
        if isinstance(other, Number):
            return self + other
        return NotImplemented

    def __iadd__(self, other: Operator) -> Operator:
        return self + other

    def __sub__(self, other: Union[Number, Operator]) -> Operator:
        if isinstance(other, Number):
            if not np.isclose(other, 0.0):
                raise ValueError("Constant shift is not implemented for Operator.")
            return self
        if isinstance(other, Operator):
            return self + (-other)
        return NotImplemented

    def __rsub__(self, other: Number) -> Operator:
        if isinstance(other, Number):
            if not np.isclose(other, 0.0):
                raise ValueError("Constant shift is not implemented for Operator.")
            return -self
        return NotImplemented

    def __isub__(self, other: Union[Number, Operator]) -> Operator:
        return self - other

    def __mul__(self, other: Union[Number, Operator]) -> Operator:
        if isinstance(other, Number):
            op_list = self.op_list.copy()
            for opstr, interaction in op_list:
                for term in interaction:
                    term[0] *= other
            return Operator(op_list)

        elif isinstance(other, Operator):
            op_list = []
            for opstr1, interaction1 in self.op_list:
                for opstr2, interaction2 in other.op_list:
                    op = [opstr1 + opstr2, []]
                    for J1, *index1 in interaction1:
                        for J2, *index2 in interaction2:
                            op[1].append([J1 * J2, *index1, *index2])
                    op_list.append(op)
            return Operator(op_list)

        return NotImplemented

    def __rmul__(self, other: Number) -> Operator:
        if isinstance(other, Number):
            return self * other
        return NotImplemented

    def __imul__(self, other: Union[Number, Operator]) -> Operator:
        if isinstance(other, Number):
            for opstr, interaction in self.op_list:
                for term in interaction:
                    term[0] *= other
            return self

    def __neg__(self) -> Operator:
        return (-1) * self

    def __truediv__(self, other: Number) -> Operator:
        if isinstance(other, Number):
            return self * (1 / other)
        return NotImplemented

    def __itruediv__(self, other: Number) -> Operator:
        if isinstance(other, Number):
            return self.__imul__(1 / other)
        return NotImplemented

    def psiOloc(
        self, state: State, samples: Union[Samples, np.ndarray, jax.Array]
    ) -> jax.Array:
        if isinstance(samples, Samples):
            x = samples.spins
            wf = samples.wave_function
        else:
            x = to_global_array(samples)
            wf = state(x)
        internal = state.init_internal(x)

        Hz = _apply_diag(x, self.jax_op_list)
        off_diags = _apply_off_diag(x, self.jax_op_list)
        psiHx = jnp.zeros(x.shape[0], Hz.dtype, device=get_global_sharding())

        for nflips, (x_conn, H_conn) in off_diags.items():
            x_conn = np.asarray(global_to_local(x_conn))
            H_conn = np.asarray(global_to_local(H_conn))
            valid = ~np.isnan(H_conn)
            idx_start = x_conn.shape[0] * jax.process_index()
            segment = np.arange(idx_start, idx_start + x_conn.shape[0])
            segment = np.repeat(segment[:, None], x_conn.shape[1], axis=1)
            segment = segment[valid]
            H_conn = H_conn[valid]
            x_conn = x_conn[valid, :]

            n_conn = np.max(process_allgather(x_conn.shape[0])).item()
            if hasattr(state, "forward_chunk") and state.forward_chunk is not None:
                chunk_size = state.forward_chunk * jax.local_device_count()
                n_res = n_conn % chunk_size
                n_conn_extend = n_conn + chunk_size - n_res if n_res > 0 else n_conn
            else:
                n_conn_extend = n_conn
            pad_width = (0, n_conn_extend - x_conn.shape[0])
            segment = local_to_global(np.pad(segment, pad_width))
            H_conn = local_to_global(np.pad(H_conn, pad_width))
            x_conn = local_to_global(np.pad(x_conn, (pad_width, (0, 0))))

            psi_conn = state.ref_forward(
                x_conn, x, nflips, segment, internal, update_maximum=True
            )
            psiHx += segment_sum(psi_conn * H_conn, segment, num_segments=x.shape[0])

        return Hz * wf + psiHx

    def Oloc(
        self, state: State, samples: Union[Samples, np.ndarray, jax.Array]
    ) -> jax.Array:
        r"""
        Computes the local operator
        :math:`O_\mathrm{loc}(s) = \sum_{s'} \frac{\psi_{s'}}{\psi_s} \left< s|O|s' \right>`

        :param state:
            A `quantax.state.State` for computing :math:`\psi`

        :param samples:
            A batch of samples :math:`s`

        :return:
            A 1D jax array :math:`O_\mathrm{loc}(s)`
        """
        if not isinstance(samples, Samples):
            spins = to_global_array(samples)
            wf = state(spins)
            samples = Samples(spins, wf)

        return self.psiOloc(state, samples) / samples.wave_function

    def expectation(
        self,
        state: State,
        samples: Union[Samples, np.ndarray, jax.Array],
        return_var: bool = False,
    ) -> Union[float, Tuple[float, float]]:
        r"""
        The expectation value of the operator

        :param state:
            The state for computing :math:`\psi` in :math:`O_\mathrm{loc}`

        :param samples:
            The samples for estimating :math:`\left< O_\mathrm{loc} \right>`

        :param return_var:
            Whether the variance should also be returned, default to False

        :return:
            Omean:
                Mean value of the operator :math:`\left< O_\mathrm{loc} \right>`

            Ovar:
                Variance of the operator
                :math:`\left< |O_\mathrm{loc}|^2 \right> - |\left< O_\mathrm{loc} \right>|^2`,
                only returned when ``return_var = True``
        """
        reweight = samples.reweight_factor if isinstance(samples, Samples) else 1.0
        Oloc = self.Oloc(state, samples)
        Omean = jnp.mean(Oloc * reweight)
        if return_var:
            Ovar = jnp.mean(jnp.abs(Oloc) ** 2 * reweight) - jnp.abs(Omean) ** 2
            return Omean.real.item(), Ovar.real.item()
        else:
            return Omean.real.item()
