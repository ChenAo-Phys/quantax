from __future__ import annotations
from typing import Optional, Tuple, Union
from numbers import Number
import numpy as np
from scipy.linalg import eigh
import jax
import jax.numpy as jnp
from jax.ops import segment_sum
from quspin.operators import quantum_LinearOperator

from ..state import State, DenseState
from ..sampler import Samples
from ..symmetry import Symmetry, Identity
from ..utils import array_to_ints, ints_to_array, to_array_shard
from ..global_defs import get_default_dtype


class Operator:
    def __init__(self, op_list: list):
        """
        Constructs an operator.
        """
        self._op_list = op_list
        self._quspin_op = dict()
        self._connectivity = None

    @property
    def op_list(self) -> list:
        return self._op_list
    
    @property
    def expression(self) -> str:
        SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        OP = str.maketrans({"x": "σˣ", "y": "σʸ", "z": "σᶻ", "+": "σ⁺", "-": "σ⁻"})
        expression = []
        for opstr, interaction in self.op_list:
            for J, *index in interaction:
                expression.append(f"{J:+}")
                for op, i in zip(opstr, index):
                    expression.append(f"{op.translate(OP)}{str(i).translate(SUB)}")
        return " ".join(expression)
    
    def __repr__(self) -> str:
        return self.expression

    def get_quspin_op(self, symm: Optional[Symmetry] = None) -> quantum_LinearOperator:
        if symm is None:
            symm = Identity()
        symm.basis.make()
        if symm not in self._quspin_op:
            self._quspin_op[symm] = quantum_LinearOperator(
                self.op_list, basis=symm.basis, dtype=get_default_dtype()
            )
        return self._quspin_op[symm]

    def todense(self, symm: Optional[Symmetry] = None) -> np.ndarray:
        quspin_op = self.get_quspin_op(symm)
        op = np.eye(quspin_op.shape[1], dtype=quspin_op.dtype)
        op = quspin_op.matmat(op)
        return op

    def __array__(self) -> np.ndarray:
        return self.todense()

    def __jax_array__(self) -> jax.Array:
        return jnp.asarray(self.todense())

    def __matmul__(self, state: State) -> DenseState:
        quspin_op = self.get_quspin_op(state.symm)
        wf = state.todense().wave_function
        wf = quspin_op.matvec(wf)
        return DenseState(wf, state.symm)

    def __rmatmul__(self, state: State) -> DenseState:
        return self.__matmul__(state)

    def diagonalize(
        self,
        symm: Optional[Symmetry] = None,
        k: Union[int, str] = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters:
            symm: Symmetry for generating basis
            k: Either a number specifying how many lowest states to obtain, or a string
                "full" meaning the full spectrum.
        Returns:
            w: ndarray
            Array of k eigenvalues.

            v: ndarray
            An array of k eigenvectors. v[:, i] is the eigenvector corresponding to
            the eigenvalue w[i].

        A = V D V*
        """

        if k == "full":
            dense = self.todense(symm)
            return eigh(dense)
        else:
            quspin_op = self.get_quspin_op(symm)
            return quspin_op.eigsh(k=k, which="SA")

    def __add__(self, other: Union[Number, Operator]) -> Operator:
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

    def apply_diag(self, fock_states: Union[np.ndarray, jax.Array]) -> np.ndarray:
        basis = Identity().basis
        basis_ints = array_to_ints(fock_states)
        dtype = get_default_dtype()
        Hz = np.zeros(basis_ints.size, dtype)

        for opstr, interaction in self.op_list:
            if all(s in ("I", "z") for s in opstr):
                for J, *index in interaction:
                    ME, bra, ket = basis.Op_bra_ket(
                        opstr, index, J, dtype, basis_ints, reduce_output=False
                    )
                    Hz += ME
        return Hz

    def apply_off_diag(
        self, fock_states: Union[np.ndarray, jax.Array], return_basis_ints: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply non-zero off-diagonal elements to input spin configurations.
        Return:
            segment, s_conn (ints_conn), H_conn
        """
        basis = Identity().basis
        basis_ints = array_to_ints(fock_states)
        dtype = get_default_dtype()
        arange = np.arange(basis_ints.size, dtype=np.uint32)
        segment = []
        s_conn = []
        H_conn = []
        
        for opstr, interaction in self.op_list:
            if all(s in ("I", "z") for s in opstr):
                continue
            for J, *index in interaction:
                ME, bra, ket = basis.Op_bra_ket(
                    opstr, index, J, dtype, basis_ints, reduce_output=False
                )
                is_nonzero = ~np.isclose(ME, 0.)
                segment.append(arange[is_nonzero])
                s_conn.append(bra[is_nonzero])
                H_conn.append(ME[is_nonzero])

        segment = np.concatenate(segment)
        s_conn = np.concatenate(s_conn)
        if not return_basis_ints:
            s_conn = ints_to_array(s_conn)
        H_conn = np.concatenate(H_conn)
        return segment, s_conn, H_conn

    def psiOloc(
        self, state: State, samples: Union[Samples, np.ndarray, jax.Array]
    ) -> jax.Array:
        if isinstance(samples, Samples):
            spins = np.asarray(samples.spins)
            wf = samples.wave_function
        else:
            spins = np.asarray(samples)
            wf = state(samples)
        
        Hz = to_array_shard(self.apply_diag(spins))

        segment, s_conn, H_conn = self.apply_off_diag(spins)
        n_conn = s_conn.shape[0]
        self._connectivity = n_conn / spins.shape[0]
        has_mp = hasattr(state, "max_parallel") and state.max_parallel is not None
        if has_mp and n_conn > 0:
            max_parallel = state.max_parallel * jax.local_device_count() // state.nsymm
            n_res = n_conn % max_parallel
            pad_width = (0, max_parallel - n_res)
            segment = np.pad(segment, pad_width)
            H_conn = np.pad(H_conn, pad_width)
            s_conn = np.pad(s_conn, (pad_width, (0, 0)), constant_values=1)
        
        psi_conn = state(s_conn)
        Hx = segment_sum(psi_conn * H_conn, segment, num_segments=spins.shape[0])
        Hx = to_array_shard(Hx)
        return Hz * wf + Hx

    def Oloc(
        self, state: State, samples: Union[Samples, np.ndarray, jax.Array]
    ) -> jax.Array:
        if not isinstance(samples, Samples):
            wf = state(samples)
            samples = Samples(samples, wf)
        else:
            wf = samples.wave_function
        return self.psiOloc(state, samples) / wf

    def expectation(
        self,
        state: State,
        samples: Union[Samples, np.ndarray, jax.Array],
        return_var: bool = False,
    ) -> Union[float, Tuple[float, float]]:
        reweight = samples.reweight_factor if isinstance(samples, Samples) else 1.0
        Oloc = self.Oloc(state, samples)
        Omean = jnp.mean(Oloc * reweight)
        if return_var:
            Ovar = jnp.mean(jnp.abs(Oloc) ** 2 * reweight)
            return Omean.real.item(), Ovar.real.item()
        else:
            return Omean.real.item()
