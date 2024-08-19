from __future__ import annotations
from typing import Optional, Tuple, Union
from numbers import Number
import numpy as np
from scipy.linalg import eigh
import jax
import jax.numpy as jnp
from jax.ops import segment_sum
from quspin.operators import hamiltonian

from ..state import State, DenseState
from ..sampler import Samples
from ..symmetry import Symmetry, Identity
from ..utils import array_to_ints, ints_to_array, to_array_shard
from ..global_defs import get_default_dtype


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
        self._quspin_op = dict()
        self._connectivity = None

    @property
    def op_list(self) -> list:
        """Operator represented as a list in the QuSpin format"""
        return self._op_list

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
        wf = quspin_op.dot(wf)
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

        :param symm: Symmetry for generating basis
        :param k: 
            Either a number specifying how many lowest states to obtain, or a string
            "full" meaning the full spectrum.
        :return:
            w:
                Array of k eigenvalues.

            v:
                An array of k eigenvectors. ``v[:, i]`` is the eigenvector corresponding to
                the eigenvalue ``w[i]``.
        """

        if k == "full":
            dense = self.todense(symm)
            return eigh(dense)
        else:
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

    def apply_diag(self, fock_states: Union[np.ndarray, jax.Array]) -> np.ndarray:
        r"""
        Apply the diagonal elements of the operator

        :param fock_states:
            A batch of fock states :math:`\left| x_i \right>` with elements 
            :math:`\pm 1` that the operator acts on
        
        :return:
            A 1D array :math:`H_z` with :math:`H_{z,i} = \left< x_i | H | x_i \right>`
        """
        basis = Identity().basis
        basis_ints = array_to_ints(fock_states)
        dtype = get_default_dtype()
        Hz = np.zeros(basis_ints.size, dtype)

        for opstr, interaction in self.op_list:
            if all(s in ("I", "n", "z") for s in opstr):
                for J, *index in interaction:
                    ME, bra, ket = basis.Op_bra_ket(
                        opstr, index, J, dtype, basis_ints, reduce_output=False
                    )
                    Hz += ME
        return Hz

    def apply_off_diag(
        self, fock_states: Union[np.ndarray, jax.Array], return_basis_ints: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Apply the non-zero non-diagonal elements of the operator. Every input fock state
        :math:`\left| x_i \right>` is mapped to multiple output fock states 
        :math:`\left| x'_j \right>`.

        :param fock_states:
            A 2D array as a batch of input fock states :math:`\left| x_i \right>` with 
            elements :math:`\pm 1`

        :param return_basis_ints:
            Whether to return the connected states :math:`\left| x'_j \right>` as basis 
            integers or fock states, default to fock states.
        
        :return:
            segment:
                A 1D array showing how :math:`\left| x'_j \right>` is connected to 
                :math:`\left| x_i \right>`. The j'th element of this array is i.

            s_conn:
                An array as a batch of connected states :math:`\left| x'_j \right>`

            H_conn:
                A 1D array :math:`H_c` with :math:`H_{c,j} = \left< x'_j | H | x_i \right>`
        """
        basis = Identity().basis
        basis_ints = array_to_ints(fock_states)
        dtype = get_default_dtype()
        arange = np.arange(basis_ints.size, dtype=np.uint32)
        segment = []
        s_conn = []
        H_conn = []

        for opstr, interaction in self.op_list:
            if not all(s in ("I", "n", "z") for s in opstr):
                for J, *index in interaction:
                    ME, bra, ket = basis.Op_bra_ket(
                        opstr, index, J, dtype, basis_ints, reduce_output=False
                    )
                    is_nonzero = ~np.isclose(ME, 0.0)
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
