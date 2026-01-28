from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple, Union, Callable, Any
from numbers import Number
import copy
from functools import partial
from warnings import warn
import numpy as np
import scipy
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import equinox as eqx
import scipy.linalg

from ..state import State, DenseState
from ..sampler import Samples
from ..symmetry import Symmetry, Identity
from ..utils import (
    to_distribute_array,
    to_replicate_numpy,
    array_extend,
    chunk_map,
    PsiArray,
)
from ..global_defs import PARTICLE_TYPE, get_sites, get_default_dtype

if TYPE_CHECKING:
    from quspin.operators import hamiltonian


def _apply_site_operator(
    x: jax.Array, opstr: str, J: jax.Array, idx: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    sites = get_sites()
    particle_type = sites.particle_type
    is_fermion = sites.is_fermion
    double_occ = sites.double_occ

    # diagonal
    if opstr == "I":
        return x, J

    if opstr == "n":
        return x, jnp.where(x[idx] > 0, J, 0)

    if opstr == "z":
        return x, J * x[idx] / 2

    # off-diagonal
    if is_fermion:
        # counting from last to first according to quspin convention
        num_fermion = jnp.cumulative_sum(x[::-1] > 0, include_initial=True)[-2::-1]
        J = jnp.where(num_fermion[idx] % 2 == 0, J, -J)
    elif opstr in ("x", "y"):
        J /= 2

    if opstr == "+":
        J = jnp.where(x[idx] < 0, J, jnp.nan)
        x = x.at[idx].set(1)

    if opstr == "-":
        J = jnp.where(x[idx] > 0, J, jnp.nan)
        x = x.at[idx].set(-1)

    if opstr == "x":
        x = x.at[idx].mul(-1)

    if opstr == "y":
        J *= 1j * x[idx]
        if is_fermion:
            J *= -1  # conventional sign difference
        x = x.at[idx].mul(-1)

    if particle_type == PARTICLE_TYPE.spinful_fermion and not double_occ:
        N = x.size // 2
        idx = jnp.where(idx < N, idx, idx - N)
        J = jnp.where(jnp.any(x.reshape(2, N)[:, idx] <= 0), J, jnp.nan)

    return x, J


@eqx.filter_jit
@partial(jax.vmap, in_axes=(0, None))
def _apply_diag(s: jax.Array, jax_op_list: list) -> jax.Array:
    Hz = 0
    apply_fn = jax.vmap(_apply_site_operator, in_axes=(None, None, 0, 0))

    for opstr, update_mode, J_array, index_array in jax_op_list:
        if all(op in ("I", "n", "z") for op in opstr):
            for op, idx in zip(opstr, index_array.T):
                _, J_array = apply_fn(s, op, J_array, idx)
            Hz += jnp.sum(J_array)

    return Hz


@eqx.filter_jit
@partial(eqx.filter_vmap, in_axes=(0, None))
def _apply_off_diag(
    s: jax.Array, jax_op_list: list
) -> list[tuple[dict[str, Any], jax.Array, jax.Array]]:
    out = []
    apply_fn = jax.vmap(_apply_site_operator, in_axes=(0, None, 0, 0))

    for opstr, update_mode, J_array, index_array in jax_op_list:
        if any(op not in ("I", "n", "z") for op in opstr):
            s_conn = jnp.repeat(s[None, :], J_array.size, axis=0)
            index_array = index_array.astype(jnp.int32)
            for op, idx in zip(reversed(opstr), reversed(index_array.T)):
                s_conn, J_array = apply_fn(s_conn, op, J_array, idx)
            out.append((update_mode, s_conn, J_array))

    return out


@partial(jax.jit, static_argnums=1)
def _get_conn_size(H_conn: jax.Array, forward_chunk: Optional[int]) -> jax.Array:
    ndevices = jax.device_count()
    ns, nconn = H_conn.shape

    if forward_chunk is None:
        H_conn = H_conn.reshape(ndevices, -1, 1, nconn)
    else:
        H_conn = H_conn.reshape(ndevices, -1, nconn)
        H_conn = array_extend(H_conn, forward_chunk, axis=1, padding_values=jnp.nan)
        H_conn = H_conn.reshape(ndevices, forward_chunk, -1, nconn)

    size = jnp.sum(~jnp.isnan(H_conn), axis=(1, 3))
    size = jnp.max(size)

    if forward_chunk is not None:
        conn_chunks = (size - 1) // forward_chunk + 1
        size = conn_chunks * forward_chunk

    return size


@partial(jax.jit, static_argnums=2)
def _get_conn(
    s_conn: jax.Array, H_conn: jax.Array, conn_size: int
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    ndevices = jax.device_count()
    nsamples, nconn, Nmodes = s_conn.shape
    H_conn = H_conn.reshape(ndevices, -1, nconn)
    s_conn = s_conn.reshape(ndevices, -1, nconn, Nmodes)

    def device_conn(s_conn, H_conn):
        is_valid = ~(jnp.isnan(H_conn) | jnp.isclose(H_conn, 0))
        segment, conn_idx = jnp.nonzero(is_valid, size=conn_size, fill_value=-1)
        s_conn = s_conn[segment, conn_idx]
        H_conn = H_conn[segment, conn_idx]
        H_conn = jnp.where(segment == -1, 0, H_conn)
        return segment, s_conn, H_conn

    segment, s_conn, H_conn = jax.vmap(device_conn)(s_conn, H_conn)
    segment = segment.flatten()
    s_conn = s_conn.reshape(-1, Nmodes)
    H_conn = H_conn.flatten()
    return segment, s_conn, H_conn


@jax.jit
def _get_Olocx(
    psi: PsiArray, segment: jax.Array, psi_conn: PsiArray, H_conn: jax.Array
) -> jax.Array:
    ndevices = jax.device_count()
    psi = psi.reshape(ndevices, -1)
    psi_conn = psi_conn.reshape(ndevices, -1)
    H_conn = H_conn.reshape(ndevices, -1)
    segment = segment.reshape(ndevices, -1)
    num_seg = psi.shape[1]

    fn_ratio = lambda psi, psi_conn, segment: jnp.asarray(psi_conn / psi[segment])
    psi_ratio = jax.vmap(fn_ratio)(psi, psi_conn, segment)

    segment_sum = lambda data, segment: jax.ops.segment_sum(data, segment, num_seg)
    Olocx = jax.vmap(segment_sum)(psi_ratio * H_conn, segment)
    return Olocx.flatten()


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
        Operator list with jax arrays, made easy for applying operator to basis states.

        The format is ``[[opstr1, update_mode1, J_array1, index_array1], [opstr2, update_mode2, J_array2, index_array2], ...]``
        """
        if self._jax_op_list is None:
            update_mode_filter = lambda opstr, indices: {}
            self.apply_update_mode_filter(update_mode_filter)

        return self._jax_op_list

    @property
    def expression(self) -> str:
        """The operator as a human-readable expression"""
        is_fermion = get_sites().is_fermion
        SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        p = "c†" if is_fermion else "S⁺"
        m = "c" if is_fermion else "S⁻"
        OP = str.maketrans({"x": "Sˣ", "y": "Sʸ", "z": "Sᶻ", "+": p, "-": m})
        expression = []
        for opstr, interaction in self.op_list:
            for J, *index in interaction:
                expression.append(f"{J:+}")
                for op, i in zip(opstr, index):
                    expression.append(f"{op.translate(OP)}{str(i).translate(SUB)}")
        return " ".join(expression)

    def __repr__(self) -> str:
        return self.expression

    def apply_update_mode_filter(
        self, update_mode_filter: Callable[[str, jax.Array], dict[str, Any]]
    ) -> None:
        """
        Apply a filter function to update the operator's jax_op_list with additional
        update_mode information for each operator term.

        :param update_mode_filter:
            A callable that takes an operator string and its corresponding site indices,
            and returns a dictionary of update_mode to be added to each term.
        """
        self._jax_op_list = []

        for opstr, interaction in self.op_list:
            values_dict = {}
            for J, *indices in interaction:
                update_mode = update_mode_filter(opstr, indices)
                values = tuple(update_mode.values())
                if values in values_dict:
                    values_dict[values][0].append(J)
                    values_dict[values][1].append(indices)
                else:
                    values_dict[values] = [[J], [indices]]

            keys = update_mode.keys()
            for values, (J_array, index_array) in values_dict.items():
                J_array = jnp.asarray(J_array, dtype=get_default_dtype())
                index_array = jnp.asarray(index_array, dtype=jnp.uint16)
                update_mode = dict(zip(keys, values))
                self._jax_op_list.append([opstr, update_mode, J_array, index_array])

    def get_quspin_op(self, symm: Optional[Symmetry] = None) -> hamiltonian:
        """
        Obtain the corresponding
        `QuSpin operator <https://quspin.github.io/QuSpin/generated/quspin.operators.hamiltonian.html#quspin.operators.hamiltonian.__init__>`_

        :param symm:
            The symmetry used for generate the operator basis, by default the basis
            without symmetry
        """
        from quspin.operators import hamiltonian

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
        return quspin_op.toarray()

    def __matmul__(self, other: Union[State, Operator]) -> DenseState:
        r"""
        Apply the operator on a ket state by ``H @ state`` to get :math:`H \left| \psi \right>`,
        or multiply two operators by ``H1 @ H2``.
        The exact expectation value :math:`\left<\psi|H|\psi \right>` can be computed by
        ``state @ H @ state``.
        """
        if isinstance(other, Operator):
            op_list = []
            for opstr1, interaction1 in self.op_list:
                for opstr2, interaction2 in other.op_list:
                    op = [opstr1 + opstr2, []]
                    for J1, *index1 in interaction1:
                        for J2, *index2 in interaction2:
                            op[1].append([J1 * J2, *index1, *index2])
                    op_list.append(op)
            return Operator(op_list)
        elif isinstance(other, State):
            quspin_op = self.get_quspin_op(other.symm)
            psi = other.todense().psi
            if isinstance(psi, jax.Array):
                psi = to_replicate_numpy(psi)
            psi = quspin_op.dot(np.asarray(psi, order="C"))
            return DenseState(psi, other.symm)

        return NotImplemented

    def __rmatmul__(self, state: State) -> DenseState:
        r"""
        Apply the operator on a bra state by ``state @ H`` to get :math:`\left< \psi \right| H`.
        The exact expectation value :math:`\left<\psi|H|\psi \right>` can be computed by
        ``state @ H @ state``.
        """
        if isinstance(state, State):
            return self.__matmul__(state)
        return NotImplemented

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
            A number specifying how many lowest states to obtain, or "full" for
            all eigenstates.
        :return:
            w:
                Array of k eigenvalues.

            v:
                An array of k eigenvectors. ``v[:, i]`` is the eigenvector corresponding to
                the eigenvalue ``w[i]``.
        """
        if isinstance(k, int):
            quspin_op = self.get_quspin_op(symm)
            return quspin_op.eigsh(k=k, which="SA")
        elif k == "full":
            array = self.todense(symm)
            return scipy.linalg.eigh(array)
        else:
            raise ValueError("Invalid value of `k`.")

    @property
    def H(self) -> Operator:
        """Hermitian conjugate"""
        op_list = copy.deepcopy(self.op_list)

        for i, (opstr, interaction) in enumerate(op_list):
            trans = str.maketrans("+-", "-+")
            opstr = opstr.translate(trans)[::-1]
            op_list[i][0] = opstr

            for term in interaction:
                term[0] = term[0].conjugate()
                term[1:] = term[-1:0:-1]

        return Operator(op_list)

    def __add__(self, other: Union[Number, Operator]) -> Operator:
        """Add two operators."""
        if isinstance(other, Number):
            if not np.isclose(other, 0.0):
                raise ValueError("Constant shift is not implemented for Operator.")
            return self

        elif isinstance(other, Operator):
            op_list = copy.deepcopy(self.op_list)
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
        """In-place addition of two operators."""
        if isinstance(other, Number):
            if not np.isclose(other, 0.0):
                raise ValueError("Constant shift is not implemented for Operator.")
            return self

        elif isinstance(other, Operator):
            op_list = self.op_list
            opstr1 = tuple(op for op, _ in op_list)
            for opstr2, interaction in other.op_list:
                try:
                    index = opstr1.index(opstr2)
                    op_list[index][1] += interaction
                except ValueError:
                    op_list.append([opstr2, interaction])
            return Operator(op_list)

        return NotImplemented

    def __sub__(self, other: Union[Number, Operator]) -> Operator:
        """Subtract two operators."""
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
        """In-place subtraction of two operators."""
        self += -other
        return self

    def __mul__(self, other: ArrayLike) -> Operator:
        """Multiply an operator with a scalar."""
        if eqx.is_array_like(other):
            if eqx.is_array(other):
                other = other.item()
            op_list = copy.deepcopy(self.op_list)
            for opstr, interaction in op_list:
                for term in interaction:
                    term[0] *= other
            return Operator(op_list)

        return NotImplemented

    def __rmul__(self, other: ArrayLike) -> Operator:
        """Multiply an operator with a scalar."""
        if eqx.is_array_like(other):
            return self * other
        return NotImplemented

    def __imul__(self, other: ArrayLike) -> Operator:
        """In-place multiplication of an operator with a scalar."""
        if eqx.is_array_like(other):
            if eqx.is_array(other):
                other = other.item()

            for opstr, interaction in self.op_list:
                for term in interaction:
                    term[0] *= other
            return self

    def __neg__(self) -> Operator:
        """Negate an operator."""
        return (-1) * self

    def __truediv__(self, other: Number) -> Operator:
        """Divide an operator by a scalar."""
        if isinstance(other, Number):
            return self * (1 / other)
        return NotImplemented

    def __itruediv__(self, other: Number) -> Operator:
        """In-place division of an operator by a scalar."""
        if isinstance(other, Number):
            return self.__imul__(1 / other)
        return NotImplemented

    def apply_diag(self, s: jax.Array) -> jax.Array:
        return _apply_diag(s, self.jax_op_list)

    def apply_off_diag(
        self, s: jax.Array
    ) -> list[tuple[dict[str, Any], jax.Array, jax.Array]]:
        return _apply_off_diag(s, self.jax_op_list)

    def _update_connectivity(
        self, off_diag: list[tuple[dict[str, Any], jax.Array, jax.Array]]
    ) -> None:
        """
        Record the average number of s' for input s. This connectivity value can help to
        improve efficiency by adjusting `max_parallel` in `quantax.state.Variational`.
        The value is recorded for each update_mode and device.
        """
        ndevices = jax.device_count()
        connectivity = []
        for update_mode, s_conn, H_conn in off_diag:
            H_conn = H_conn.reshape(ndevices, -1, H_conn.shape[-1])
            n_conn = jnp.sum(~jnp.isnan(H_conn), axis=(1, 2)) / H_conn.shape[1]
            connectivity.append((update_mode, n_conn))
        self._connectivity = connectivity

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
        forward_chunk = getattr(state, "forward_chunk", None)
        ref_chunk = getattr(state, "ref_chunk", None)
        if (
            forward_chunk is not None
            and ref_chunk is not None
            and forward_chunk < ref_chunk
        ):
            raise ValueError("Unsupported chunk size: forward_chunk < ref_chunk.")

        if isinstance(samples, Samples):
            s = samples.spins
            psi = samples.psi
            internal = samples.state_internal
        else:
            s = to_distribute_array(samples)
            psi = state(s)
            internal = None

        Oloc = self.apply_diag(s)
        off_diags = self.apply_off_diag(s)
        self._update_connectivity(off_diags)

        for update_mode, s_conn, H_conn in off_diags:
            use_ref = state.use_ref
            if use_ref:
                mode_keys = update_mode.keys()
                if not all(mode in mode_keys for mode in state.required_update_modes):
                    warn(
                        "The update_modes required by the state are not all provided "
                        "in the operator. The fast local updates are not utilized."
                    )
                    use_ref = False

            chunk_size = ref_chunk if use_ref else forward_chunk
            conn_size = _get_conn_size(H_conn, forward_chunk).item()

            def get_Oloc_terms(s, psi, s_conn, H_conn, internal):
                segment, s_conn, H_conn = _get_conn(s_conn, H_conn, conn_size)
                if use_ref:
                    if internal is None:
                        internal = state.init_internal(s)
                    psi_conn = state.ref_forward(
                        s_conn, s, update_mode, segment, internal
                    )
                else:
                    psi_conn = state(s_conn)
                return _get_Olocx(psi, segment, psi_conn, H_conn)

            in_axes = (0, 0, 0, 0, None) if internal is None else 0
            get_Oloc_terms = chunk_map(get_Oloc_terms, in_axes, chunk_size=chunk_size)
            Oloc += get_Oloc_terms(s, psi, s_conn, H_conn, internal)

        return Oloc

    def expectation(
        self, state: State, samples: Union[Samples, PsiArray], return_var: bool = False
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
