from __future__ import annotations
from typing import TYPE_CHECKING, Sequence, Callable, Any, overload, Literal
from dataclasses import dataclass
from numpy.typing import NDArray
from jax.typing import ArrayLike
from jaxtyping import PyTree
import copy
from functools import partial
from warnings import warn
import numpy as np
import scipy
import jax
import jax.numpy as jnp
import equinox as eqx
import scipy.linalg
from .update_mode_filters import (
    none_filter,
    nflips_filter,
    nflips_up_dn_filter,
    DIAGONAL_OPS,
)
from ..state import State, DenseState
from ..sampler import Samples
from ..symmetry import Symmetry, Identity
from ..utils import (
    get_replicated_sharding,
    to_distributed_array,
    to_replicated_numpy,
    chunk_map,
    PsiArray,
)
from ..global_defs import PARTICLE_TYPE, get_sites, get_default_dtype

if TYPE_CHECKING:
    import quspin.operators


def _apply_site_operator(
    s: jax.Array, opstr: str, J: jax.Array, idx: jax.Array
) -> tuple[jax.Array, jax.Array]:
    sites = get_sites()
    particle_type = sites.particle_type
    is_fermion = sites.is_fermion
    double_occ = sites.double_occ

    # diagonal
    if opstr == "I":
        return s, J

    if opstr == "n":
        return s, jnp.where(s[idx] > 0, J, 0)

    if opstr == "z":
        return s, J * s[idx] / 2

    # off-diagonal
    if is_fermion:
        # counting from last to first according to quspin convention
        num_fermion = jnp.cumulative_sum(s[::-1] > 0, include_initial=True)[-2::-1]
        J = jnp.where(num_fermion[idx] % 2 == 0, J, -J)
    elif opstr in ("x", "y"):
        J /= 2

    if opstr == "+":
        J = jnp.where(s[idx] < 0, J, jnp.nan)
        s = s.at[idx].set(1)

    if opstr == "-":
        J = jnp.where(s[idx] > 0, J, jnp.nan)
        s = s.at[idx].set(-1)

    if opstr == "x":
        s = s.at[idx].mul(-1)

    if opstr == "y":
        J *= 1j * s[idx]
        if is_fermion:
            J *= -1  # conventional sign difference
        s = s.at[idx].mul(-1)

    if particle_type == PARTICLE_TYPE.spinful_fermion and not double_occ:
        N = s.size // 2
        idx = jnp.where(idx < N, idx, idx - N)
        J = jnp.where(jnp.any(s.reshape(2, N)[:, idx] <= 0), J, jnp.nan)

    return s, J


@eqx.filter_jit
@partial(eqx.filter_vmap, in_axes=(0, None))
def _apply_diag(
    s: jax.Array, jax_op_list: list[tuple[dict[str, Any], tuple[OpTermJAX, ...]]]
) -> jax.Array:
    Hz = jnp.array(0.0, get_default_dtype())
    apply_fn = jax.vmap(_apply_site_operator, in_axes=(None, None, 0, 0))

    for update_mode, op_terms in jax_op_list:
        for op_term in op_terms:
            if all(op in DIAGONAL_OPS for op in op_term.opstr):
                strength = op_term.strength
                for op, idx in zip(op_term.opstr, op_term.indices.T):
                    _, strength = apply_fn(s, op, strength, idx)
                Hz += jnp.sum(strength)

    return Hz


@eqx.filter_jit
@partial(eqx.filter_vmap, in_axes=(0, None))
def _apply_off_diag(
    s: jax.Array, jax_op_list: list[tuple[dict[str, Any], tuple[OpTermJAX, ...]]]
) -> list[tuple[dict[str, Any], jax.Array, jax.Array]]:
    out = []
    apply_fn = jax.vmap(_apply_site_operator, in_axes=(0, None, 0, 0))

    for update_mode, op_terms in jax_op_list:
        s_conn_list = []
        strength_list = []
        for op_term in op_terms:
            if any(op not in DIAGONAL_OPS for op in op_term.opstr):
                strength = op_term.strength
                s_conn = jnp.repeat(s[None, :], strength.size, axis=0)
                for op, idx in zip(reversed(op_term.opstr), op_term.indices.T[::-1]):
                    s_conn, strength = apply_fn(s_conn, op, strength, idx)
                s_conn_list.append(s_conn)
                strength_list.append(strength)

        if len(s_conn_list) > 0:
            s_conn = jnp.concatenate(s_conn_list, axis=0)
            strength = jnp.concatenate(strength_list, axis=0)
            out.append((update_mode, s_conn, strength))

    return out


@jax.jit
def _get_ndiff(psi: jax.Array, psi_accurate: jax.Array) -> jax.Array:
    diff1 = jnp.asarray(psi - psi_accurate)
    cond1 = jnp.abs(diff1) < 1e-8
    diff2 = jnp.asarray(psi / psi_accurate - 1)
    cond2 = jnp.abs(diff2) < 1e-3
    is_psi_close = cond1 | cond2
    ndiff = jnp.sum(~is_psi_close)
    return ndiff


def _check_samples(
    state: State, samples: Samples, use_ref: bool
) -> tuple[jax.Array, PsiArray | None, PyTree]:
    s = samples.spins
    psi = samples.psi
    internal = samples.state_internal

    if use_ref:
        if internal is None or psi is None:
            psi_accurate, internal = state.init_internal(s)
            if psi is not None:
                ndiff = _get_ndiff(psi, psi_accurate)
                if ndiff > 0 and jax.process_index() == 0:
                    warn(
                        f"{ndiff} out of {s.shape[0]} wavefunctions are not "
                        "close in direct forward pass and local updates. "
                        "This may indicate inaccurate local updates."
                    )
            psi = psi_accurate

    return s, psi, internal


@partial(jax.jit, static_argnums=(0, 1, 2))
def _init_Olocx(shape, dtype, sharding) -> jax.Array:
    dtype = jax.dtypes.canonicalize_dtype(dtype)
    return jax.lax.with_sharding_constraint(jnp.zeros(shape, dtype), sharding)


@eqx.filter_jit
def _chunk_and_ref(
    state: State, off_diags: list[tuple[dict[str, Any], jax.Array, jax.Array]]
) -> tuple[int | None, int | None, list[bool], bool]:
    forward_chunk = getattr(state, "forward_chunk", None)
    ref_chunk = getattr(state, "ref_chunk", None)
    if (
        forward_chunk is not None
        and ref_chunk is not None
        and forward_chunk < ref_chunk
    ):
        raise ValueError("Unsupported chunk size: forward_chunk < ref_chunk.")

    if state.use_ref:
        update_modes = [item[0] for item in off_diags]
        required_modes = state.required_update_modes
        use_ref = []
        for update_mode in update_modes:
            if not all(mode in update_mode.keys() for mode in required_modes):
                warn(
                    f"The update mode {required_modes} required by the state are not "
                    "all provided in the operator. Fall back to direct forward pass."
                )
                use_ref.append(False)
            else:
                use_ref.append(True)
    else:
        use_ref = [False] * len(off_diags)

    any_use_ref = any(use_ref)
    if not any_use_ref:
        ref_chunk = forward_chunk

    return forward_chunk, ref_chunk, use_ref, any_use_ref


@partial(jax.jit, static_argnums=(1, 2))
def _get_conn_size(
    H_conn: jax.Array, forward_chunk: int | None, psi_needed: bool
) -> jax.Array:
    ndevices = jax.device_count()
    valid = ~(jnp.isnan(H_conn) | jnp.isclose(H_conn, 0))
    size = jnp.sum(valid.reshape(ndevices, -1), axis=1)
    size = jnp.max(size)

    if psi_needed:
        size += H_conn.shape[0] // ndevices

    if forward_chunk is not None:
        size = ((size - 1) // forward_chunk + 1) * forward_chunk

    return size


@partial(jax.jit, static_argnums=(2, 4))
def _get_conn(
    s_conn: jax.Array, H_conn: jax.Array, conn_size: int, s: jax.Array, psi_needed: bool
) -> tuple[jax.Array, jax.Array, jax.Array]:
    ndevices = jax.device_count()
    nsamples, nconn, Nmodes = s_conn.shape
    H_conn = H_conn.reshape(ndevices, -1, nconn)
    s_conn = s_conn.reshape(ndevices, -1, nconn, Nmodes)
    s = s.reshape(ndevices, -1, Nmodes)

    def device_conn(s_conn: jax.Array, H_conn: jax.Array, s: jax.Array):
        is_valid = ~(jnp.isnan(H_conn) | jnp.isclose(H_conn, 0))
        segment, conn_idx = jnp.nonzero(is_valid, size=conn_size, fill_value=-1)
        s_conn = s_conn[segment, conn_idx]
        if psi_needed:
            s_conn = s_conn.at[-s.shape[0] :].set(s)
        H_conn = H_conn[segment, conn_idx]
        H_conn = jnp.where(segment == -1, 0, H_conn)
        return segment, s_conn, H_conn

    segment, s_conn, H_conn = jax.vmap(device_conn)(s_conn, H_conn, s)
    segment += jnp.arange(ndevices)[:, None] * (nsamples // ndevices)
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
    segment -= jnp.arange(ndevices)[:, None] * psi.shape[1]
    num_seg = psi.shape[1]

    @jax.vmap
    def fn(psi, segment, psi_conn, H_conn):
        psi_ratio = jnp.asarray(psi_conn / psi[segment])
        Olocx = jax.ops.segment_sum(psi_ratio * H_conn, segment, num_seg)
        return Olocx

    Olocx = fn(psi, segment, psi_conn, H_conn)
    return Olocx.flatten()


@partial(jax.jit, static_argnums=1)
def _stripe_psi(psi_conn: PsiArray, nsamples: int) -> PsiArray:
    ndevices = jax.device_count()
    psi_conn = psi_conn.reshape(ndevices, -1)
    psi = psi_conn[:, -nsamples // ndevices :]
    return psi.flatten()


def _Oloc(
    state: State,
    samples: Samples | NDArray[np.integer] | jax.Array,
    jax_op_list: list[tuple[dict[str, Any], tuple[OpTermJAX, ...]]],
) -> jax.Array:
    if not isinstance(samples, Samples):
        samples = Samples(to_distributed_array(samples))

    Oloc = _apply_diag(samples.spins, jax_op_list)
    off_diags = _apply_off_diag(samples.spins, jax_op_list)
    if len(off_diags) == 0:
        return Oloc

    forward_chunk, ref_chunk, use_ref, any_use_ref = _chunk_and_ref(state, off_diags)

    def get_Olocx_terms(samples, off_diags):
        s, psi, internal = _check_samples(state, samples, any_use_ref)
        Olocx = _init_Olocx(s.shape[:1], get_default_dtype(), s.sharding)

        for is_using_ref, (update_mode, s_conn, H_conn) in zip(use_ref, off_diags):
            psi_needed = psi is None
            conn_size = _get_conn_size(H_conn, forward_chunk, psi_needed).item()
            segment, s_conn, H_conn = _get_conn(
                s_conn, H_conn, conn_size, s, psi_needed
            )

            if is_using_ref:
                psi_conn = state.segment_ref_forward(
                    s_conn, s, update_mode, segment, internal
                )
            else:
                psi_conn = state(s_conn)

            if psi_needed:
                psi = _stripe_psi(psi_conn, s.shape[0])

            Olocx += _get_Olocx(psi, segment, psi_conn, H_conn)

        return Olocx

    get_Olocx_terms = chunk_map(get_Olocx_terms, chunk_size=ref_chunk)
    Oloc += get_Olocx_terms(samples, off_diags)
    return Oloc


@dataclass
class OpTerm:
    opstr: str
    strength: list[complex]
    indices: list[list[int]]

    def __post_init__(self):
        if len(self.strength) != len(self.indices):
            raise ValueError(
                f"`strength` and `indices` must have the same length, got "
                f"{len(self.strength)} and {len(self.indices)}"
            )

        for inds in self.indices:
            if len(inds) != len(self.opstr):
                raise ValueError(
                    f"`opstr`and each term of `indices` must have the same length, got "
                    f"{len(self.opstr)} and {len(inds)}"
                )


@jax.tree_util.register_pytree_node_class
@dataclass
class OpTermJAX:
    opstr: str
    strength: jax.Array
    indices: jax.Array

    def tree_flatten(self) -> tuple[tuple[jax.Array, jax.Array], str]:
        return (self.strength, self.indices), self.opstr

    @classmethod
    def tree_unflatten(
        cls, aux_data: str, children: tuple[jax.Array, jax.Array]
    ) -> OpTermJAX:
        strength, indices = children
        return cls(opstr=aux_data, strength=strength, indices=indices)


class Operator:
    """Quantum operator"""

    def __init__(self, op_list: list[OpTerm]):
        """
        :param op_list:
            The operator represented as a list of :class:`OpTerm`. Each :class:`OpTerm`
            groups all terms that share the same operator string ``opstr`` together with
            their strengths and site indices:

                opstr:
                    a `string <https://quspin.github.io/QuSpin/basis.html>`_ representing the operator type.
                    The convention is chosen the same as ``pauli=0`` in
                    `QuSpin <https://quspin.github.io/QuSpin/generated/quspin.basis.spin_basis_general.html#quspin.basis.spin_basis_general.__init__>`_

                strength:
                    a list of interaction strengths, one per term

                indices:
                    a list of site-index tuples, one per term, each matching the
                    length of ``opstr``
        """
        self._op_list = op_list
        self._reset_cache()

    def _reset_cache(self) -> None:
        """Reset the lazily-built caches derived from ``op_list``."""
        self._quspin_static_list = None
        self._jax_op_list = None
        self._quspin_op = dict()

    @property
    def op_list(self) -> list[OpTerm]:
        """Operator represented as a list in the QuSpin format"""
        return self._op_list

    @property
    def quspin_static_list(self) -> list:
        """
        The operator in the QuSpin static-list format
        ``[[opstr, [[strength, *site_indices], ...]], ...]``.
        """
        if self._quspin_static_list is None:
            static_list = []
            for op in self.op_list:
                terms = [[J, *inds] for J, inds in zip(op.strength, op.indices)]
                static_list.append([op.opstr, terms])
            self._quspin_static_list = static_list

        return self._quspin_static_list

    @property
    def jax_op_list(self) -> list[tuple[dict[str, Any], tuple[OpTermJAX, ...]]]:
        """
        Operator list with jax arrays, made easy for applying operator to basis states.

        The format is ``[(update_mode1, op_terms1), (update_mode2, op_terms2), ...]``,
        where each ``update_mode`` is a dictionary produced by the update-mode filter
        and each ``op_terms`` is a tuple of :class:`OpTermJAX` holding the jax-array
        strengths and indices of the terms in that update mode.
        """
        if self._jax_op_list is None:
            self._jax_op_list = []
            self.apply_update_mode_filter(none_filter)

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

        for op_term in self.op_list:
            for J, indices in zip(op_term.strength, op_term.indices):
                expression.append(f"{J:+}")
                for op, i in zip(op_term.opstr, indices):
                    expression.append(f"{op.translate(OP)}{str(i).translate(SUB)}")

        return " ".join(expression)

    def __repr__(self) -> str:
        return self.expression

    def apply_update_mode_filter(
        self, update_mode_filter: Callable[[str, Sequence[int]], dict[str, Any]]
    ) -> None:
        """
        Apply a filter function to update the operator's jax_op_list with additional
        update_mode information for each operator term.

        :param update_mode_filter:
            A function that takes an operator string and its corresponding site indices,
            and returns a dictionary of update_mode.
        """
        self._jax_op_list = []

        update_mode = {}
        values_dict = {}
        for op in self.op_list:
            for J, inds in zip(op.strength, op.indices):
                update_mode = update_mode_filter(op.opstr, inds)
                values = tuple(update_mode.values())
                new_op_list = values_dict.setdefault(values, [])
                opstr_list = [item[0] for item in new_op_list]
                try:
                    index = opstr_list.index(op.opstr)
                    new_op_list[index][1].append(J)
                    new_op_list[index][2].append(inds)
                except ValueError:
                    new_op_list.append([op.opstr, [J], [inds]])

        keys = update_mode.keys()
        sharding = get_replicated_sharding()
        for values, op in values_dict.items():
            update_mode = dict(zip(keys, values))
            op_list = []
            for opstr, strength, indices in op:
                strength = jnp.asarray(
                    strength, dtype=get_default_dtype(), device=sharding
                )
                indices = jnp.asarray(indices, dtype=jnp.int32, device=sharding)
                op_term = OpTermJAX(opstr, strength, indices)
                op_list.append(op_term)
            self._jax_op_list.append((update_mode, tuple(op_list)))

    def get_quspin_op(
        self, symm: Symmetry | None = None
    ) -> quspin.operators.hamiltonian:
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
            static_list = self.quspin_static_list
            self._quspin_op[symm] = hamiltonian(
                static_list=static_list,
                dynamic_list=[],
                basis=symm.basis,
                check_symm=False,
                check_herm=False,
                check_pcon=False,
                dtype=get_default_dtype(),  # type: ignore
            )
        return self._quspin_op[symm]

    def todense(self, symm: Symmetry | None = None) -> np.ndarray:
        """
        Obtain the dense matrix representing the operator

        :param symm:
            The symmetry used for generate the operator basis, by default the basis
            without symmetry
        """
        quspin_op = self.get_quspin_op(symm)
        return quspin_op.toarray()

    def _apply_to_state(self, state: State) -> DenseState:
        quspin_op = self.get_quspin_op(state.symm)
        psi = state.todense().psi
        if not isinstance(psi, np.ndarray):
            psi = jnp.asarray(psi)
            psi = to_replicated_numpy(psi)
        psi = quspin_op.dot(np.asarray(psi, order="C"))
        psi = np.asarray(psi)
        return DenseState(psi, state.symm)

    @overload
    def __matmul__(self, other: Operator) -> Operator: ...

    @overload
    def __matmul__(self, other: State) -> DenseState: ...

    def __matmul__(self, other: Operator | State) -> Operator | DenseState:
        r"""
        Apply the operator on a ket state by ``H @ state`` to get :math:`H \left| \psi \right>`,
        or multiply two operators by ``H1 @ H2``.
        The exact expectation value :math:`\left<\psi|H|\psi \right>` can be computed by
        ``state @ H @ state``.
        """
        if isinstance(other, Operator):
            op_list = []
            for op_term1 in self.op_list:
                for op_term2 in other.op_list:
                    opstr = op_term1.opstr + op_term2.opstr
                    strength = []
                    indices = []
                    for J1, inds1 in zip(op_term1.strength, op_term1.indices):
                        for J2, inds2 in zip(op_term2.strength, op_term2.indices):
                            strength.append(J1 * J2)
                            indices.append(inds1 + inds2)
                    op_list.append(OpTerm(opstr, strength, indices))
            return Operator(op_list)
        elif isinstance(other, State):
            return self._apply_to_state(other)

        return NotImplemented

    def __rmatmul__(self, state: State) -> DenseState:
        r"""
        Apply the operator on a bra state by ``state @ H`` to get :math:`\left< \psi \right| H`.
        The exact expectation value :math:`\left<\psi|H|\psi \right>` can be computed by
        ``state @ H @ state``.
        """
        if isinstance(state, State):
            return self._apply_to_state(state)
        return NotImplemented

    def diagonalize(
        self,
        symm: Symmetry | None = None,
        k: int | Literal["full"] = 1,
    ) -> tuple[NDArray, NDArray]:
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
            return quspin_op.eigsh(k=k, which="SA")  # type: ignore
        elif k == "full":
            array = self.todense(symm)
            return scipy.linalg.eigh(array)
        else:
            raise ValueError("Invalid value of `k`.")

    @property
    def H(self) -> Operator:
        """Hermitian conjugate"""
        op_list = copy.deepcopy(self.op_list)
        trans = str.maketrans("+-", "-+")
        for op_term in op_list:
            op_term.opstr = op_term.opstr.translate(trans)[::-1]
            for i, (J, inds) in enumerate(zip(op_term.strength, op_term.indices)):
                op_term.strength[i] = J.conjugate()
                op_term.indices[i] = inds[::-1]

        return Operator(op_list)

    def __add__(self, other: complex | Operator) -> Operator:
        """Add two operators."""
        if isinstance(other, (int, float, complex)):
            if not np.isclose(other, 0.0):
                raise ValueError("Constant shift is not implemented for Operator.")
            return self

        elif isinstance(other, Operator):
            op_list = copy.deepcopy(self.op_list)
            opstr1 = [op_term.opstr for op_term in op_list]
            for op_term in copy.deepcopy(other.op_list):
                try:
                    index = opstr1.index(op_term.opstr)
                    op_list[index].strength += op_term.strength
                    op_list[index].indices += op_term.indices
                except ValueError:
                    op_list.append(op_term)
                    opstr1.append(op_term.opstr)
            return Operator(op_list)

        return NotImplemented

    def __radd__(self, other: complex) -> Operator:
        if isinstance(other, (int, float, complex)):
            return self + other
        return NotImplemented

    def __iadd__(self, other: complex | Operator) -> Operator:
        """In-place addition of two operators."""
        if isinstance(other, (int, float, complex)):
            if not np.isclose(other, 0.0):
                raise ValueError("Constant shift is not implemented for Operator.")
            return self

        elif isinstance(other, Operator):
            op_list = self.op_list
            opstr1 = [op_term.opstr for op_term in op_list]
            for op_term in copy.deepcopy(other.op_list):
                try:
                    index = opstr1.index(op_term.opstr)
                    op_list[index].strength += op_term.strength
                    op_list[index].indices += op_term.indices
                except ValueError:
                    op_list.append(op_term)
                    opstr1.append(op_term.opstr)
            self._reset_cache()
            return self

        return NotImplemented

    def __sub__(self, other: complex | Operator) -> Operator:
        """Subtract two operators."""
        if isinstance(other, (int, float, complex)):
            if not np.isclose(other, 0.0):
                raise ValueError("Constant shift is not implemented for Operator.")
            return self
        if isinstance(other, Operator):
            return self + (-other)
        return NotImplemented

    def __rsub__(self, other: complex) -> Operator:
        if isinstance(other, (int, float, complex)):
            if not np.isclose(other, 0.0):
                raise ValueError("Constant shift is not implemented for Operator.")
            return -self
        return NotImplemented

    def __isub__(self, other: complex | Operator) -> Operator:
        """In-place subtraction of two operators."""
        self += -other
        return self

    def __mul__(self, other: ArrayLike) -> Operator:
        """Multiply an operator with a scalar."""
        num = np.asarray(other).item()
        op_list = copy.deepcopy(self.op_list)
        for op_term in op_list:
            op_term.strength = [J * num for J in op_term.strength]
        return Operator(op_list)

    def __rmul__(self, other: ArrayLike) -> Operator:
        """Multiply an operator with a scalar."""
        if eqx.is_array_like(other):
            return self * other
        return NotImplemented

    def __imul__(self, other: ArrayLike) -> Operator:
        """In-place multiplication of an operator with a scalar."""
        num = np.asarray(other).item()
        for op_term in self.op_list:
            op_term.strength = [J * num for J in op_term.strength]
        self._reset_cache()
        return self

    def __neg__(self) -> Operator:
        """Negate an operator."""
        return (-1) * self

    def __truediv__(self, other: complex) -> Operator:
        """Divide an operator by a scalar."""
        if isinstance(other, (int, float, complex)):
            return self * (1 / other)
        return NotImplemented

    def __itruediv__(self, other: complex) -> Operator:
        """In-place division of an operator by a scalar."""
        if isinstance(other, (int, float, complex)):
            return self.__imul__(1 / other)
        return NotImplemented

    def apply_diag(self, s: jax.Array) -> jax.Array:
        r"""
        Apply the diagonal part of the operator to a batch of configurations ``s``,
        returning the diagonal matrix elements :math:`\left< s|O|s \right>`.
        """
        return _apply_diag(s, self.jax_op_list)

    def apply_off_diag(
        self, s: jax.Array
    ) -> list[tuple[dict[str, Any], jax.Array, jax.Array]]:
        r"""
        Apply the off-diagonal part of the operator to a batch of configurations ``s``.

        :return:
            A list of ``(update_mode, s_conn, strength)`` grouped by update mode,
            where ``s_conn`` are the connected configurations :math:`s'` and
            ``strength`` the corresponding matrix elements :math:`\left< s'|O|s \right>`.
        """
        return _apply_off_diag(s, self.jax_op_list)

    def Oloc(
        self, state: State, samples: Samples | NDArray[np.integer] | jax.Array
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
        if self._jax_op_list is None:
            if not state.use_ref:
                self.apply_update_mode_filter(none_filter)
            elif "nflips_up" in state.required_update_modes:
                self.apply_update_mode_filter(nflips_up_dn_filter)
            else:
                self.apply_update_mode_filter(nflips_filter)

        return _Oloc(state, samples, self.jax_op_list)

    @overload
    def expectation(
        self,
        state: State,
        samples: Samples | jax.Array,
        return_var: Literal[False] = False,
    ) -> complex: ...

    @overload
    def expectation(
        self, state: State, samples: Samples | jax.Array, return_var: Literal[True]
    ) -> tuple[complex, float]: ...

    def expectation(
        self, state: State, samples: Samples | jax.Array, return_var: bool = False
    ) -> complex | tuple[complex, float]:
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
        if isinstance(samples, Samples):
            if samples.reweight_factor is None:
                reweight = 1.0
            else:
                reweight = samples.reweight_factor
        else:
            reweight = 1.0
        Oloc = self.Oloc(state, samples)
        Omean = jnp.mean(Oloc * reweight)
        if return_var:
            Ovar = jnp.mean(jnp.abs(Oloc) ** 2 * reweight) - jnp.abs(Omean) ** 2
            return Omean.item(), Ovar.real.item()
        else:
            return Omean.item()
