import warnings
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
import quantax as qtx
from quantax.sites import Chain
from quantax.operator import (
    Operator,
    OpTerm,
    sigma_x,
    sigma_y,
    sigma_z,
    sigma_p,
    sigma_m,
    create_u,
    create_d,
    annihilate_u,
    annihilate_d,
    number_u,
    number_d,
    Heisenberg,
    Hubbard,
    tJ,
)
from quantax.operator.operator import _get_conn_size, _chunk_and_ref
from quantax.nn import RefModel, exp_by_log
from quantax.sampler import Samples
from quantax.state import DenseState, State, Variational
from quantax.symmetry import Identity
from quantax.utils import LogArray, ints_to_array, array_to_ints, to_distributed_array
from _dtype import use_dtype
from _op_helpers import dense

PT = qtx.PARTICLE_TYPE
NDEV = jax.device_count()


def _full_basis() -> jax.Array:
    """
    All configurations in QuSpin basis order, repeated NDEV times to be divisible
    over the devices. The uniform repetition keeps every mean over the basis exact.
    """
    symm = Identity()
    symm.basis_make()
    configs = np.tile(ints_to_array(symm.basis.states), (NDEV, 1))
    return to_distributed_array(jnp.asarray(configs))


def _tiled_diag(op: Operator) -> np.ndarray:
    """Diagonal of the dense matrix, repeated as in `_full_basis`."""
    return np.tile(np.diag(dense(op)), NDEV)


class _ExpModel(RefModel):
    r"""
    :math:`\psi(s) = \exp(a \cdot s + s^T W s)`, cheap and non-zero everywhere,
    returned as a `~quantax.utils.LogArray` like the models of quantax.
    The internal of a reference is its :math:`\log\psi`, and ``ref_forward`` fails
    like a low-rank update when more sites are flipped than ``update_mode`` allows.
    """

    a: jax.Array
    w: jax.Array

    def __init__(self, Nmodes: int):
        rng = np.random.default_rng(0)
        dtype = qtx.get_default_dtype()
        a = rng.standard_normal(Nmodes) + 1j * rng.standard_normal(Nmodes)
        w = rng.standard_normal((Nmodes,) * 2) + 1j * rng.standard_normal((Nmodes,) * 2)
        if not jnp.issubdtype(dtype, jnp.complexfloating):
            a, w = a.real, w.real
        self.a = jnp.asarray(0.3 * a, dtype)
        self.w = jnp.asarray(0.1 * w, dtype)

    def log_psi(self, s):
        s = s.astype(self.a.dtype)
        return self.a @ s + s @ self.w @ s

    def __call__(self, s):
        return exp_by_log(self.log_psi(s))

    def init_internal(self, s):
        log_psi = self.log_psi(s)
        return exp_by_log(log_psi), log_psi

    @property
    def required_update_modes(self):
        return ("nflips",)

    def ref_forward(self, s, s_old, update_mode, internal, return_update=False):
        log_psi = internal + self.log_psi(s) - self.log_psi(s_old)
        nflips = jnp.sum(s != s_old)
        log_psi = jnp.where(nflips <= update_mode["nflips"], log_psi, jnp.nan)
        psi = exp_by_log(log_psi)
        return (psi, log_psi) if return_update else psi


_batch_psi = eqx.filter_jit(
    eqx.filter_vmap(lambda model, s: model(s), in_axes=(None, 0))
)


class _CountingState(State):
    """A state evaluating `_ExpModel` as a plain array, counting its forward passes."""

    def __init__(self, Nmodes: int, forward_chunk: int | None = None):
        super().__init__()
        self.model = _ExpModel(Nmodes)
        self.forward_chunk = forward_chunk
        self.ncalls = 0

    def __call__(self, s):
        self.ncalls += 1
        return jnp.asarray(_batch_psi(self.model, jnp.asarray(s)))


def test_op_term_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        OpTerm("zz", [1.0, 2.0], [[0, 1]])  # one strength per term
    with pytest.raises(ValueError, match="same length"):
        OpTerm("zz", [1.0], [[0]])  # one site per operator character


# --- algebra against the QuSpin dense matrix (ground truth) ---


def test_sub_matches_dense():
    Chain(4, boundary=1)
    A = sigma_z(0) @ sigma_z(1)
    B = sigma_z(1) @ sigma_z(2)
    assert np.allclose(dense(A - B), dense(A) - dense(B))


def test_scalar_mul_div_neg():
    Chain(4, boundary=1)
    A = sigma_z(0) @ sigma_z(1)
    DA = dense(A)
    assert np.allclose(dense(2.0 * A), 2.0 * DA)
    assert np.allclose(dense(A * 2.0), 2.0 * DA)
    assert np.allclose(dense(A / 2.0), DA / 2.0)
    assert np.allclose(dense(-A), -DA)


def test_matmul_operator_product_is_ordered():
    # Operator product must respect matrix order: (A @ B) acts B first.
    Chain(4, boundary=1)
    A = sigma_x(0) @ sigma_x(1)
    B = sigma_z(1) @ sigma_z(2)
    assert np.allclose(dense(A @ B), dense(A) @ dense(B))
    # the product is non-commuting here, guarding against accidental symmetry
    assert not np.allclose(dense(A @ B), dense(B @ A))


def test_hermitian_conjugate():
    use_dtype(jnp.complex64)
    Chain(4, boundary=1)
    A = 1j * (sigma_p(0) @ sigma_m(1)) + sigma_z(2)
    assert np.allclose(dense(A.H), dense(A).conj().T)
    # H does not mutate the original operator
    assert np.allclose(
        dense(A), (1j * dense(sigma_p(0) @ sigma_m(1))) + dense(sigma_z(2))
    )


def test_hermitian_conjugate_fermions():
    # Reversing the operator string must give the fermionic order of the adjoint,
    # (c_0^† c_2)^† = c_2^† c_0, including the Jordan-Wigner signs of the modes between.
    use_dtype(jnp.complex64)
    Chain(3, boundary=-1, particle_type=PT.spinful_fermion)
    A = (1 + 2j) * (create_u(0) @ annihilate_u(2)) + 0.5j * (create_u(1) @ create_d(2))
    assert np.allclose(dense(A.H), dense(A).conj().T)


def test_expression():
    Chain(3, boundary=-1, particle_type=PT.spinful_fermion)
    assert (sigma_z(0) @ sigma_x(2)).expression == "+4.0 Sᶻ₀ Sˣ₂"
    # fermionic systems print creation and annihilation operators, with the
    # spin-down modes after the spin-up ones
    assert repr(-(create_u(1) @ annihilate_d(2))) == "-1.0 c†₁ c₅"


# --- regressions for in-place operators (cache invalidation) ---


def test_imul_invalidates_cache():
    Chain(4, boundary=1)
    H = sigma_z(0) @ sigma_z(1)
    before = dense(H)  # populates the QuSpin / jax caches
    H *= 3.0
    assert np.allclose(dense(H), 3.0 * before)
    # the jax op list (used by Oloc) must also reflect the new strength
    strengths = [J for _, terms in H.jax_op_list for t in terms for J in t.strength]
    assert np.allclose(np.real(strengths), 3.0 * 4.0)  # sigma_z(0)@sigma_z(1) -> 4.0


def test_itruediv_invalidates_cache():
    Chain(4, boundary=1)
    H = sigma_z(0) @ sigma_z(1)
    before = dense(H)
    H /= 2.0
    assert np.allclose(dense(H), before / 2.0)


def test_iadd_returns_self_and_invalidates_cache():
    Chain(4, boundary=1)
    H = sigma_z(0) @ sigma_z(1)
    B = sigma_z(1) @ sigma_z(2)
    before = dense(H)  # populate cache before mutating
    hid = id(H)
    H += B
    assert hid == id(H)  # __iadd__ returns the same object
    assert np.allclose(dense(H), before + dense(sigma_z(1) @ sigma_z(2)))


def test_isub_returns_self():
    Chain(4, boundary=1)
    H = 2.0 * (sigma_z(0) @ sigma_z(1))
    before = dense(H)
    hid = id(H)
    H -= sigma_z(0) @ sigma_z(1)
    assert hid == id(H)
    assert np.allclose(dense(H), before - dense(sigma_z(0) @ sigma_z(1)))


# --- regressions for duplicate-term merging in __add__ / __iadd__ ---


def test_add_merges_same_opstr_from_other():
    # `other` contributes two terms sharing opstr 'zz' that is absent from `self`.
    Chain(4, boundary=1)
    A = sigma_x(0) @ sigma_x(1)
    B = (sigma_z(0) @ sigma_z(1)) + (sigma_z(2) @ sigma_z(3))
    C = A + B
    opstrs = [t.opstr for t in C.op_list]
    assert opstrs.count("zz") == 1
    assert opstrs.count("xx") == 1
    assert np.allclose(dense(C), dense(A) + dense(B))


def test_iadd_merges_same_opstr_from_other():
    Chain(4, boundary=1)
    A = sigma_x(0) @ sigma_x(1)
    expected = (
        dense(A) + dense(sigma_z(0) @ sigma_z(1)) + dense(sigma_z(2) @ sigma_z(3))
    )
    A += (sigma_z(0) @ sigma_z(1)) + (sigma_z(2) @ sigma_z(3))
    opstrs = [t.opstr for t in A.op_list]
    assert opstrs.count("zz") == 1
    assert np.allclose(dense(A), expected)


# --- constant shift is explicitly unsupported ---


@pytest.mark.parametrize(
    "shift, sign",
    [
        (lambda H, c: H + c, 1),
        (lambda H, c: c + H, 1),
        (lambda H, c: H - c, 1),
        (lambda H, c: c - H, -1),
        (lambda H, c: H.__iadd__(c), 1),
        (lambda H, c: H.__isub__(c), 1),
    ],
)
def test_nonzero_constant_shift_raises(shift, sign):
    Chain(4, boundary=1)
    H = sigma_z(0) @ sigma_z(1)
    with pytest.raises(ValueError, match="Constant shift"):
        shift(H, 1.0)
    # shifting by zero is a no-op and allowed
    assert np.allclose(dense(shift(H, 0.0)), sign * dense(H))


# --- exact application through the QuSpin path ---


def test_apply_to_dense_state():
    Chain(4, boundary=1)
    H = Heisenberg(J=1.0)
    w, v = H.diagonalize(k=1)
    gs = DenseState(np.asarray(v[:, 0]))
    # ket: H|gs> = E|gs>, bra: <gs|H|gs> = E
    assert np.allclose((H @ gs).psi, w[0] * gs.psi, atol=1e-5)
    energy = gs @ H @ gs
    assert np.isclose(complex(energy), float(w[0]), atol=1e-5)
    # a jax wave function is applied the same way as a numpy one
    psi = jnp.asarray(np.random.default_rng(0).standard_normal(16), jnp.float32)
    assert np.allclose((H @ DenseState(psi)).psi, dense(H) @ np.asarray(psi), atol=1e-5)


def test_diagonalize_full_matches_dense_eigh():
    Chain(4, boundary=1)
    H = Heisenberg(J=1.0)
    w_full, v = H.diagonalize(k="full")
    Hd = dense(H)
    # the full spectrum matches a direct dense eigendecomposition...
    assert np.allclose(np.sort(w_full), np.linalg.eigvalsh(Hd), atol=1e-5)
    # ...and the returned eigenvectors actually diagonalize H
    assert np.allclose(v.conj().T @ Hd @ v, np.diag(w_full), atol=1e-5)


def test_diagonalize_invalid_k_raises():
    Chain(4, boundary=1)
    H = Heisenberg(J=1.0)
    with pytest.raises(ValueError):
        H.diagonalize(k="lowest")


# --- jax application path against the QuSpin dense matrix ---


def _spin_ops():
    # a single sigma^y pins its sign, which cancels in sigma^y sigma^y
    Chain(4, boundary=1)
    return (
        0.5 * sigma_y(1)
        + 0.3 * (sigma_x(0) @ sigma_y(2))
        + 0.7 * (sigma_p(1) @ sigma_m(3))
        + sigma_z(0) @ sigma_z(3)
    )


def _spinful_ops():
    # non-Hermitian terms check that s' and <s|O|s'> are not transposed
    Chain(3, boundary=-1, particle_type=PT.spinful_fermion)
    return (
        Hubbard(U=4.0)
        + 0.2 * (create_u(1) @ annihilate_d(2))
        + 0.3 * (sigma_x(0) @ sigma_y(2))
    )


def _spinful_no_double_occ_ops():
    # the hoppings into double occupancy must be dropped, not only weighted by psi,
    # and the exchange strings act on the up and down modes of the same site; the
    # hopping without its transpose tells O from O^T. Like QuSpin, only the final
    # configuration of a string is projected, so c_{0↓} c^†_{0↑} passing through a
    # double occupancy is -c^†_{0↑} c_{0↓}, not zero.
    Chain(3, boundary=-1, particle_type=PT.spinful_fermion, double_occ=False)
    return (
        tJ(J=0.5)
        + 0.3 * number_u(0)
        + 0.2j * (create_u(0) @ annihilate_u(1))
        + 0.4 * (annihilate_d(0) @ create_u(0))
    )


@pytest.mark.parametrize("build", [_spin_ops, _spinful_ops, _spinful_no_double_occ_ops])
def test_apply_matches_dense_matrix_elements(build):
    # The rows <s|O|s'> assembled from apply_diag / apply_off_diag over all basis
    # states s must reproduce the QuSpin matrix, including Jordan-Wigner signs of
    # fermions and the transpose of non-symmetric terms.
    use_dtype(jnp.complex64)
    H = build()
    symm = Identity()
    symm.basis_make()
    states = symm.basis.states
    index = {int(b): i for i, b in enumerate(states)}
    spins = jnp.asarray(ints_to_array(states))

    M = np.diag(np.asarray(H.apply_diag(spins)))
    for _, s_conn, strength in H.apply_off_diag(spins):
        s_conn, strength = np.asarray(s_conn), np.asarray(strength)
        valid = ~(np.isnan(strength) | (strength == 0))
        rows = np.nonzero(valid)[0]
        cols = [index[int(b)] for b in array_to_ints(s_conn[valid])]
        np.add.at(M, (rows, cols), strength[valid])
    np.testing.assert_allclose(M, dense(H), atol=1e-6)


# --- Oloc and expectation ---


def _spin_hamiltonian() -> Operator:
    """
    A spin operator with all spin operator characters. The complex coefficients test
    the conjugation of the matrix elements, and the non-symmetric terms (O^T != O)
    that Oloc sums the row <s|O|s'> over s', not the column <s'|O|s>.
    """
    Chain(4, boundary=1)
    bonds = [(0, 1), (1, 2), (2, 3), (3, 0)]
    # terms of equal size share the compilation of their jax arrays
    field = sum((0.7 + 0.2j) * sigma_x(i) for i in range(4))
    yy = sum(0.3 * (sigma_y(i) @ sigma_y(j)) for i, j in bonds)
    # sigma^+ sigma^- without its transpose, and the imaginary antisymmetric
    # sigma^x sigma^y
    asym = 0.4 * (sigma_p(0) @ sigma_m(2)) + 0.5 * (sigma_x(1) @ sigma_y(3))
    identity = Operator([OpTerm("I", [0.5] * 4, [[i] for i in range(4)])])
    return Heisenberg(J=1.0) + field + yy + asym + identity


def _exact_oloc(H: Operator, psi: jax.Array) -> np.ndarray:
    """(H psi)(s) / psi(s) over the repeated full basis, in double precision."""
    H = dense(H)
    psi = np.asarray(psi, np.complex128).reshape(-1, H.shape[0])
    return ((psi @ H.T) / psi).ravel()


def test_oloc_chunked_fused_forward_matches_dense():
    # forward_chunk smaller than the per-device batch forces the chunk_map loop in
    # _Oloc, and raw samples without psi exercise the fused forward pass where the
    # samples ride in the tail of the conn buffer.
    use_dtype(jnp.complex64)
    H = _spin_hamiltonian()
    state = _CountingState(4, forward_chunk=8)
    spins = _full_basis()
    given_psi = state(spins)
    psi = np.asarray(given_psi, np.complex128)
    Oloc_exact = _exact_oloc(H, psi)
    Oloc = H.Oloc(state, spins)
    np.testing.assert_allclose(np.asarray(Oloc), Oloc_exact, rtol=1e-5, atol=1e-5)
    # a given psi, as returned by Metropolis, takes the place of the sample tail
    Oloc = H.Oloc(state, Samples(spins, psi=given_psi))
    np.testing.assert_allclose(np.asarray(Oloc), Oloc_exact, rtol=1e-5, atol=1e-5)

    # psi evaluated in the conn forward pass is written back into the samples
    # carrying only prob, and r is updated from it. All configurations drawn equally
    # often follow a uniform prob, so the reweighted mean is the exact expectation
    # value. State.expectation forwards to Operator.expectation.
    state.ncalls = 0
    samples = Samples(spins, prob=jnp.ones(spins.shape[0]))
    Omean, Ovar = state.expectation(H, samples, return_var=True)
    nchunks = spins.shape[0] // NDEV // state.forward_chunk
    assert state.ncalls == nchunks  # one fused forward pass per chunk
    np.testing.assert_allclose(np.asarray(samples.psi), psi, rtol=1e-6)
    w = np.abs(psi) ** 2 / np.mean(np.abs(psi) ** 2)
    np.testing.assert_allclose(np.asarray(samples.r), w, rtol=1e-5)
    psi0 = psi[: psi.size // NDEV]  # a single copy of the basis
    exact = (psi0.conj() @ dense(H) @ psi0) / (psi0.conj() @ psi0)
    assert np.isclose(Omean, exact, atol=1e-5)
    assert np.isclose(Ovar, np.mean(w * np.abs(Oloc_exact - exact) ** 2), rtol=1e-5)


@pytest.mark.parametrize("build", [_spinful_ops, _spinful_no_double_occ_ops])
def test_oloc_fermion_ops_match_dense(build):
    # Oloc(s) = (O psi)(s) / psi(s), including the Jordan-Wigner signs and the
    # no-double-occupancy constraint of the transposed strings; the non-Hermitian
    # terms tell O from O^T
    use_dtype(jnp.complex64)
    H = build()
    state = _CountingState(qtx.get_sites().Nfmodes)
    spins = _full_basis()
    psi = state(spins)
    Oloc = H.Oloc(state, Samples(spins, psi=psi))
    Oloc_exact = _exact_oloc(H, np.asarray(psi, np.complex128))
    np.testing.assert_allclose(np.asarray(Oloc), Oloc_exact, rtol=1e-5, atol=1e-5)


def test_oloc_ref_forward_matches_dense():
    # The connected configurations are evaluated from the internals of their
    # reference, in the groups of nflips_filter required by the model.
    use_dtype(jnp.complex64)
    H = _spin_hamiltonian()  # nflips = 0, 1 and 2
    state = Variational(_ExpModel(4))
    spins = _full_basis()
    psi, internal = state.init_internal(spins)
    Oloc_exact = _exact_oloc(H, psi)

    # a given psi is used as it is, while r is updated from it
    samples = Samples(spins, psi=psi, prob=jnp.abs(psi), state_internal=internal)
    Oloc = H.Oloc(state, samples)
    np.testing.assert_allclose(np.asarray(Oloc), Oloc_exact, rtol=1e-5, atol=1e-5)
    assert sorted(mode["nflips"] for mode, _ in H.jax_op_list) == [0, 1, 2]
    assert samples.psi is psi
    r = np.abs(np.asarray(psi))
    np.testing.assert_allclose(np.asarray(samples.r), r / r.mean(), rtol=1e-5)

    # samples carrying only prob, as returned by Metropolis for ref states, get psi
    # from init_internal
    samples = Samples(spins, prob=jnp.abs(psi))
    Oloc = H.Oloc(state, samples)
    np.testing.assert_allclose(np.asarray(Oloc), Oloc_exact, rtol=1e-5, atol=1e-5)
    assert isinstance(samples.psi, LogArray)  # not materialized, avoiding overflow
    np.testing.assert_allclose(np.asarray(samples.psi), np.asarray(psi), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(samples.r), r / r.mean(), rtol=1e-5)

    # without internals, psi of init_internal replaces the given one, with a warning
    # only if they disagree
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        H.Oloc(state, Samples(spins, psi=psi))
    with pytest.warns(UserWarning, match="not close"):
        Oloc = H.Oloc(state, Samples(spins, psi=2 * psi))
    np.testing.assert_allclose(np.asarray(Oloc), Oloc_exact, rtol=1e-5, atol=1e-5)


def test_oloc_chunked_ref_checks_only_real_samples():
    # ref_chunk = 12 pads the 16 samples per device with zeros to 24. The psi of the
    # padding rows must not be compared with init_internal, and the mismatches of all
    # chunks are reported in a single warning.
    use_dtype(jnp.complex64)
    Chain(4, boundary=1)
    H = Heisenberg(J=1.0)
    state = Variational(_ExpModel(4), max_parallel=12)
    spins = _full_basis()
    psi, _ = state.init_internal(spins)
    Oloc_exact = _exact_oloc(H, psi)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Oloc = H.Oloc(state, Samples(spins, psi=psi))
    np.testing.assert_allclose(np.asarray(Oloc), Oloc_exact, rtol=1e-5, atol=1e-5)

    wrong = np.arange(spins.shape[0]) % 5 == 0
    psi_wrong = np.where(wrong, 2, 1) * np.asarray(psi)
    psi_wrong = to_distributed_array(jnp.asarray(psi_wrong))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        Oloc = H.Oloc(state, Samples(spins, psi=psi_wrong))
    messages = [str(w.message) for w in record if "not close" in str(w.message)]
    assert len(messages) == 1
    assert messages[0].startswith(f"{wrong.sum()} out of {spins.shape[0]} ")
    np.testing.assert_allclose(np.asarray(Oloc), Oloc_exact, rtol=1e-5, atol=1e-5)


class _RefStub(State):
    """The chunk sizes and ref forward settings read by Oloc from a state."""

    required_update_modes = ("nflips",)
    use_ref = True

    def __init__(self, forward_chunk, ref_chunk):
        super().__init__()
        self.forward_chunk = forward_chunk
        self.ref_chunk = ref_chunk


class _DirectStub(_RefStub):
    use_ref = False


class _SpinfulRefStub(_RefStub):
    required_update_modes = ("nflips_up", "nflips_dn")


def test_chunk_and_ref_falls_back_without_update_modes():
    # Pins the internal contract of _chunk_and_ref, which returns
    # (forward_chunk, ref_chunk, use_ref per group, any use_ref), so this test must
    # be updated together with that helper.
    Chain(4, boundary=1)
    state = _RefStub(4, 2)
    s_conn, H_conn = jnp.zeros((4, 1, 4)), jnp.zeros((4, 1))
    ref = ({"nflips": 2}, s_conn, H_conn)
    no_mode = ({}, s_conn, H_conn)
    assert _chunk_and_ref(state, [ref]) == (4, 2, [True], True)
    # equal chunks are the default of Variational(max_parallel=int)
    assert _chunk_and_ref(_RefStub(2, 2), [ref]) == (2, 2, [True], True)
    # a group without the update mode required by the model is evaluated by the
    # direct forward pass
    with pytest.warns(UserWarning, match="Fall back to direct forward pass"):
        assert _chunk_and_ref(state, [ref, no_mode]) == (4, 2, [True, False], True)
    # the samples are chunked by forward_chunk if no group uses the ref forward
    with pytest.warns(UserWarning, match="Fall back to direct forward pass"):
        assert _chunk_and_ref(state, [no_mode]) == (4, 4, [False], False)
    assert _chunk_and_ref(_DirectStub(4, 2), [ref]) == (4, 4, [False], False)

    with pytest.raises(ValueError, match="forward_chunk < ref_chunk"):
        _chunk_and_ref(_RefStub(2, 4), [ref])


def test_oloc_groups_by_spin_resolved_update_modes():
    # A state whose local updates treat the two spin sectors separately gets the
    # terms grouped by nflips_up_dn_filter.
    Chain(3, boundary=-1, particle_type=PT.spinful_fermion)
    U = number_u(0) @ number_d(0)
    Oloc = U.Oloc(_SpinfulRefStub(None, None), _full_basis())
    np.testing.assert_allclose(np.asarray(Oloc), _tiled_diag(U))
    assert [mode for mode, _ in U.jax_op_list] == [{"nflips_up": 0, "nflips_dn": 0}]


def test_oloc_evaluates_psi_only_if_needed():
    use_dtype(jnp.complex64)
    Chain(4, boundary=1)
    Z = sigma_z(0) @ sigma_z(1)
    state = _CountingState(4)
    spins = _full_basis()
    psi = state(spins)
    state.ncalls = 0

    # A diagonal operator doesn't evaluate any connected configuration...
    samples = Samples(spins)
    Oloc = Z.Oloc(state, samples)
    np.testing.assert_allclose(np.asarray(Oloc), _tiled_diag(Z), atol=1e-6)
    assert samples.psi is None and samples.r is None
    # ...and sigma^+ sigma^+ has no non-zero connection, so with a given psi the
    # forward pass of the empty connected configurations is skipped.
    Oloc = (sigma_p(0) @ sigma_p(0) + Z).Oloc(state, Samples(spins, psi=psi))
    np.testing.assert_allclose(np.asarray(Oloc), _tiled_diag(Z), atol=1e-6)
    assert state.ncalls == 0

    # psi is evaluated by a separate forward pass only if prob requires it for r
    samples = Samples(spins, prob=jnp.ones(spins.shape[0]))
    Omean = Z.expectation(state, samples)
    assert state.ncalls == 1
    np.testing.assert_allclose(np.asarray(samples.psi), np.asarray(psi), rtol=1e-6)
    w = np.abs(np.asarray(psi)) ** 2
    np.testing.assert_allclose(np.asarray(samples.r), w / w.mean(), rtol=1e-5)
    assert np.isclose(Omean, np.sum(w * _tiled_diag(Z)) / w.sum(), atol=1e-5)


def test_expectation_weighted_mean_and_variance():
    # Oloc is replaced by a given array to test the statistics alone. r with
    # mean != 1 (e.g. given by hand) is normalized by sum(r), the variance of a
    # complex Oloc is <|O - <O>|^2>, and it doesn't suffer from the catastrophic
    # cancellation of <|O|^2> - |<O>|^2 in single precision under a large shift.
    n = 4 * NDEV
    rng = np.random.default_rng(0)
    Oloc = 1e4 + rng.standard_normal(n) + 1j * rng.standard_normal(n)
    Oloc = Oloc.astype(np.complex64)
    r = rng.uniform(0.5, 1.5, n) * 3.0
    op = Operator([])
    op.Oloc = lambda state, samples: to_distributed_array(jnp.asarray(Oloc))

    r_dist = to_distributed_array(jnp.asarray(r, jnp.float32))
    samples = Samples(jnp.ones((n, 4)), r=r_dist)
    Omean, Ovar = op.expectation(None, samples, return_var=True)
    Oloc, w = Oloc.astype(np.complex128), r / r.sum()
    mean = np.sum(w * Oloc)
    assert np.isclose(Omean, mean, rtol=1e-6)
    assert np.isclose(Ovar, np.sum(w * np.abs(Oloc - mean) ** 2), rtol=1e-4)
    # raw configurations are weighted uniformly
    assert np.isclose(op.expectation(None, jnp.ones((n, 4))), Oloc.mean(), rtol=1e-6)


def test_get_conn_size():
    # identical 6x3 blocks on every device so the expected count is
    # independent of the device number
    block = np.ones((6, 3), np.float32)
    block[0, :] = np.nan  # invalid connections
    block[1, 1] = 0.0  # explicit zeros are not valid connections
    H_conn = jnp.asarray(np.tile(block, (NDEV, 1)))  # 14 valid per device

    assert int(_get_conn_size(H_conn, None, False)) == 14
    # counts the whole device batch even when it exceeds forward_chunk,
    # then rounds up to a multiple of forward_chunk
    assert int(_get_conn_size(H_conn, 4, False)) == 16
    # psi_needed reserves one extra slot per sample on each device, before rounding
    assert int(_get_conn_size(H_conn, 4, True)) == 20
