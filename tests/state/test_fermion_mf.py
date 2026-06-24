"""
Tests for the fermionic mean-field states in ``quantax.state.fermion_mf``.

``mf_expectation`` evaluates :math:`\\left< O \\right>` by Wick contraction of the
mean-field one-/two-body density matrices, so it can be checked against two
independent references:

* For (multi-)determinant states the wavefunction has a fixed particle number, so
  the mean-field value equals the exact :math:`\\left<\\psi|O|\\psi\\right> /
  \\left<\\psi|\\psi\\right>` obtained from the dense vector (``todense``).
* For any Gaussian state the four-point function obeys the explicit Wick formula
  written out in :func:`_wick_4point`, evaluated from the density matrices returned
  by ``rho_from_model``.

The multi-component paths (``MultiDetState`` / ``MultiPfState``) are additionally
pinned against their single-component limits, which directly guards the
``strength``/``indices`` contraction arguments.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest
import quantax as qtx
from quantax.sites import Chain
from quantax.operator import Operator, OpTerm, tV, Hubbard, Heisenberg, number
from quantax.model import (
    GeneralDet,
    RestrictedDet,
    UnrestrictedDet,
    MultiDet,
    GeneralPf,
    SingletPair,
    MultiPf,
)
from quantax.state import (
    MeanFieldFermionState,
    GeneralDetState,
    RestrictedDetState,
    UnrestrictedDetState,
    MultiDetState,
    GeneralPfState,
    SingletPairState,
    MultiPfState,
)
from _dtype import use_dtype


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _spinless_chain(L=6, N=3):
    return Chain(
        L, boundary=1, particle_type=qtx.PARTICLE_TYPE.spinless_fermion, Nparticles=N
    )


def _antisym(rng, L, dtype):
    A = rng.standard_normal((L, L))
    if np.iscomplexobj(np.empty(0, dtype)):
        A = A + 1j * rng.standard_normal((L, L))
    A = np.asarray(A, dtype)
    return A - A.T


def _exact_dense(state, H):
    """Exact ``<psi|H|psi> / <psi|psi>`` from the dense wavefunction."""
    psi = np.asarray(state.todense().psi).flatten()
    Hd = H.todense()
    return (psi.conj() @ Hd @ psi) / (psi.conj() @ psi)


def _rho_kappa(state):
    """Density matrices as numpy; kappa is zero for unpaired states."""
    out = type(state).rho_from_model(state.model)
    if isinstance(out, tuple):
        return np.asarray(out[0]), np.asarray(out[1])
    rho = np.asarray(out)
    return rho, np.zeros_like(rho)


def _wick_4point(rho, kappa, terms):
    r"""Reference for ``sum_t J_t <c^+_a c_b c^+_c c_d>`` via Wick's theorem.

    <c^+_a c_b c^+_c c_d> = rho_ab rho_cd - kappa_ac kbar_bd + rho_ad rbar_bc,
    with rbar = I - rho^T and kbar = -conj(kappa).
    """
    I = np.eye(rho.shape[0], dtype=rho.dtype)
    rbar = I - rho.T
    kbar = -np.conj(kappa)
    out = 0.0 + 0.0j
    for J, a, b, c, d in terms:
        out += J * (
            rho[a, b] * rho[c, d] - kappa[a, c] * kbar[b, d] + rho[a, d] * rbar[b, c]
        )
    return out


def _four_point_op(rng, L, n_terms=8, cplx=False):
    """Random operator ``sum J c^+_a c_b c^+_c c_d`` and its (J, a, b, c, d) terms."""
    terms = []
    strength, indices = [], []
    for _ in range(n_terms):
        a, b, c, d = (int(i) for i in rng.integers(0, L, size=4))
        J = rng.standard_normal()
        if cplx:
            J = J + 1j * rng.standard_normal()
        terms.append((J, a, b, c, d))
        strength.append(J)
        indices.append([a, b, c, d])
    return Operator([OpTerm("+-+-", strength, indices)]), terms


# --------------------------------------------------------------------------- #
# determinant states vs the exact dense expectation
# --------------------------------------------------------------------------- #
def test_generaldet_matches_exact_dense(x64):
    use_dtype(jnp.float64)
    L, N = 6, 3
    _spinless_chain(L, N)
    rng = np.random.default_rng(0)
    U = jnp.asarray(rng.standard_normal((L, N)))
    state = GeneralDetState(GeneralDet(U=U))
    H = tV(V=2.0, t=1.0)
    mf = complex(state.mf_expectation(H))
    exact = complex(_exact_dense(state, H))
    assert np.isclose(mf, exact, atol=1e-9)
    assert abs(mf.imag) < 1e-9  # Hermitian H -> real


def test_restricteddet_matches_exact_dense(x64):
    use_dtype(jnp.float64)
    L, Nud = 4, 2
    Chain(
        L,
        boundary=1,
        particle_type=qtx.PARTICLE_TYPE.spinful_fermion,
        Nparticles=(Nud, Nud),
    )
    rng = np.random.default_rng(1)
    U = jnp.asarray(rng.standard_normal((L, Nud)))
    state = RestrictedDetState(RestrictedDet(U=U))
    H = Hubbard(U=4.0, t=1.0)
    mf = complex(state.mf_expectation(H))
    exact = complex(_exact_dense(state, H))
    assert np.isclose(mf, exact, atol=1e-9)


def test_unrestricteddet_matches_exact_dense(x64):
    use_dtype(jnp.float64)
    L = 4
    Nup, Ndn = 2, 1
    Chain(
        L,
        boundary=1,
        particle_type=qtx.PARTICLE_TYPE.spinful_fermion,
        Nparticles=(Nup, Ndn),
    )
    rng = np.random.default_rng(2)
    Uup = jnp.asarray(rng.standard_normal((L, Nup)))
    Udn = jnp.asarray(rng.standard_normal((L, Ndn)))
    state = UnrestrictedDetState(UnrestrictedDet(U=(Uup, Udn)))
    H = Hubbard(U=4.0, t=1.0)
    mf = complex(state.mf_expectation(H))
    exact = complex(_exact_dense(state, H))
    assert np.isclose(mf, exact, atol=1e-9)


# --------------------------------------------------------------------------- #
# Wick four-point contraction vs explicit textbook formula (det and pfaffian)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("paired", [False, True], ids=["det", "pfaffian"])
def test_four_point_matches_explicit_wick(paired, x64):
    use_dtype(jnp.complex128)
    L, N = 6, 3
    _spinless_chain(L, N)
    rng = np.random.default_rng(10)
    if paired:
        F = jnp.asarray(_antisym(rng, L, np.complex128))
        state = GeneralPfState(GeneralPf(F=F))
    else:
        U = jnp.asarray(rng.standard_normal((L, N)) + 1j * rng.standard_normal((L, N)))
        state = GeneralDetState(GeneralDet(U=U))

    op, terms = _four_point_op(rng, L, n_terms=10, cplx=True)
    rho, kappa = _rho_kappa(state)
    mf = complex(state.mf_expectation(op))
    ref = complex(_wick_4point(rho, kappa, terms))
    assert np.isclose(mf, ref, atol=1e-9)


def test_two_point_matches_density_matrices(x64):
    """<c^+_i c_j> == rho_ij and <c^+_i c^+_j> == kappa_ij for a paired state."""
    use_dtype(jnp.complex128)
    L, N = 6, 3
    _spinless_chain(L, N)
    rng = np.random.default_rng(11)
    F = jnp.asarray(_antisym(rng, L, np.complex128))
    state = GeneralPfState(GeneralPf(F=F))
    rho, kappa = _rho_kappa(state)
    i, j = 1, 4
    cdc = complex(state.mf_expectation(Operator([OpTerm("+-", [1.0], [[i, j]])])))
    cdcd = complex(state.mf_expectation(Operator([OpTerm("++", [1.0], [[i, j]])])))
    assert np.isclose(cdc, complex(rho[i, j]), atol=1e-9)
    assert np.isclose(cdcd, complex(kappa[i, j]), atol=1e-9)


def test_singletpair_four_point_matches_explicit_wick(x64):
    """SingletPair is a spinful paired state with its own ``rho_from_model``."""
    use_dtype(jnp.complex128)
    L, N = 4, 2
    Chain(
        L,
        boundary=1,
        particle_type=qtx.PARTICLE_TYPE.spinful_fermion,
        Nparticles=(N, N),
    )
    rng = np.random.default_rng(12)
    F = jnp.asarray(rng.standard_normal((L, L)) + 1j * rng.standard_normal((L, L)))
    state = SingletPairState(SingletPair(F=F))
    rho, kappa = _rho_kappa(state)
    op, terms = _four_point_op(rng, 2 * L, n_terms=10, cplx=True)  # 2L spinful modes
    mf = complex(state.mf_expectation(op))
    ref = complex(_wick_4point(rho, kappa, terms))
    assert np.isclose(mf, ref, atol=1e-9)


def test_determinant_particle_number(x64):
    """The trace of rho (= sum_i <n_i>) equals the fixed particle number."""
    use_dtype(jnp.float64)
    L, N = 6, 3
    _spinless_chain(L, N)
    rng = np.random.default_rng(13)
    U = jnp.asarray(rng.standard_normal((L, N)))
    state = GeneralDetState(GeneralDet(U=U))
    opN = sum(number(i) for i in range(L))
    assert np.isclose(complex(state.mf_expectation(opN)).real, N, atol=1e-9)


# --------------------------------------------------------------------------- #
# multi-component states reduce to their single-component limits
# (regression guard for the strength/indices contraction arguments)
# --------------------------------------------------------------------------- #
def test_multidet_reduces_to_generaldet(x64):
    use_dtype(jnp.float64)
    L, N = 6, 3
    _spinless_chain(L, N)
    rng = np.random.default_rng(20)
    U = jnp.asarray(rng.standard_normal((L, N)))
    H = tV(V=1.5, t=1.0)
    e_single = complex(GeneralDetState(GeneralDet(U=U)).mf_expectation(H))
    e_multi = complex(MultiDetState(MultiDet(ndets=1, U=U)).mf_expectation(H))
    assert np.isclose(e_single, e_multi, atol=1e-9)


def test_multidet_matches_exact_dense(x64):
    use_dtype(jnp.float64)
    L, N = 6, 3
    _spinless_chain(L, N)
    rng = np.random.default_rng(21)
    U = jnp.asarray(rng.standard_normal((3, L, N)))
    state = MultiDetState(MultiDet(ndets=3, U=U))
    H = tV(V=2.0, t=1.0)
    mf = complex(state.mf_expectation(H))
    exact = complex(_exact_dense(state, H))
    assert np.isclose(mf, exact, atol=1e-8)


def test_multipf_reduces_to_generalpf(x64):
    use_dtype(jnp.float64)
    L, N = 6, 3
    _spinless_chain(L, N)
    rng = np.random.default_rng(22)
    F = _antisym(rng, L, np.float64)
    H = tV(V=1.5, t=1.0)
    e_single = complex(GeneralPfState(GeneralPf(F=jnp.asarray(F))).mf_expectation(H))
    e_multi = complex(
        MultiPfState(MultiPf(npfs=1, F=jnp.asarray(F[None]))).mf_expectation(H)
    )
    assert np.isclose(e_single, e_multi, atol=1e-9)


def test_multipf_duplicate_blocks_reduce_to_generalpf(x64):
    """Two identical Pfaffian blocks must give the single-Pfaffian value.

    This exercises the off-diagonal transition (cross-pair) contractions that the
    single-component limit cannot reach.
    """
    use_dtype(jnp.float64)
    L, N = 6, 3
    _spinless_chain(L, N)
    rng = np.random.default_rng(23)
    F = _antisym(rng, L, np.float64)
    H = tV(V=1.5, t=1.0)
    e_single = complex(GeneralPfState(GeneralPf(F=jnp.asarray(F))).mf_expectation(H))
    Fstack = jnp.asarray(np.stack([F, F], axis=0))
    e_multi = complex(MultiPfState(MultiPf(npfs=2, F=Fstack)).mf_expectation(H))
    assert np.isclose(e_single, e_multi, atol=1e-9)


def test_multidet_duplicate_blocks_reduce_to_generaldet(x64):
    use_dtype(jnp.float64)
    L, N = 6, 3
    _spinless_chain(L, N)
    rng = np.random.default_rng(24)
    U = rng.standard_normal((L, N))
    H = tV(V=1.5, t=1.0)
    e_single = complex(GeneralDetState(GeneralDet(U=jnp.asarray(U))).mf_expectation(H))
    Ustack = jnp.asarray(np.stack([U, U], axis=0))
    e_multi = complex(MultiDetState(MultiDet(ndets=2, U=Ustack)).mf_expectation(H))
    assert np.isclose(e_single, e_multi, atol=1e-9)


# --------------------------------------------------------------------------- #
# spin systems go through the Abrikosov-fermion reformatting path
# --------------------------------------------------------------------------- #
def test_spin_operator_path_multidet_matches_generaldet(x64):
    use_dtype(jnp.float64)
    L = 6
    Chain(L, boundary=1)  # spin-1/2 chain
    rng = np.random.default_rng(30)
    U = jnp.asarray(rng.standard_normal((2 * L, L)))  # 2N Abrikosov modes, half filling
    H = Heisenberg(J=1.0)
    e_single = complex(GeneralDetState(GeneralDet(U=U)).mf_expectation(H))
    e_multi = complex(MultiDetState(MultiDet(ndets=1, U=U)).mf_expectation(H))
    assert np.isclose(e_single, e_multi, atol=1e-9)
    assert abs(e_single.imag) < 1e-9


# --------------------------------------------------------------------------- #
# optimization path: gradient of the energy is finite
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "make",
    [
        lambda: GeneralDetState(GeneralDet()),
        lambda: MultiDetState(MultiDet(ndets=3)),
        lambda: GeneralPfState(GeneralPf()),
        lambda: MultiPfState(MultiPf(npfs=2)),
    ],
    ids=["generaldet", "multidet", "generalpf", "multipf"],
)
def test_get_step_returns_finite_gradient(make, x64):
    use_dtype(jnp.float64)
    _spinless_chain(6, 3)
    state = make()
    H = tV(V=1.0, t=1.0)
    g = state.get_step(H)
    assert g.shape == (state.nparams,)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert state.energy is not None and np.isfinite(complex(state.energy).real)


# --------------------------------------------------------------------------- #
# input validation and abstract-base behaviour
# --------------------------------------------------------------------------- #
def test_base_rho_from_model_not_implemented(x64):
    use_dtype(jnp.float64)
    _spinless_chain(4, 2)
    with pytest.raises(NotImplementedError):
        MeanFieldFermionState.rho_from_model(jnp.zeros((4, 4)))


def test_base_state_requires_model(x64):
    use_dtype(jnp.float64)
    _spinless_chain(4, 2)
    with pytest.raises(NotImplementedError):
        MeanFieldFermionState()


@pytest.mark.parametrize(
    "StateCls, wrong_model",
    [
        (GeneralDetState, lambda: GeneralPf()),
        (MultiDetState, lambda: GeneralDet()),
        (GeneralPfState, lambda: GeneralDet()),
        (MultiPfState, lambda: GeneralPf()),
    ],
    ids=["generaldet", "multidet", "generalpf", "multipf"],
)
def test_check_model_rejects_wrong_type(StateCls, wrong_model, x64):
    use_dtype(jnp.float64)
    _spinless_chain(6, 3)
    with pytest.raises(ValueError):
        StateCls(wrong_model())


def test_is_paired_flags(x64):
    use_dtype(jnp.float64)
    _spinless_chain(6, 3)
    assert GeneralDetState(GeneralDet()).is_paired() is False
    assert MultiDetState(MultiDet(ndets=2)).is_paired() is False
    assert GeneralPfState(GeneralPf()).is_paired() is True
    assert MultiPfState(MultiPf(npfs=2)).is_paired() is True
