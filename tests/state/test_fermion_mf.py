"""
Tests for the fermionic mean-field states in ``quantax.state.fermion_mf``.

``mf_expectation`` evaluates :math:`\\left< O \\right>` by Wick contraction of the
mean-field one-/two-body density matrices. It is checked against the exact
expectation of the wavefunction the model represents, for random operators with 2-,
4- and 6-operator strings and constants (strings of identities only):

* (multi-)determinant states have a fixed particle number, so the mean-field value
  is the exact one in the dense particle-number sector;
* Pfaffian states are evaluated over the full Fock space (``Nparticles=None``),
  where the anomalous terms connect the sectors;
* ``SingletPair`` is pinned to the equivalent ``GeneralPf``, and spin operators to
  the explicit Abrikosov-fermion bilinears.

The exact references are :math:`\\left< \\psi|O|\\psi \\right>` of the dense
wavefunction, which the VMC estimator ``Operator.Oloc`` reproduces as well (see
`test_mf_expectation_matches_vmc_estimator`).
"""

import numpy as np
import jax
import jax.flatten_util
import jax.numpy as jnp
import equinox as eqx
import pytest
import quantax as qtx
from quantax.sites import Chain
from quantax.operator import Operator, OpTerm, tV, Hubbard
from quantax.symmetry import Identity
from quantax.utils import ints_to_array, to_distributed_array
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
    GeneralDetState,
    RestrictedDetState,
    UnrestrictedDetState,
    MultiDetState,
    GeneralPfState,
    SingletPairState,
    MultiPfState,
)
from _dtype import use_dtype
from _fermion import pfaffian


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _spinless_chain(L=6, N=3):
    return Chain(
        L, boundary=1, particle_type=qtx.PARTICLE_TYPE.spinless_fermion, Nparticles=N
    )


def _spinful_chain(L, Nparticles):
    return Chain(
        L,
        boundary=1,
        particle_type=qtx.PARTICLE_TYPE.spinful_fermion,
        Nparticles=Nparticles,
    )


def _random(rng, shape, cplx):
    x = rng.standard_normal(shape)
    return x + 1j * rng.standard_normal(shape) if cplx else x


def _antisym(rng, L, cplx):
    A = _random(rng, (L, L), cplx)
    return A - A.T


def _random_op(rng, nmodes, opstrs, n_terms=4, cplx=True, pairs=None):
    """
    ``sum_t J_t O_t`` with ``n_terms`` random index tuples per operator string. The
    creation (annihilation) operators of a term act on distinct modes, so that the
    terms don't vanish by the Pauli principle. ``pairs`` optionally restricts each
    consecutive index pair to one of the given blocks of modes (e.g. one spin
    sector), so that the term conserves them.
    """
    op_terms = []
    for opstr in opstrs:
        strength = list(_random(rng, n_terms, cplx))
        if pairs is None:
            indices = rng.integers(0, nmodes, size=(n_terms, len(opstr)))
            for c in "+-":
                pos = [k for k, o in enumerate(opstr) if o == c]
                for row in indices:
                    row[pos] = rng.choice(nmodes, len(pos), replace=False)
        else:
            block = rng.integers(0, len(pairs), size=(n_terms, len(opstr) // 2))
            block = np.repeat(block, 2, axis=1)
            sizes = np.array([len(p) for p in pairs])[block]
            pos = (rng.random((n_terms, len(opstr))) * sizes).astype(int)
            indices = np.array(
                [[pairs[b][p] for b, p in zip(br, pr)] for br, pr in zip(block, pos)]
            )
        op_terms.append(OpTerm(opstr, strength, indices.tolist()))
    return Operator(op_terms)


def _basis_spins():
    symm = Identity()
    symm.basis_make()
    return ints_to_array(symm.basis.states)


def _dense_psi(model):
    """The model's amplitudes on the basis of ``Operator.todense``."""
    spins = jnp.asarray(_basis_spins())
    return np.asarray(jax.jit(jax.vmap(lambda s: model(s).value()))(spins))


def _exact_expectation(psi, op):
    r"""
    :math:`\left< \psi|O|\psi \right> / \left< \psi|\psi \right>` of the dense
    amplitudes. Complex coefficients and anomalous terms tell it apart from
    :math:`\left< \psi|O^T|\psi \right>`.
    """
    M = np.asarray(op.todense())
    return (psi.conj() @ (M @ psi)) / (psi.conj() @ psi)


def _assert_close(x, y, atol=1e-9):
    assert np.isclose(complex(x), complex(y), atol=atol, rtol=0), (x, y)


# number-conserving strings: hopping, density and 2-body terms; "I" is padding, and
# "II" a constant, which the reformatting turns into an empty string with <> = 1
_NUMBER_CONSERVING = ["+-", "n", "I+-", "+-+-", "++--", "-+n", "II"]
# strings with anomalous terms; the pairings of "+-+-" contract to "+-", "++", "--"
# and "-+", and "+++-" has two anomalous pairs; "I" is a constant
_PAIRING = ["+-", "++", "--", "+-+-", "+++-", "I"]


# --------------------------------------------------------------------------- #
# determinant states vs the exact dense expectation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("multi", [False, True], ids=["generaldet", "multidet"])
def test_det_states_match_exact_dense(multi, x64):
    use_dtype(jnp.complex128)
    L, N = 6, 3
    _spinless_chain(L, N)
    rng = np.random.default_rng(0)
    if multi:
        # three distinct, non-orthonormal determinants: the transition density
        # matrices between different determinants enter
        U = jnp.asarray(_random(rng, (3, L, N), True))
        coeffs = jnp.asarray(_random(rng, 3, True))
        model = MultiDet(ndets=3, U=U, coeffs=coeffs)
        with pytest.warns(UserWarning, match="orthonormal"):
            state = MultiDetState(model)
    else:
        model = GeneralDet(U=jnp.asarray(_random(rng, (L, N), True)))
        state = GeneralDetState(model)
    assert not state.is_paired()

    # 3-body terms run the shared Wick recursion one level deeper; MultiDet's copy
    # of the recursion is the same code, checked here up to 2-body terms
    opstrs = _NUMBER_CONSERVING + ([] if multi else ["+-+-+-"])
    op = tV(V=2.0, t=1.0) + _random_op(rng, L, opstrs)
    # the reference is the model as given: MultiDetState orthonormalizes the
    # orbitals, which must keep the wavefunction up to its normalization
    exact = _exact_expectation(_dense_psi(model), op)
    _assert_close(state.mf_expectation(op), exact)


def test_mf_expectation_matches_vmc_estimator(x64):
    # Mean field and VMC evaluate the same <psi|O|psi>. A complex Hermitian current
    # and non-Hermitian strings tell it from <psi|O^T|psi>, which both of them
    # returned in a former convention.
    use_dtype(jnp.complex128)
    L, N = 4, 2
    _spinless_chain(L, N)
    rng = np.random.default_rng(2)
    state = GeneralDetState(GeneralDet(U=jnp.asarray(_random(rng, (L, N), True))))
    current = Operator([OpTerm("+-", [1j, -1j], [[0, 1], [1, 0]])])
    op = current + _random_op(rng, L, ["+-", "+-+-"])
    exact = _exact_expectation(_dense_psi(state.model), op)

    spins = np.tile(_basis_spins(), (jax.device_count(), 1))
    spins = to_distributed_array(jnp.asarray(spins))
    Oloc = np.asarray(op.Oloc(state, spins))  # through the ref forward path
    w = np.abs(np.asarray(jnp.asarray(state(spins)))) ** 2
    _assert_close(np.sum(w * Oloc) / np.sum(w), exact)
    _assert_close(state.mf_expectation(op), exact)


@pytest.mark.parametrize(
    "restricted, Nparticles",
    [(True, (2, 2)), (False, (2, 1))],
    ids=["restricteddet", "unrestricteddet"],
)
def test_spin_resolved_det_states_match_exact_dense(restricted, Nparticles, x64):
    # Spin-dependent terms distinguish the spin blocks of rho, to which the
    # spin-symmetric Hubbard model is blind, and unequal populations tell the
    # spin-up and spin-down orbitals of UnrestrictedDet apart.
    use_dtype(jnp.complex128)
    L = 4
    _spinful_chain(L, Nparticles)
    rng = np.random.default_rng(1)
    up, dn = list(range(L)), list(range(L, 2 * L))
    opstrs = ["+-", "+-+-", "II"]
    op = Hubbard(U=4.0, t=1.0) + _random_op(rng, 2 * L, opstrs, pairs=[up, dn])
    if restricted:
        U = jnp.asarray(_random(rng, (L, Nparticles[0]), True))
        state = RestrictedDetState(RestrictedDet(U=U))
    else:
        U = tuple(jnp.asarray(_random(rng, (L, n), True)) for n in Nparticles)
        state = UnrestrictedDetState(UnrestrictedDet(U=U))
    exact = _exact_expectation(_dense_psi(state.model), op)
    _assert_close(state.mf_expectation(op), exact)


# --------------------------------------------------------------------------- #
# Pfaffian states vs the exact Fock-space expectation
# --------------------------------------------------------------------------- #
def _fock_space_psi(F):
    """
    Amplitudes of sum_k pf(F_k[n, n]) on the full Fock space (``Nparticles=None``),
    with the QuSpin reordering sign (-1)^{N(N-1)/2} of the models: the relative sign
    between the particle-number sectors fixes the anomalous expectations. The
    models' ``__call__`` is pinned to this formula in test_fermion_mf_call.py; it
    can't be jitted here, as the particle number differs between configurations.
    """
    psi = []
    for s in _basis_spins():
        idx = np.flatnonzero(s > 0)
        sign = (-1) ** (idx.size * (idx.size - 1) // 2)
        psi.append(sign * sum(pfaffian(Fk[np.ix_(idx, idx)]) for Fk in F))
    return np.asarray(psi)


@pytest.mark.parametrize(
    "npfs, cplx",
    [(None, False), (None, True), (2, True)],
    ids=["generalpf-real", "generalpf-cplx", "multipf-cplx"],
)
def test_pf_states_match_fock_space(npfs, cplx, x64):
    # The real case of MultiPfState runs the same algebra as the complex one with
    # trivial conjugations, so only the complex one is compiled.
    use_dtype(jnp.complex128 if cplx else jnp.float64)
    L = 4
    Chain(L, boundary=1, particle_type=qtx.PARTICLE_TYPE.spinless_fermion)
    rng = np.random.default_rng(40)
    if npfs is None:
        state = GeneralPfState(GeneralPf(F=jnp.asarray(_antisym(rng, L, cplx))))
        F_full = [state.model.F_full]
    else:
        # distinct Pfaffians: the cross-pair transition contractions enter
        F = np.stack([_antisym(rng, L, cplx) for _ in range(npfs)])
        state = MultiPfState(MultiPf(npfs=npfs, F=jnp.asarray(F)))
        F_full = state.model.F_full
    assert state.is_paired()

    # odd strings vanish by parity
    opstrs = _PAIRING + ["+--"]
    op = _random_op(rng, L, opstrs, cplx=cplx)
    exact = _exact_expectation(_fock_space_psi(np.asarray(F_full)), op)
    _assert_close(state.mf_expectation(op), exact)


def test_multi_states_odd_strings_vanish(x64):
    # Odd strings vanish by parity, but the vmapped contractions of MultiDetState and
    # MultiPfState must return one zero per term: a scalar zero gave an
    # (ndets, ndets) array that failed to broadcast against the strengths (it only
    # broadcast by accident for 1 or ndets terms, hence 3 terms with 2 states).
    use_dtype(jnp.float64)
    L = 4
    _spinless_chain(L, 2)
    rng = np.random.default_rng(2)
    U = np.linalg.qr(rng.standard_normal((2, L, 2)))[0]
    F = np.stack([_antisym(rng, L, False) for _ in range(2)])
    even = _random_op(rng, L, ["+-"], cplx=False)
    odd = _random_op(rng, L, ["+--", "-"], n_terms=3, cplx=False)
    for state in [
        MultiDetState(MultiDet(ndets=2, U=jnp.asarray(U))),
        MultiPfState(MultiPf(npfs=2, F=jnp.asarray(F))),
    ]:
        _assert_close(state.mf_expectation(odd), 0.0)
        _assert_close(state.mf_expectation(even + odd), state.mf_expectation(even))


def test_singletpair_matches_equivalent_generalpf(x64):
    """SingletPair has its own rho_from_model; it must agree with the equivalent
    GeneralPf, F_pf = [[0, F], [-F^T, 0]], including anomalous and spin-flip terms."""
    use_dtype(jnp.complex128)
    L = 4
    _spinful_chain(L, (2, 2))
    rng = np.random.default_rng(41)
    F = _random(rng, (L, L), True)
    Fpf = np.block([[np.zeros((L, L)), F], [-F.T, np.zeros((L, L))]])
    sp_state = SingletPairState(SingletPair(F=jnp.asarray(F)))
    pf_state = GeneralPfState(GeneralPf(F=jnp.asarray(Fpf)))
    assert sp_state.is_paired()

    op = _random_op(rng, 2 * L, _PAIRING)
    _assert_close(sp_state.mf_expectation(op), pf_state.mf_expectation(op))


# --------------------------------------------------------------------------- #
# spin systems go through the Abrikosov-fermion reformatting path
# --------------------------------------------------------------------------- #
# S^a = sum_{st} c^+_s M^a_st c_t in the (up, down) basis, with up at site i and
# down at mode i + L; quantax's "x", "y", "z" are the spin-1/2 operators S^a
_SPIN_MATS = {
    "+": np.array([[0, 1], [0, 0]]),
    "-": np.array([[0, 0], [1, 0]]),
    "x": np.array([[0, 0.5], [0.5, 0]]),
    "y": np.array([[0, -0.5j], [0.5j, 0]]),
    "z": np.array([[0.5, 0], [0, -0.5]]),
}


def _abrikosov_wick(rho, L, op_terms):
    """<sum J S^a_i S^b_j + const> from Wick's theorem on the Abrikosov bilinears."""
    I = np.eye(2 * L)
    out = 0.0
    for op_term in op_terms:
        for J, sites in zip(op_term.strength, op_term.indices):
            ops = [(o, i) for o, i in zip(op_term.opstr, sites) if o != "I"]
            if not ops:
                out += J
                continue
            (a, i), (b, j) = ops
            A, B = _SPIN_MATS[a], _SPIN_MATS[b]
            for s, t, u, v in np.ndindex(2, 2, 2, 2):
                w, x, y, z = i + s * L, i + t * L, j + u * L, j + v * L
                # <c^+_w c_x c^+_y c_z> = rho_wx rho_yz + rho_wz (delta_xy - rho_yx)
                four = rho[w, x] * rho[y, z] + rho[w, z] * (I[x, y] - rho[y, x])
                out += J * A[s, t] * B[u, v] * four
    return out


def test_spin_operators_map_to_abrikosov_fermions(x64):
    use_dtype(jnp.complex128)
    L = 3
    Chain(L, boundary=1)  # spin-1/2 chain
    rng = np.random.default_rng(30)
    U = jnp.asarray(_random(rng, (2 * L, L), True))  # 2L Abrikosov modes, half filling
    state = GeneralDetState(GeneralDet(U=U))
    rho = np.asarray(GeneralDetState.rho_from_model(state.model))

    # two-spin terms, including same-site products and the "I" padding, and constants
    opstrs = ["+-", "-+", "zz", "xx", "yy", "xy", "zx", "+z", "y-", "Ixz", "I", "II"]
    op = _random_op(rng, L, opstrs, n_terms=3)
    _assert_close(state.mf_expectation(op), _abrikosov_wick(rho, L, op.op_list))


# --------------------------------------------------------------------------- #
# optimization path: get_step is the gradient of the mean-field energy
# --------------------------------------------------------------------------- #
def _orthonormal(rng, shape, cplx):
    return jnp.asarray(np.linalg.qr(_random(rng, shape, cplx))[0])


# state factories on the (6, 3) spinless chain, and whether they are complex
_GET_STEP_CASES = {
    "generaldet-real": (
        lambda rng: GeneralDetState(
            GeneralDet(U=jnp.asarray(_random(rng, (6, 3), False)))
        ),
        False,
    ),
    # orthonormal orbitals, so the state keeps the parameters as given
    "multidet-cplx": (
        lambda rng: MultiDetState(
            MultiDet(ndets=3, U=_orthonormal(rng, (3, 6, 3), True))
        ),
        True,
    ),
    "generalpf-real": (
        lambda rng: GeneralPfState(GeneralPf(F=jnp.asarray(_antisym(rng, 6, False)))),
        False,
    ),
    "multipf-cplx": (
        lambda rng: MultiPfState(
            MultiPf(npfs=2, F=jnp.asarray([_antisym(rng, 6, True) for _ in range(2)]))
        ),
        True,
    ),
}


@pytest.mark.parametrize("case", _GET_STEP_CASES)
def test_get_step_is_the_energy_gradient(case, x64):
    # update(step) moves the parameters to theta - step, so get_step must be the
    # ascent direction: the energy changes by eps |g| along g / |g|. For complex
    # parameters this pins the conjugation of the JAX gradient. get_step is shared by
    # all mean-field states, but each state differentiates its own expectation.
    make, cplx = _GET_STEP_CASES[case]
    use_dtype(jnp.complex128 if cplx else jnp.float64)
    _spinless_chain(6, 3)
    rng = np.random.default_rng(50)
    state = make(rng)
    # random hopping, plus pairing for Pfaffians: a quadratic operator keeps the
    # compilation of the gradient cheap
    opstrs = ["+-", "++"] if state.is_paired() else ["+-"]
    H = _random_op(rng, 6, opstrs, n_terms=8, cplx=cplx)
    g = state.get_step(H)
    assert g.shape == (state.nparams,)
    assert np.iscomplexobj(g) == cplx

    loss_fn = state.get_loss_fn(H)
    params = eqx.filter(state.model, eqx.is_inexact_array)
    _assert_close(loss_fn(params), complex(state.energy).real)

    flat, unravel = jax.flatten_util.ravel_pytree(params)
    eps = 1e-5
    v = g / jnp.linalg.norm(g)
    slope = (loss_fn(unravel(flat + eps * v)) - loss_fn(unravel(flat - eps * v))) / (
        2 * eps
    )
    np.testing.assert_allclose(slope, float(jnp.linalg.norm(g)), rtol=1e-5)

    # a constant, i.e. a term of identities only, shifts the energy but not the step
    E = complex(state.energy).real
    const = Operator([OpTerm("II", [2.0, -0.5], [[0, 1], [3, 3]])])
    g_const = state.get_step(H + const)
    _assert_close(state.energy, E + 1.5)
    atol = 1e-12 * float(jnp.linalg.norm(g))
    np.testing.assert_allclose(g_const, g, rtol=0, atol=atol)


@pytest.mark.parametrize("fermion", [True, False], ids=["fermion-z", "spin-n"])
def test_unsupported_site_operators_raise(fermion):
    # The Wick contraction can't take a fermion "z" = n - 1/2, which has even parity
    # but odd length and was read as 0; "n" isn't a spin operator and was dropped,
    # i.e. read as an identity.
    if fermion:
        _spinless_chain(4, 2)
        U = jnp.eye(4)[:, :2]
        opstr = "z"
    else:
        Chain(4, boundary=1)
        U = jnp.eye(8)[:, :4]
        opstr = "n"
    state = GeneralDetState(GeneralDet(U=U))
    with pytest.raises(NotImplementedError, match=f"'{opstr}'"):
        state.mf_expectation(Operator([OpTerm(opstr, [1.0], [[0]])]))


# --------------------------------------------------------------------------- #
# input validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "StateCls, wrong_model",
    [
        (GeneralDetState, lambda: GeneralPf(F=jnp.zeros((6, 6)))),
        (MultiDetState, lambda: GeneralDet(U=jnp.zeros((6, 3)))),
        (GeneralPfState, lambda: GeneralDet(U=jnp.zeros((6, 3)))),
        (MultiPfState, lambda: GeneralPf(F=jnp.zeros((6, 6)))),
    ],
    ids=["generaldet", "multidet", "generalpf", "multipf"],
)
def test_check_model_rejects_wrong_type(StateCls, wrong_model):
    _spinless_chain(6, 3)
    with pytest.raises(ValueError, match="must be an instance of"):
        StateCls(wrong_model())


def test_default_model():
    # Without a model the states build the default one of their family, whose
    # orbitals are checked in tests/model/test_fermion_mf_init.py on this lattice.
    # The default MultiDet and MultiPf have 4 components, which would compile a
    # construction of their own.
    _spinless_chain(6, 3)
    assert type(GeneralDetState().model) is GeneralDet
    assert type(GeneralPfState().model) is GeneralPf
