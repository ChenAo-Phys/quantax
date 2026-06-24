import numpy as np
import jax.numpy as jnp
import pytest
from quantax.sites import Chain, Grid, Triangular, Cube, Pyrochlore
from quantax.nn import compute_sign, marshall_sign, stripe_sign, neel120_phase
from quantax.utils import neel, stripe

# ---------- compute_sign ----------


def test_compute_sign_outputs():
    kernel = jnp.array([1.0, 2.0, 3.0])
    s = jnp.array([1.0, -1.0, 1.0])
    phase = float(jnp.dot(kernel, s))  # 1 - 2 + 3 = 2
    assert np.isclose(compute_sign(kernel, s, "cos"), np.cos(phase))
    assert np.isclose(compute_sign(kernel, s, "phase"), np.exp(1j * phase))
    assert np.isclose(compute_sign(kernel, s, "sign"), np.sign(np.cos(phase)))


def test_compute_sign_flattens_inputs():
    # kernel and s with different shapes but matching size are flattened.
    kernel = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    s = jnp.array([1.0, -1.0, 1.0, -1.0])
    expected = float(jnp.dot(kernel.flatten(), s))
    assert np.isclose(compute_sign(kernel, s, "cos"), np.cos(expected))


def test_compute_sign_neg():
    kernel = jnp.array([0.3, 0.7])
    s = jnp.array([1.0, 1.0])
    base = compute_sign(kernel, s, "cos")
    assert np.isclose(compute_sign(kernel, s, "cos", neg=True), -base)


def test_compute_sign_bias():
    kernel = jnp.array([1.0, 1.0])
    s = jnp.array([1.0, -1.0])  # dot = 0, so the exponent is just the bias
    assert np.isclose(
        compute_sign(kernel, s, "phase", bias=jnp.pi / 2), np.exp(1j * np.pi / 2)
    )


def test_compute_sign_unknown_output_raises():
    with pytest.raises(ValueError):
        compute_sign(jnp.array([1.0]), jnp.array([1.0]), "bogus")


# ---------- marshall_sign / stripe_sign ----------


def _brute_sign(reference, s):
    """Marshall-type sign (-1)^{n_down} on the ``+1`` sublattice of ``reference``."""
    A = np.asarray(reference) > 0
    n_down = int(np.sum(np.asarray(s)[A] < 0))
    return (-1.0) ** n_down


def _random_spins(N, seed):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.choice([-1.0, 1.0], size=(20, N)).astype(np.float32))


# L = 6, 10 give an odd-sized sublattice (N_A = L/2 odd), the regression case
# where the old sign(cos(pi/4 * neel . s)) form landed on a node (cos = 0).
@pytest.mark.parametrize("L", [4, 6, 8, 10])
def test_marshall_sign_matches_definition(L):
    Chain(L)
    # Independent reference: +1 on even sites, matching neel()'s sublattice A.
    ref = np.where(np.arange(L) % 2 == 0, 1, -1)
    assert np.array_equal(np.asarray(neel()), ref)
    for s in _random_spins(L, seed=L):
        out = marshall_sign(s)
        assert out != 0  # no nodes, even for odd-sized sublattices
        assert np.isclose(out, _brute_sign(ref, s))


@pytest.mark.parametrize("extent", [(2, 2), (2, 3), (3, 3), (1, 6)])
def test_stripe_sign_matches_definition(extent):
    Grid(list(extent))
    N = int(np.prod(extent))
    ref = np.asarray(stripe(1))
    for s in _random_spins(N, seed=N):
        out = stripe_sign(s)
        assert out != 0
        assert np.isclose(out, _brute_sign(ref, s))


def test_stripe_sign_alternate_dim():
    Grid([3, 3])
    ref = np.asarray(stripe(0))
    for s in _random_spins(9, seed=0):
        out = stripe_sign(s, alternate_dim=0)
        assert np.isclose(out, _brute_sign(ref, s))


# ---------- neel120_phase ----------


def test_neel120_phase_unit_modulus():
    Triangular(3)
    N = 9
    for s in _random_spins(N, seed=7):
        out = neel120_phase(s)
        assert np.iscomplexobj(np.asarray(out))
        assert np.isclose(abs(out), 1.0)


def test_neel120_phase_matches_kernel():
    # |out|=1 holds for any phase; pin the actual 120-degree kernel value.
    Triangular(3)
    Lx, Ly = 3, 3
    x = 2 * np.arange(Lx)
    y = np.arange(Ly)
    kernel = ((x[:, None] + y[None, :]) % 3) * np.pi / 3 - np.pi / 6
    for s in _random_spins(9, seed=11):
        phase = float(np.dot(kernel.flatten(), np.asarray(s)))
        np.testing.assert_allclose(
            np.asarray(neel120_phase(s)), np.exp(1j * phase), atol=1e-5
        )


# Non-2D (Chain / Cube) or multi-site-per-cell (Pyrochlore) lattices are rejected;
# the guard fires before ``s`` is used, so its shape is irrelevant here.
@pytest.mark.parametrize(
    "make_lattice", [lambda: Chain(4), lambda: Cube(2), lambda: Pyrochlore(1)]
)
def test_neel120_phase_rejects_unsupported_lattice(make_lattice):
    make_lattice()
    with pytest.raises(ValueError):
        neel120_phase(jnp.ones(1))
