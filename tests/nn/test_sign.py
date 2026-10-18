import itertools
import numpy as np
import jax
import pytest
from quantax.sites import Chain, Grid, Triangular, TriangularB, Cube, Pyrochlore
from quantax.nn import compute_sign, marshall_sign, stripe_sign, neel120_phase

# Functions are evaluated jitted and vmapped over many configurations; eager
# evaluation compiles every primitive separately and dominates the test time.


def _all_spins(N):
    return np.array(list(itertools.product([1.0, -1.0], repeat=N)), np.float32)


# ---------- compute_sign ----------


def test_compute_sign_outputs():
    # The exponent is kernel . s + bias with both flattened; neg flips the output.
    kernel = np.array([[0.3, 1.2], [-0.7, 2.0]], np.float32)
    s = np.array([1.0, -1.0, -1.0, 1.0], np.float32)
    phase = 0.3 - 1.2 + 0.7 + 2.0 + 0.4  # cos(phase) < 0

    base = {"sign": -1.0, "phase": np.exp(1j * phase), "cos": np.cos(phase)}
    flips = (False, True)

    @jax.jit
    def f(kernel, s):
        return [compute_sign(kernel, s, o, neg, 0.4) for o in base for neg in flips]

    expected = [v * (-1) ** neg for v in base.values() for neg in flips]
    np.testing.assert_allclose(np.array(f(kernel, s)), expected, rtol=1e-6)


def test_compute_sign_unknown_output_raises():
    with pytest.raises(ValueError):
        jax.jit(lambda k, s: compute_sign(k, s, "bogus"))(np.ones(1), np.ones(1))


# ---------- marshall_sign / stripe_sign ----------


def _sublattice_sign(spins, A):
    """(-1)^{n_down} with n_down the number of down spins on the sites ``A``."""
    return (-1.0) ** np.sum(spins[:, A] < 0, axis=1)


# Grid([2, 3]) has an odd-sized sublattice (N_A = 3), the regression case where the
# old sign(cos(pi/4 * neel . s)) form landed on a node (cos = 0).
@pytest.mark.parametrize("extent", [(4,), (2, 3)], ids=["chain4", "grid2x3"])
def test_marshall_sign_matches_definition(extent):
    Grid(list(extent))
    # Independent reference: sublattice A has an even coordinate sum, with sites
    # ordered row-major over the grid.
    A = np.indices(extent).sum(axis=0).ravel() % 2 == 0
    s = _all_spins(A.size)
    out = np.asarray(jax.jit(jax.vmap(marshall_sign))(s))
    np.testing.assert_array_equal(out, _sublattice_sign(s, A))


def test_stripe_sign_matches_definition():
    # The spins alternate along alternate_dim (default 1): N_A = 3 (odd) along the
    # y axis of size 2, and N_A = 4 along the x axis of size 3.
    Grid([3, 2])
    s = _all_spins(6)

    @jax.jit
    @jax.vmap
    def f(s):
        return stripe_sign(s), stripe_sign(s, alternate_dim=0)

    out_default, out_x = f(s)
    coords = np.indices((3, 2)).reshape(2, -1)
    np.testing.assert_array_equal(out_default, _sublattice_sign(s, coords[1] % 2 == 0))
    np.testing.assert_array_equal(out_x, _sublattice_sign(s, coords[0] % 2 == 0))


# ---------- neel120_phase ----------


_W = np.exp(1j * np.pi / 3)


# psi(all up) and w_j of the first four sites (row-major over (x, y)) pin the global
# angle and the chirality of the 120 degree order, which a gauge-blind check misses.
@pytest.mark.parametrize(
    "make_lattice, psi_up, w_head",
    [
        (lambda: Triangular(3), -1j, [_W, np.conj(_W), -1, -1]),
        (lambda: TriangularB(2), 1, [_W, _W, -1, -1]),
    ],
    ids=["triangular", "triangularB"],
)
def test_neel120_phase_is_120_degree_product_state(make_lattice, psi_up, w_head):
    # The amplitudes of the coplanar product state prod_j (|up> + w_j |dn>) up to a
    # constant, where w_j = exp(i phi_j) and phi_j is the angle of sublattice j:
    # nearest neighbors have angles 120 degrees apart.
    lattice = make_lattice()
    N = lattice.Nsites
    up = np.ones((1, N), np.float32)
    flips = up - 2 * np.eye(N, dtype=np.float32)
    rng = np.random.default_rng(0)
    s = rng.choice([-1.0, 1.0], size=(64, N)).astype(np.float32)
    psi = np.asarray(jax.jit(jax.vmap(neel120_phase))(np.concatenate([up, flips, s])))
    w = psi[1 : N + 1] / psi[0]
    np.testing.assert_allclose(np.abs(w), 1.0, rtol=1e-5)
    np.testing.assert_allclose(psi[0], psi_up, atol=1e-6)
    np.testing.assert_allclose(w[:4], w_head, atol=1e-6)

    expected = psi[0] * np.prod(np.where(s < 0, w, 1.0), axis=1)
    np.testing.assert_allclose(psi[N + 1 :], expected, atol=1e-5)
    i, j = lattice.get_neighbor(1).T
    assert i.size == 3 * N
    np.testing.assert_allclose(np.real(w[i] * np.conj(w[j])), -0.5, atol=1e-5)


# Non-2D (Chain / Cube) or multi-site-per-cell (Pyrochlore) lattices are rejected;
# the guard fires before ``s`` is used, so its shape is irrelevant here.
@pytest.mark.parametrize(
    "make_lattice",
    [lambda: Chain(4), lambda: Cube(2), lambda: Pyrochlore(1)],
    ids=["chain", "cube", "pyrochlore"],
)
def test_neel120_phase_rejects_unsupported_lattice(make_lattice):
    make_lattice()
    with pytest.raises(ValueError):
        neel120_phase(np.ones(1))
