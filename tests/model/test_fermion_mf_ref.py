"""
Tests of the accelerated ``ref_forward`` low-rank updates of mean-field models,
which must agree with the direct ``__call__`` evaluation along Markov chains.
"""

import numpy as np
import jax.numpy as jnp
import equinox as eqx
import pytest
import quantax as qtx
from quantax.model import GeneralDet, GeneralPf, SingletPair
from quantax.sites import Sites

from _dtype import use_dtype


def _random_spin_update(rng, s_np, k):
    """Exchange k up-sites with k down-sites."""
    ups = np.flatnonzero(s_np > 0)
    dns = np.flatnonzero(s_np < 0)
    s_new = s_np.copy()
    s_new[rng.choice(ups, size=k, replace=False)] = -1
    s_new[rng.choice(dns, size=k, replace=False)] = 1
    return s_new


def _random_spin_flips(rng, s_np, k):
    """Flip k random spins, which changes Sz unless the flips cancel."""
    s_new = s_np.copy()
    s_new[rng.choice(s_np.size, size=k, replace=False)] *= -1
    return s_new


def _random_fermion_hop(rng, s_np, sector, N, k=1):
    """Hop k fermions within one spin sector (0=up, 1=dn)."""
    out = s_np.copy()
    lo, hi = sector * N, (sector + 1) * N
    occ = lo + np.flatnonzero(out[lo:hi] > 0)
    emp = lo + np.flatnonzero(out[lo:hi] < 0)
    out[rng.choice(occ, size=k, replace=False)] = -1
    out[rng.choice(emp, size=k, replace=False)] = 1
    return out


def _random_model(model_cls, rng):
    """A model with generic random parameters (the default init is not under test)."""
    sites = qtx.get_sites()
    if model_cls is GeneralDet:
        U = rng.standard_normal((sites.Nfmodes, sites.Ntotal))
        return GeneralDet(U=jnp.asarray(U))
    if model_cls is GeneralPf:
        F = rng.standard_normal((sites.Nfmodes, sites.Nfmodes))
        return GeneralPf(F=jnp.asarray(F - F.T))
    return SingletPair(F=jnp.asarray(rng.standard_normal((sites.Nsites,) * 2)))


class _Chain:
    """
    A Markov chain of ``ref_forward`` updates checked against ``__call__``.

    Every step compares the probe (``return_update=False``), the same probe with the
    padded upper-bound mode, and the chain update with the direct evaluation of the
    new configuration. The evaluations are jitted, as eager ``ref_forward`` calls
    would compile every primitive separately, which is several times slower. Only the
    probe and the chain update depend on the update mode; the padded probe and the
    direct evaluation are compiled once per chain. The jitted functions are fresh
    closures, so no trace outlives the test's lattice.

    The stored inverse carries the rounding errors of all previous updates, which
    grow as the inverse of the smallest amplitude visited. So the chain, like a
    Metropolis chain, rejects moves to configurations near a node (2-5% of them),
    and the tolerance is relative to the largest amplitude along the chain. The
    errors then stayed below 3e-11 of it over 100 seeds of each chain; accepting
    every move gave up to 5e-9, and a wrong update gives O(1).
    """

    def __init__(self, model, s, pad_mode):
        self.model = model
        self._steps = {}
        self._pad = eqx.filter_jit(
            lambda m, s_new, s, internal: m.ref_forward(s_new, s, pad_mode, internal)
        )
        self._call = eqx.filter_jit(lambda m, s: m(s))
        psi, self.internal = eqx.filter_jit(lambda m, s: m.init_internal(s))(model, s)
        self.scale = 0.0
        self._assert_close(self._call(model, s), psi)
        self.s = s

    def _assert_close(self, direct, *psi_ref):
        direct = complex(direct.value())
        self.scale = max(self.scale, abs(direct))
        for psi in psi_ref:
            np.testing.assert_allclose(
                complex(psi.value()), direct, rtol=0, atol=1e-9 * self.scale
            )

    def _get_step(self, mode):
        key = tuple(sorted(mode.items()))
        if key not in self._steps:

            @eqx.filter_jit
            def step(model, s_new, s, internal):
                probe = model.ref_forward(s_new, s, mode, internal)
                chain, internal = model.ref_forward(s_new, s, mode, internal, True)
                return probe, chain, internal

            self._steps[key] = step
        return self._steps[key]

    def step(self, s_new, mode):
        s_new = jnp.asarray(s_new)
        direct = self._call(self.model, s_new)
        # update modes are upper bounds: padded hop slots must be no-ops
        pad = self._pad(self.model, s_new, self.s, self.internal)
        step = self._get_step(mode)
        probe, chain, internal = step(self.model, s_new, self.s, self.internal)
        self._assert_close(direct, probe, pad, chain)
        # reject moves to near-node configurations, see above
        if abs(complex(direct.value())) > 1e-2 * self.scale:
            self.s, self.internal = s_new, internal


def test_singletpair_spin_ref_forward_matches_call(x64):
    use_dtype(jnp.float64)
    Sites(6, particle_type="spin", Nparticles=(3, 3))
    rng = np.random.default_rng(0)
    model = _random_model(SingletPair, rng)
    assert model.required_update_modes == ("nflips",)

    # n(n-1)/2 is odd for n=3, exercising the determinant reordering sign
    s = jnp.array([1, 1, 1, -1, -1, -1], dtype=jnp.float64)
    chain = _Chain(model, s, {"nflips": 4})
    for _ in range(30):
        k = int(rng.choice([1, 2]))
        chain.step(_random_spin_update(rng, np.asarray(chain.s), k), {"nflips": 2 * k})


def test_singletpair_fermion_ref_forward_matches_call(x64):
    use_dtype(jnp.float64)
    N = 6
    Sites(N, particle_type="spinful_fermion", Nparticles=(3, 3))
    rng = np.random.default_rng(1)
    model = _random_model(SingletPair, rng)
    assert model.required_update_modes == ("nflips_up", "nflips_dn")

    s_np = -np.ones(2 * N)
    s_np[rng.choice(N, 3, replace=False)] = 1
    s_np[N + rng.choice(N, 3, replace=False)] = 1

    # the exact per-sector modes, as given by ParticleHopUp/Dn and by the
    # nflips_up_dn_filter of Operator.Oloc, with empty row or column updates
    updates = {
        "up": (lambda x: _random_fermion_hop(rng, x, 0, N), (2, 0)),
        "dn": (lambda x: _random_fermion_hop(rng, x, 1, N), (0, 2)),
        "both": (
            lambda x: _random_fermion_hop(rng, _random_fermion_hop(rng, x, 0, N), 1, N),
            (2, 2),
        ),
        "up2": (lambda x: _random_fermion_hop(rng, x, 0, N, k=2), (4, 0)),
    }
    # update modes are upper bounds, as provided by ParticleHop
    chain = _Chain(model, jnp.asarray(s_np), {"nflips_up": 4, "nflips_dn": 4})
    for _ in range(10):
        for fn, (nflips_up, nflips_dn) in updates.values():
            mode = {"nflips_up": nflips_up, "nflips_dn": nflips_dn}
            chain.step(fn(np.asarray(chain.s)), mode)


@pytest.mark.parametrize("model_cls", [GeneralDet, GeneralPf])
def test_spin_multihop_chain_matches_call(model_cls, x64):
    # Multi-hop updates (nflips=4) must stay exact after the internal index
    # list becomes unsorted from previous in-place updates.
    use_dtype(jnp.float64)
    Sites(8, particle_type="spin", Nparticles=(4, 4))
    rng = np.random.default_rng(2)
    model = _random_model(model_cls, rng)
    assert model.required_update_modes == ("nflips",)

    s = jnp.asarray(np.array([1] * 4 + [-1] * 4, dtype=np.float64))
    # MixSampler merges the update modes of its samplers into their maximum
    chain = _Chain(model, s, {"nflips": 6})
    for _ in range(30):
        k = int(rng.choice([1, 2]))
        chain.step(_random_spin_update(rng, np.asarray(chain.s), k), {"nflips": 2 * k})


@pytest.mark.parametrize("model_cls", [GeneralDet, GeneralPf])
def test_spin_flip_chain_matches_call(model_cls, x64):
    # Without Sz conservation, spin flips (LocalFlip, sigma_x terms in Oloc) change
    # the number of up spins, and with it the fermion_inverse_sign of __call__.
    use_dtype(jnp.float64)
    Sites(6, particle_type="spin")
    rng = np.random.default_rng(3)
    model = _random_model(model_cls, rng)

    s = jnp.asarray(np.array([1, 1, 1, -1, -1, -1], dtype=np.float64))
    chain = _Chain(model, s, {"nflips": 3})
    for _ in range(30):
        k = int(rng.choice([1, 2]))
        chain.step(_random_spin_flips(rng, np.asarray(chain.s), k), {"nflips": k})
