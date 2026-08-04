"""
Tests of the accelerated ``ref_forward`` low-rank updates of mean-field models,
which must agree with the direct ``__call__`` evaluation along Markov chains.
"""

import numpy as np
import jax.numpy as jnp
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


def _random_fermion_hop(rng, s_np, sector, N, k=1):
    """Hop k fermions within one spin sector (0=up, 1=dn)."""
    out = s_np.copy()
    lo, hi = sector * N, (sector + 1) * N
    occ = lo + np.flatnonzero(out[lo:hi] > 0)
    emp = lo + np.flatnonzero(out[lo:hi] < 0)
    out[rng.choice(occ, size=k, replace=False)] = -1
    out[rng.choice(emp, size=k, replace=False)] = 1
    return out


def _assert_close(psi_ref, psi_direct):
    np.testing.assert_allclose(
        complex(psi_ref.value()), complex(psi_direct.value()), rtol=1e-8
    )


def test_singletpair_spin_ref_forward_matches_call(x64):
    use_dtype(jnp.float64)
    Sites(6, particle_type="spin", Nparticles=(3, 3))
    model = SingletPair()
    rng = np.random.default_rng(0)

    s = jnp.array([1, 1, 1, -1, -1, -1], dtype=jnp.float64)
    psi, internal = model.init_internal(s)
    # n(n-1)/2 is odd for n=3, exercising the determinant reordering sign
    _assert_close(psi, model(s))

    for _ in range(30):
        k = int(rng.choice([1, 2]))
        s_new = jnp.asarray(_random_spin_update(rng, np.asarray(s), k))
        psi_probe = model.ref_forward(s_new, s, {"nflips": 2 * k}, internal)
        # nflips is an upper bound: padded slots must be no-ops
        psi_pad = model.ref_forward(s_new, s, {"nflips": 4}, internal)
        psi_chain, internal = model.ref_forward(
            s_new, s, {"nflips": 2 * k}, internal, True
        )
        psi_direct = model(s_new)
        _assert_close(psi_probe, psi_direct)
        _assert_close(psi_pad, psi_direct)
        _assert_close(psi_chain, psi_direct)
        s = s_new


def test_singletpair_fermion_ref_forward_matches_call(x64):
    use_dtype(jnp.float64)
    N = 6
    sites = Sites(N, particle_type="spinful_fermion", Nparticles=(3, 3))
    model = SingletPair()
    assert model.required_update_modes == ("nflips_up", "nflips_dn")

    rng = np.random.default_rng(1)
    s_np = -np.ones(2 * N)
    s_np[rng.choice(N, 3, replace=False)] = 1
    s_np[N + rng.choice(N, 3, replace=False)] = 1
    s = jnp.asarray(s_np)

    psi, internal = model.init_internal(s)
    _assert_close(psi, model(s))

    updates = {
        "up": (lambda x: _random_fermion_hop(rng, x, 0, N), (2, 0)),
        "dn": (lambda x: _random_fermion_hop(rng, x, 1, N), (0, 2)),
        "both": (
            lambda x: _random_fermion_hop(rng, _random_fermion_hop(rng, x, 0, N), 1, N),
            (2, 2),
        ),
        "up2": (lambda x: _random_fermion_hop(rng, x, 0, N, k=2), (4, 0)),
    }
    for _ in range(10):
        for fn, (nflips_up, nflips_dn) in updates.values():
            s_new = jnp.asarray(fn(np.asarray(s)))
            mode = {"nflips_up": nflips_up, "nflips_dn": nflips_dn}
            psi_probe = model.ref_forward(s_new, s, mode, internal)
            # update modes are upper bounds, as provided by ParticleHop
            psi_pad = model.ref_forward(
                s_new, s, {"nflips_up": 4, "nflips_dn": 4}, internal
            )
            psi_chain, internal = model.ref_forward(s_new, s, mode, internal, True)
            psi_direct = model(s_new)
            _assert_close(psi_probe, psi_direct)
            _assert_close(psi_pad, psi_direct)
            _assert_close(psi_chain, psi_direct)
            s = s_new


@pytest.mark.parametrize("model_cls", [GeneralDet, GeneralPf])
def test_spin_multihop_chain_matches_call(model_cls, x64):
    # Multi-hop updates (nflips=4) must stay exact after the internal index
    # list becomes unsorted from previous in-place updates.
    use_dtype(jnp.float64)
    Sites(8, particle_type="spin", Nparticles=(4, 4))
    model = model_cls()
    rng = np.random.default_rng(2)

    s = jnp.asarray(np.array([1] * 4 + [-1] * 4, dtype=np.float64))
    _, internal = model.init_internal(s)
    for _ in range(30):
        k = int(rng.choice([1, 2]))
        s_new = jnp.asarray(_random_spin_update(rng, np.asarray(s), k))
        psi_chain, internal = model.ref_forward(
            s_new, s, {"nflips": 2 * k}, internal, True
        )
        _assert_close(psi_chain, model(s_new))
        s = s_new


def test_singletpair_fermion_sampler_and_oloc(x64):
    use_dtype(jnp.float64)
    from quantax.state import Variational, DenseState
    from quantax.sampler import ParticleHopUp, ParticleHopDn, MixSampler
    from quantax.operator import Hubbard
    from quantax.utils import array_to_ints

    qtx.sites.Square(2, particle_type="spinful_fermion", Nparticles=(2, 2))
    model = SingletPair()
    state = Variational(model)
    H = Hubbard(U=4.0)

    hop_up = ParticleHopUp(state, 8, thermal_steps=0)
    hop_dn = ParticleHopDn(state, 8, thermal_steps=0)
    sampler = MixSampler([hop_up, hop_dn], thermal_steps=40, sweep_steps=20)
    assert sampler.use_ref
    samples = sampler.sweep()  # exercises ref_forward in the Metropolis chain
    Eloc_ref = np.asarray(H.Oloc(state, samples))

    dense = state.todense()
    Hpsi = DenseState(H.get_quspin_op().dot(np.asarray(dense.psi)))
    basis_ints = array_to_ints(np.asarray(samples.spins))
    Eloc_exact = Hpsi[basis_ints].flatten() / dense[basis_ints].flatten()
    np.testing.assert_allclose(Eloc_ref, Eloc_exact, rtol=1e-8)
