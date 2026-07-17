import numpy as np
import jax
import jax.numpy as jnp
import pytest
from quantax import set_random_seed
from quantax.sites import Chain
from quantax.state import DenseState, Variational, GeneralDetState
from quantax.model import RBM_Dense
from quantax.symmetry import Identity
from quantax.sampler import LocalFlip, SiteFlip, MixSampler
from quantax.sampler.metropolis import (
    _get_update_size,
    _get_updated_spins,
    _get_new_psi,
)
from quantax.sampler.samples import Samples
from quantax.utils import (
    LogArray,
    ints_to_array,
    to_distributed_array,
    to_replicated_numpy,
)
from quantax.global_defs import get_sites

NDEV = jax.device_count()
# All sample counts are multiples of NDEV so that `nsamples % device_count == 0`
# holds for any device layout (CPU CI uses 4, the workstation uses GPUs).
STAT_NS = 10000 * NDEV


def _skewed_dense_state(L: int) -> tuple[DenseState, np.ndarray, np.ndarray]:
    """
    A real, positive `DenseState` on a length-``L`` spin chain biased toward
    ``s[0] == +1`` (amplitude 2 vs 1), with no Sz conservation so the full
    ``2**L`` Hilbert space is reachable by single-spin flips.
    """
    Chain(L, boundary=1)
    symm = Identity()
    symm.basis_make()
    configs = ints_to_array(symm.basis.states)  # (Ns, L), in basis order
    psi = np.where(configs[:, 0] > 0, 2.0, 1.0).astype(np.float64)
    return DenseState(jnp.asarray(psi), symm), configs, psi


# --- LocalFlip ---


def test_localflip_outputs_valid_pm1_samples():
    Chain(3, boundary=1)
    ns = 8 * NDEV
    samples = LocalFlip(DenseState(jnp.ones(8)), ns).sweep()
    spins = np.asarray(samples.spins)
    assert spins.shape == (ns, get_sites().Nmodes)
    assert set(np.unique(spins).tolist()).issubset({-1, 1})
    assert samples.state_internal is None
    assert samples.reweight_factor is None  # reweight=2 -> trivial factor


def test_localflip_reweighting_recovers_observable():
    # Markov chain sampling |psi|^1; the reweighting factor must recover
    # <s0>_{|psi|^2} = 0.6 from a proposal whose own <s0> is far lower.
    state, configs, psi = _skewed_dense_state(3)
    p = psi**2 / np.sum(psi**2)
    exact_s0 = float(np.sum(p * configs[:, 0]))

    samples = LocalFlip(state, STAT_NS, reweight=1.0).sweep()
    spins = np.asarray(samples.spins)
    r = np.asarray(samples.reweight_factor)
    est = float(np.mean(r * spins[:, 0]))
    assert np.isclose(est, exact_s0, atol=0.05)


# --- reweight normalization in the chunked reference-forward path ---


def test_chunked_ref_reweight_is_globally_normalized():
    # Regression: with `use_ref` + chunking the reweighting factor must be
    # normalized over ALL samples, not per chunk. Per-chunk normalization used to
    # bias the mean by the zero-padding ratio (here 100/96) for reweight != 2.
    Chain(8, particle_type="spinful_fermion", Nparticles=(4, 4))
    # max_parallel < per-device batch (96) forces chunking; 96 % 50 != 0 forces
    # zero-padding of the final chunk, which is what exposed the bug.
    state = GeneralDetState(max_parallel=50)
    sampler = SiteFlip(state, nsamples=96 * NDEV, reweight=1.0, thermal_steps=10)

    assert sampler.use_ref
    chunk = state.ref_chunk
    assert chunk is not None and chunk < sampler.nsamples // NDEV  # chunking active

    samples = sampler.sweep(5)
    r = np.asarray(samples.reweight_factor)
    assert r.shape == (96 * NDEV,)
    assert np.all(np.isfinite(r))
    assert np.isclose(r.mean(), 1.0, atol=1e-4)


# --- MixSampler ---


def test_mixsampler_rejects_mismatched_reweight():
    Chain(4, boundary=1)
    state = Variational(RBM_Dense(features=4))
    s1 = LocalFlip(state, 4 * NDEV, reweight=2.0, thermal_steps=0)
    s2 = LocalFlip(state, 4 * NDEV, reweight=1.0, thermal_steps=0)
    with pytest.raises(ValueError):
        MixSampler([s1, s2])


def test_mixsampler_inherits_subsampler_reweight():
    Chain(4, boundary=1)
    state = Variational(RBM_Dense(features=4))
    s1 = LocalFlip(state, 4 * NDEV, reweight=1.5, thermal_steps=0)
    s2 = LocalFlip(state, 4 * NDEV, reweight=1.5, thermal_steps=0)
    mix = MixSampler([s1, s2], thermal_steps=0)
    assert float(mix.reweight) == 1.5
    assert mix.nsamples == 8 * NDEV


def test_mixsampler_sweep_smoke():
    Chain(4, boundary=1)
    state = Variational(RBM_Dense(features=4))
    s1 = LocalFlip(state, 4 * NDEV, reweight=1.0, thermal_steps=5)
    s2 = LocalFlip(state, 4 * NDEV, reweight=1.0, thermal_steps=5)
    mix = MixSampler([s1, s2], thermal_steps=5)
    samples = mix.sweep()
    spins = np.asarray(samples.spins)
    assert spins.shape == (8 * NDEV, get_sites().Nmodes)
    assert set(np.unique(spins).tolist()).issubset({-1, 1})
    r = np.asarray(samples.reweight_factor)
    assert r.shape == (8 * NDEV,)
    assert np.isclose(r.mean(), 1.0, atol=1e-4)


# --- chunked branch (_chunk_sweep): only the updated configs are forwarded and
#     scattered back, so it must reproduce _partial_sweep (forwards all) exactly ---


def _fixed_initial_spins() -> jax.Array:
    return jnp.asarray(np.resize([1, -1], get_sites().Nmodes).astype(np.int8))


def test_chunk_sweep_matches_partial_sweep():
    # Same model + initial spins + RNG: forwarding only the changed walkers
    # (max_parallel < ns/dev -> _chunk_sweep) must give bit-identical samples to
    # forwarding all of them (max_parallel=None -> _partial_sweep).
    Chain(4, boundary=1)
    model = RBM_Dense(features=4)  # shared params, non-RefModel -> non-ref paths
    ns = 16 * NDEV  # ns/dev = 16
    s0 = _fixed_initial_spins()

    sc_state = Variational(model, max_parallel=2)
    sp_state = Variational(model)
    chunked = LocalFlip(sc_state, ns, initial_spins=s0, thermal_steps=0)
    partial = LocalFlip(sp_state, ns, initial_spins=s0, thermal_steps=0)
    assert sc_state.forward_chunk == 2 < ns // NDEV  # _chunk_sweep is active
    assert sp_state.forward_chunk is None  # _partial_sweep

    set_random_seed(0)
    sc = chunked.sweep(20)
    set_random_seed(0)
    sp = partial.sweep(20)
    assert np.array_equal(np.asarray(sc.spins), np.asarray(sp.spins))


def test_chunk_sweep_invariant_to_chunk_size():
    # The size-rounding / gather / scatter must be correct for any chunk size,
    # including ones that do not divide ns/dev (non-trivial padding).
    Chain(4, boundary=1)
    model = RBM_Dense(features=4)
    ns = 16 * NDEV
    s0 = _fixed_initial_spins()

    ref = None
    for chunk in (2, 3, 5):
        samp = LocalFlip(
            Variational(model, max_parallel=chunk),
            ns,
            initial_spins=s0,
            thermal_steps=0,
        )
        assert chunk < ns // NDEV
        set_random_seed(0)
        spins = np.asarray(samp.sweep(20).spins)
        if ref is None:
            ref = spins
        else:
            assert np.array_equal(spins, ref)


def test_mixsampler_chunk_sweep_matches_partial():
    # MixSampler has its own _chunk_sweep; it too must match _partial_sweep.
    Chain(4, boundary=1)
    model = RBM_Dense(features=4)
    s0 = _fixed_initial_spins()

    def make_mix(max_parallel):
        state = Variational(
            model, max_parallel=max_parallel
        )  # components share one state
        s1 = LocalFlip(state, 8 * NDEV, thermal_steps=0)
        s2 = LocalFlip(state, 8 * NDEV, thermal_steps=0)
        mix = MixSampler(
            [s1, s2], initial_spins=s0, thermal_steps=0
        )  # total ns/dev = 16
        return mix, state

    mix_c, state_c = make_mix(2)  # _chunk_sweep
    mix_p, state_p = make_mix(None)  # _partial_sweep
    assert state_c.forward_chunk == 2 and state_p.forward_chunk is None

    set_random_seed(0)
    sc = mix_c.sweep(20)
    set_random_seed(0)
    sp = mix_p.sweep(20)
    assert np.array_equal(np.asarray(sc.spins), np.asarray(sp.spins))


# --- accept rule (_update): escape from nan / zero / overflowed psi ---


def _accepted_mask(old_psi, new_psi) -> np.ndarray:
    """
    Run ``Metropolis._update`` with crafted wavefunction values (proposed spins
    always differ from the old ones) and return the per-walker accept mask.
    """
    ns = old_psi.shape[0]
    sampler = LocalFlip(DenseState(jnp.ones(2 ** get_sites().Nsites)), ns)
    old_spins = jnp.tile(_fixed_initial_spins(), (ns, 1))
    new_spins = np.asarray(-old_spins)
    out = sampler._update(
        jax.random.key(0),
        None,
        Samples(old_spins, old_psi),
        Samples(-old_spins, new_psi),
    )
    return (np.asarray(out.spins) == new_spins).all(axis=1)


def test_update_escapes_bad_psi():
    # nan/inf psi are never adopted and always escaped to a finite proposal;
    # otherwise an inf psi would be an absorbing state.
    Chain(3, boundary=1)
    nan, inf = np.nan, np.inf
    old = [nan, nan, 1.0, 1.0, inf, inf, 0.0, 0.0]
    new = [1.0, nan, nan, 2.0, inf, 1.0, 0.0, 1.0]
    expect = [True, False, False, True, False, True, False, True]

    old_psi = jnp.tile(jnp.asarray(old, jnp.float32), NDEV)
    new_psi = jnp.tile(jnp.asarray(new, jnp.float32), NDEV)
    accepted = _accepted_mask(old_psi, new_psi)
    assert np.array_equal(accepted, np.tile(expect, NDEV))


def test_update_escapes_bad_psi_logarray():
    # Same escapes with LogArray psi: finiteness must be read from components
    # (isfinite), not the densified value which overflows at large finite logabs.
    Chain(3, boundary=1)
    nan, inf = np.nan, np.inf
    old_sign, old_logabs = [nan, 1.0, 1.0, 1.0], [0.0, inf, 0.0, inf]
    new_sign, new_logabs = [1.0, 1.0, nan, 1.0], [0.0, inf, 0.0, 0.0]
    expect = [True, False, False, True]

    def make(sign, logabs):
        return LogArray(
            jnp.tile(jnp.asarray(sign, jnp.float32), NDEV),
            jnp.tile(jnp.asarray(logabs, jnp.float32), NDEV),
        )

    accepted = _accepted_mask(make(old_sign, old_logabs), make(new_sign, new_logabs))
    assert np.array_equal(accepted, np.tile(expect, NDEV))


def test_chunk_helpers_gather_scatter():
    # Direct check of the gather/scatter helpers with unequal per-device update
    # counts and padding -- the multi-device core of _chunk_sweep.
    per_dev = 4
    n = NDEV * per_dev
    counts = [(d % per_dev) + 1 for d in range(NDEV)]  # 1..4 cycling, <= per_dev
    flat = np.zeros(n, dtype=bool)
    for d in range(NDEV):
        flat[d * per_dev : d * per_dev + counts[d]] = True
    is_updated = to_distributed_array(flat)

    chunk = 2
    size = int(_get_update_size(is_updated, chunk))
    assert (
        size == ((max(counts) - 1) // chunk + 1) * chunk
    )  # global max rounded up to a chunk

    spins = to_distributed_array(np.arange(n, dtype=np.int8).reshape(n, 1))
    s_up, idx = _get_updated_spins(spins, is_updated, size)

    # scatter: write (index + 100) at updated positions, keep old (= index) elsewhere
    old_psi = to_distributed_array(np.arange(n, dtype=np.float32))
    new_psi = to_distributed_array(
        to_replicated_numpy(s_up).ravel().astype(np.float32) + 100.0
    )
    merged = to_replicated_numpy(_get_new_psi(old_psi, new_psi, is_updated, idx))

    expected = np.arange(n, dtype=np.float32)
    for d in range(NDEV):
        expected[d * per_dev : d * per_dev + counts[d]] += 100.0
    assert np.array_equal(merged, expected)
