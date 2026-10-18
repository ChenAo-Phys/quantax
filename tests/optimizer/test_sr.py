"""
Characterization tests for quantax/optimizer/sr.py and the EnergyGrad source of
quantax/optimizer/gradient.py: the SR energy gradient and statistics, local-energy
clipping, the SPRING / MARCH / AdamSR momentum variants, and exact reconfiguration
(ER).

On CPU the runtime is dominated by compilation. The jitted functions of a state
are compiled per instance and eager ops per shape, so the tests sharing a state
are merged, and the shapes of `make_state` / `make_samples` (4-site chain, NS
samples) are reused wherever possible.

All tests run in 64-bit precision to compare with the numpy references at tight
tolerances. SR at the default float32 precision, fed by a quantax sampler, is
exercised end to end by tests/optimizer/test_vmc_loop.py.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from quantax.sites import Chain
from quantax.state import Variational
from quantax.model import RBM_Dense, Jastrow
from quantax.operator import Ising
from quantax.sampler import Samples
from quantax.symmetry import Identity, TransND
from quantax.optimizer import SR, SPRING, MARCH, AdamSR, ER, auto_shift_eig
from quantax.optimizer import EnergyGrad
from quantax.utils import to_distributed_array, to_replicated_array

from _dtype import use_dtype
from _refs import (
    make_state,
    make_samples,
    padded_basis,
    prob_only,
    ref_obar,
    ref_energy_ebar,
    ref_solve,
    ref_spring_step,
    ref_march_step,
    ref_adam_step,
)

NDEV = jax.device_count()
NS = -(-32 // NDEV) * NDEV  # samples per step, 32 rounded up to a multiple of NDEV


def _assert_close(actual, desired, rtol=1e-10, atol=1e-14):
    np.testing.assert_allclose(np.asarray(actual), desired, rtol=rtol, atol=atol)


# ====================================================================
# SR statistics and estimators
# ====================================================================


def test_sr_estimators_match_reference(x64):
    # energy and VarE center at the weighted mean; Ebar and Obar center at the
    # unweighted mean. Both behaviors are pinned for plain and reweighted samples.
    # The unweighted centering makes F = Obar^† Ebar biased by
    # (<O>_p - <O>_q)^* (<E>_p - <E>_q) for samples drawn from q != p = |psi|^2.
    state = make_state("holomorphic")
    H = Ising(h=1.0)
    opt = SR(state, H, solver=auto_shift_eig())
    assert opt.hamiltonian is H
    assert opt.energy is None and opt.VarE is None
    for reweight in (False, True):
        samples = make_samples(state, NS, reweight=reweight)
        opt.get_step(samples)
        Ebar_ref, energy_ref, VarE_ref = ref_energy_ebar(state, H, samples)
        _assert_close(opt.energy, energy_ref)
        _assert_close(opt.VarE, VarE_ref)
        _assert_close(opt.get_Ebar(samples), Ebar_ref)
        _assert_close(opt.get_Obar(samples), ref_obar(state, samples))

    # raw spins are wrapped into Samples without psi, i.e. with r = 1
    plain = Samples(samples.spins)
    _assert_close(opt.get_Ebar(samples.spins), ref_energy_ebar(state, H, plain)[0])
    _assert_close(opt.get_Obar(samples.spins), ref_obar(state, plain))

    # Samples carrying only prob get psi and r filled in by the local energy before
    # r is used, so Ebar matches the samples with the expected psi and r ...
    samples, expected = prob_only(make_samples(state, NS))
    Ebar = opt.get_Ebar(samples)
    _assert_close(samples.psi, np.asarray(expected.psi))
    _assert_close(samples.r, np.asarray(expected.r))
    _assert_close(Ebar, ref_energy_ebar(state, H, expected)[0])
    # ... and get_Obar fills them in by itself if it's called first
    samples, _ = prob_only(make_samples(state, NS))
    _assert_close(opt.get_Obar(samples), ref_obar(state, expected))


# ====================================================================
# Local-energy clipping and non-finite local energies
# ====================================================================


class _StubHamiltonian:
    """A fake Hamiltonian returning designed local energies."""

    def __init__(self, eloc):
        # distributed like the output of Operator.Oloc
        self._eloc = to_distributed_array(jnp.asarray(eloc))

    def Oloc(self, state, samples):
        return self._eloc


def _stub_samples(n, r=None):
    # EnergyGrad only reads r and nsamples of the samples when the Hamiltonian is
    # stubbed, so no state or psi is needed
    spins = np.random.default_rng(0).choice([-1, 1], size=(n, 4)).astype(np.int8)
    if r is not None:
        r = to_distributed_array(jnp.asarray(r))
    return Samples(to_distributed_array(jnp.asarray(spins)), r=r)


def test_eloc_clip_bounds_outliers_and_keeps_stats(x64):
    # Designed outliers on both sides are clipped in Ebar, while the reported
    # energy/VarE stay the unclipped, unbiased estimators and Ebar is recentered.
    use_dtype(jnp.float64)
    n, clip = NS, 2.0
    eloc = np.linspace(-1.0, 1.0, n)
    eloc[0] += 50.0  # far above Emean + clip * sigma
    eloc[1] -= 50.0  # far below Emean - clip * sigma
    H = _StubHamiltonian(eloc)
    samples = _stub_samples(n)

    grad = EnergyGrad(H, clip=clip)
    Ebar = np.asarray(grad.ebar(None, samples.spins))  # raw spins are wrapped
    Ebar_ref, _, _ = ref_energy_ebar(None, H, samples, clip=clip)
    _assert_close(Ebar, Ebar_ref, rtol=1e-12)

    # reported statistics are the unclipped ones
    _assert_close(grad.energy, eloc.mean(), rtol=1e-12)
    _assert_close(grad.VarE, np.mean(np.abs(eloc - eloc.mean()) ** 2), rtol=1e-12)
    # both bounds engaged: the clipped local energies span exactly 2 * clip * sigma
    sigma = np.sqrt(float(np.asarray(grad.VarE)))
    eloc_rec = Ebar * np.sqrt(n)  # unit weights: Ebar = (Eloc_clip - mean)/sqrt(n)
    np.testing.assert_allclose(np.ptp(eloc_rec), 2 * clip * sigma, rtol=1e-12)
    assert np.ptp(eloc_rec) < np.ptp(eloc)
    # recentering: Ebar sums to zero after the clip
    np.testing.assert_allclose(np.sum(Ebar), 0.0, atol=1e-12)


def test_eloc_clip_complex_componentwise(x64):
    # Complex local energies: real/imag parts clip independently with the
    # shared sigma = sqrt(VarE); the dtype stays complex. One outlier per component
    # and side engages that bound of that component only.
    use_dtype(jnp.complex128)
    n, clip = NS, 1.0
    rng = np.random.default_rng(3)
    eloc = rng.normal(size=n) + 1j * rng.normal(size=n)
    eloc[0] += 30.0  # real-part outliers, above and below
    eloc[2] -= 30.0
    eloc[1] += 30.0j  # imaginary-part outliers, above and below
    eloc[3] -= 30.0j
    H = _StubHamiltonian(eloc)
    samples = _stub_samples(n)

    grad = EnergyGrad(H, clip=clip)
    Ebar = np.asarray(grad.ebar(None, samples))
    assert np.iscomplexobj(Ebar)
    Ebar_ref, _, VarE_ref = ref_energy_ebar(None, H, samples, clip=clip)
    _assert_close(Ebar, Ebar_ref, rtol=1e-12)

    # each outlier exceeds the threshold in its own component only, while the other
    # samples stay within it, so the componentwise reference differs from clipping
    # the modulus or a single component
    t = clip * np.sqrt(VarE_ref)
    d = eloc - np.mean(eloc)
    big = np.abs(np.stack([d.real, d.imag])) > t
    np.testing.assert_array_equal(np.argwhere(big), [[0, 0], [0, 2], [1, 1], [1, 3]])


def test_ebar_excludes_nonfinite_samples(x64):
    # A NaN / inf local energy (e.g. from a NaN psi sample) or reweighting factor
    # (e.g. from prob = 0) must not leak into Emean or sigma, which would defeat
    # the clip and turn the whole Ebar into NaN: the bad samples are excluded from
    # the weighted statistics and contribute zero to Ebar.
    use_dtype(jnp.float64)
    n, clip = NS, 2.0
    eloc = np.linspace(-1.0, 1.0, n)
    eloc[0] += 10.0  # an outlier engaging the clip
    eloc[3] = np.nan
    eloc[7] = np.inf
    r = np.random.default_rng(1).uniform(0.5, 1.5, n)
    r[5] = np.inf
    good = np.isfinite(eloc) & np.isfinite(r)
    H = _StubHamiltonian(eloc)

    grad = EnergyGrad(H, clip=clip)
    with pytest.warns(UserWarning, match="3 sample.*non-finite local energy"):
        Ebar = np.asarray(grad.ebar(None, _stub_samples(n, r)))

    assert np.all(np.isfinite(Ebar))
    np.testing.assert_array_equal(Ebar[~good], 0.0)

    # reported statistics are the weighted ones of the finite samples only
    w = np.where(good, r, 0.0)
    Emean = np.sum(np.where(good, eloc, 0.0) * w) / np.sum(w)
    VarE = np.sum(np.where(good, eloc - Emean, 0.0) ** 2 * w) / np.sum(w)
    _assert_close(grad.energy, Emean, rtol=1e-12)
    _assert_close(grad.VarE, VarE, rtol=1e-12)

    # good entries follow the usual clipped, recentered estimator, with the
    # bad samples pinned at Emean (zero deviation) and given zero weight
    sigma = np.sqrt(VarE)
    e_pin = np.where(good, eloc, Emean)
    e_clip = Emean + np.clip(e_pin - Emean, -clip * sigma, clip * sigma)
    assert e_clip[0] < eloc[0]
    Ebar_ref = (e_clip - e_clip.mean()) * np.sqrt(w / n)
    _assert_close(Ebar, Ebar_ref, rtol=1e-12)


def test_wrappers_pass_arguments_through(x64):
    # clipping is opt-in: the default None is a regression guard, as clip=5 biased
    # the converged energies of the J1J2 tutorial. imag_time and solver must reach
    # the QNGD base too: dropping them would silently fall back to imaginary time or
    # the default solver, which the momentum tests can't detect as they pass the
    # default values.
    state = make_state("holomorphic")
    H = Ising(h=1.0)
    solver = auto_shift_eig()
    eloc = np.linspace(-1.0, 1.0, NS)
    eloc[0] += 50.0  # an outlier engaging the clip
    stub = _StubHamiltonian(eloc)
    samples = _stub_samples(NS)
    clipped, _, _ = ref_energy_ebar(None, stub, samples, clip=1.5)
    unclipped, _, _ = ref_energy_ebar(None, stub, samples)
    assert not np.allclose(clipped, unclipped)
    for cls in (SR, SPRING, MARCH, AdamSR):
        _assert_close(cls(state, stub, clip=1.5).get_Ebar(samples), clipped)
        opt = cls(state, stub)
        _assert_close(opt.get_Ebar(samples), unclipped)
        assert opt.imag_time is True
        opt = cls(state, H, imag_time=False, solver=solver)
        assert opt.imag_time is False and opt._solver is solver


# ====================================================================
# Momentum variants
# ====================================================================


def _two_step_data(state, H, seeds=(0, 1)):
    out = []
    for seed in seeds:
        samples = make_samples(state, NS, seed=seed)
        Ebar, _, _ = ref_energy_ebar(state, H, samples)
        out.append((samples, ref_obar(state, samples), Ebar))
    return out


def _assert_two_steps(opt, data, ref_step, bufs, rtol):
    """
    Two steps of ``opt`` against ``ref_step(Obar, Ebar, bufs) -> (step, bufs)``,
    starting from the reference buffers ``bufs``. Before each step and after the
    last one the buffers of ``opt`` are checked against the reference ones. The
    reference step is conditioned on the realized buffers to avoid compounding the
    step-1 round-off through another 1/V rescaling. The buffers after step 2 are
    the first to depend on non-trivial previous buffers (phi = 0, v = 1 or 0 at
    step 1). Returns the last step and the reference buffers after it.
    """
    new_bufs = bufs
    for i, (samples, Obar, Ebar) in enumerate(data):
        for k in bufs:
            _assert_close(opt._buffers[k], new_bufs[k], rtol=rtol[0], atol=1e-12)
        bufs = {k: np.asarray(opt._buffers[k]) for k in bufs}
        step = np.asarray(opt.get_step(samples))
        ref, new_bufs = ref_step(Obar, Ebar, bufs)
        _assert_close(step, ref, rtol=rtol[i], atol=1e-12)
    for k in bufs:
        _assert_close(opt._buffers[k], new_bufs[k], rtol=rtol[-1], atol=1e-12)
    return step, new_bufs


def test_momentum_optimizers_two_steps(x64):
    # Two steps of SPRING, MARCH and AdamSR against the numpy references, with
    # non-default hyperparameters so that their pass-through is pinned too. With
    # more parameters than samples (the MinSR regime), the diagonal preconditioner
    # of MARCH and AdamSR changes the min-norm solution, so it is pinned as well.
    # The default eig-family solver has no native diag_preconditioner and gets the
    # emulated right preconditioning, which the references reproduce.
    state = make_state("holomorphic", features=8)  # nparams = 40 > NS
    H = Ising(h=1.0)
    solver = auto_shift_eig()
    mu, beta = 0.8, 0.9
    data = _two_step_data(state, H)
    zeros = np.zeros(state.nparams, dtype=np.complex128)

    def spring(Obar, Ebar, bufs):
        step = ref_spring_step(state, solver, Obar, Ebar, bufs["phi"], mu)
        return step, {"phi": step}

    def march(Obar, Ebar, bufs):
        return ref_march_step(state, solver, Obar, Ebar, bufs, mu, beta)

    def adam(Obar, Ebar, bufs):
        return ref_adam_step(state, solver, Obar, Ebar, bufs, mu, beta)

    # the first SPRING step (phi = 0) is plain SR
    opt = SPRING(state, H, solver=solver, mu=mu)
    _assert_two_steps(opt, data, spring, {"phi": zeros}, rtol=(1e-9, 1e-8))

    # MARCH's "v" buffer starts at 1, so its first-step preconditioner is neutral
    opt = MARCH(state, H, solver=solver, mu=mu, beta=beta)
    bufs = {"phi": zeros, "v": np.ones(state.nparams)}
    _assert_two_steps(opt, data, march, bufs, rtol=(1e-9, 1e-6))

    opt = AdamSR(state, H, solver=solver, mu=mu, beta=beta)
    bufs = {"m": zeros, "v": np.zeros(state.nparams), "t": 0}
    step2, bufs2 = _assert_two_steps(opt, data, adam, bufs, rtol=(1e-9, 1e-9))
    # without the preconditioner the second solve gives a different solution, far
    # beyond the tolerance, so the references do pin its value
    _, O2, e2 = data[1]
    mhat2 = bufs2["m"] / (1 - mu ** bufs2["t"])
    no_precond = ref_solve(state, solver, O2, e2 - O2 @ mhat2) + mhat2
    assert np.max(np.abs(step2 - no_precond)) > 1e-3 * np.linalg.norm(step2)


# ====================================================================
# ER (exact reconfiguration)
# ====================================================================


def _er_dense_reference(state, H, solver):
    """
    The ER step and energy from dense numpy algebra in the Hilbert space of the
    state's symmetry, whose basis states are the symmetry representatives with
    amplitudes psi(s) / amp(s).
    """
    basis = state.symm.basis
    spins, Ns = padded_basis(state.symm)
    amp = np.asarray(basis.get_amp(basis.states))
    assert np.ptp(np.abs(amp)) > 0  # orbits of different sizes
    psi = np.asarray(state(spins))[:Ns] / amp
    psi = psi / np.linalg.norm(psi)

    Omat = np.asarray(state.jacobian(spins))[:Ns] * psi[:, None]
    Obar = Omat - psi[:, None] * (psi.conj() @ Omat)[None, :]
    Hd = np.asarray(H.todense(state.symm))
    energy = (psi.conj() @ Hd @ psi).real
    Ebar = Hd @ psi - energy * psi
    return energy, ref_solve(state, solver, Obar, Ebar)


def test_er_step_convergence_passthrough_and_nonfinite_jacobian(x64, monkeypatch):
    # A complex, translation-symmetric state: ER works in the basis of the symmetry
    # representatives, so psi is divided by their amplitudes, which differ between
    # orbits of different sizes (1, 2 or 4 on a 4-site ring). The complex psi pins
    # the conjugations of the dense Obar. The first step and energy match the dense
    # reference, and following the steps converges to the ground state. ER is
    # deterministic; with lr = 0.05 the relative error after 30 steps is below 4e-5
    # for 20 tried seeds, starting from 18-48%.
    use_dtype(jnp.complex128)
    Chain(4, boundary=1)
    state = Variational(RBM_Dense(2, dtype=jnp.complex128), symm=TransND())
    H = Ising(h=1.0)
    solver = auto_shift_eig()
    er = ER(state, H, solver=solver)
    step = er.get_step()
    assert step.dtype == jnp.complex128
    energy, expected = _er_dense_reference(state, H, solver)
    _assert_close(er.energy, energy)
    _assert_close(step, expected, rtol=1e-8, atol=1e-12)

    for _ in range(30):
        state.update(0.05 * er.get_step())
    er.get_step()  # refresh the energy after the last update
    # the ground state of the sector, which is the global one for this Hamiltonian
    E0 = np.linalg.eigvalsh(np.asarray(H.todense(state.symm)))[0]
    assert float(np.asarray(er.energy)) < E0 + 1e-3 * abs(E0)

    # the arguments reach the ExactQNGD base, e.g. the full Hilbert space for symm
    er2 = ER(state, H, imag_time=False, solver=solver, symm=Identity())
    assert er2.imag_time is False and er2._solver is solver and er2._Ns == 16

    # Non-finite Jacobian entries (e.g. from psi = 0) are zeroed before the
    # projection, so they can't spread through Omean into the whole Obar
    rng = np.random.default_rng(0)
    nspins = er._spins.shape[0]
    psi = rng.normal(size=nspins) + 1j * rng.normal(size=nspins)
    psi[er._Ns :] = 0
    J = np.array(state.jacobian(er._spins))
    J[0, 1] = np.nan
    J[2, 0] = np.inf
    monkeypatch.setattr(
        state, "jacobian", lambda s: to_distributed_array(jnp.asarray(J))
    )
    Obar = er.get_Obar(to_replicated_array(jnp.asarray(psi)))
    Omat = J * psi[:, None]
    Omat = np.where(np.isfinite(Omat), Omat, 0)
    expected = Omat - psi[:, None] * (psi.conj() @ Omat)[None, :]
    _assert_close(Obar, expected, rtol=1e-12)


# ====================================================================
# SR integration with Monte Carlo samples
# ====================================================================


def test_sr_converges_with_sampled_configurations(x64):
    # SR with fresh samples drawn from |psi|^2 every step converges to the ground
    # state. The 16 configurations are sampled exactly in numpy rather than by a
    # quantax sampler, whose first compilation would dominate the test. A complex
    # Jastrow factor has no nodes, unlike an RBM of cosh, where SR gets stuck or
    # diverges for a few percent of the initializations once psi nearly vanishes on
    # some configurations; its best energy is 3.4e-4 above the ground energy. The
    # exact energy of the final state is checked, which is free of sampling noise:
    # over 100 numpy and 60 quantax seeds its relative error after 20 steps is at
    # most 2.1e-3, starting from 17-71%, while a wrong update sign ends up at +180%
    # and the exact ground state of h = 0.5 is 7% off.
    use_dtype(jnp.complex128)
    Chain(4, boundary=1)
    state = Variational(Jastrow(dtype=jnp.complex128))
    H = Ising(h=1.0)
    opt = SR(state, H)
    rng = np.random.default_rng(0)
    state.symm.basis_make()
    configs, Ns = padded_basis(state.symm)
    for _ in range(20):
        psi = np.asarray(state(configs))[:Ns]
        p = np.abs(psi) ** 2 / np.sum(np.abs(psi) ** 2)
        idx = rng.choice(len(p), size=NS, p=p)
        spins = to_distributed_array(configs[idx])
        prob = to_distributed_array(jnp.asarray(p[idx]))
        step = opt.get_step(Samples(spins, psi=state(spins), prob=prob))
        state.update(step * 0.05)

    Hd = np.asarray(H.todense())
    psi = np.asarray(state(configs))[:Ns]
    energy = (psi.conj() @ Hd @ psi).real / np.vdot(psi, psi).real
    E0 = np.linalg.eigvalsh(Hd)[0]
    assert energy < E0 + 1e-2 * abs(E0)
