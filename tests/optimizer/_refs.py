"""
Builders and numpy reference implementations shared by the optimizer tests.

The references re-implement the documented optimizer math independently of the
class structure in `quantax.optimizer`, so these tests pin behavior through
structural refactors. The linear solver itself is reused (same callable as the
optimizer under test) because the solver layer is not part of the refactor.
"""

import numpy as np
import jax
import jax.numpy as jnp
import quantax as qtx
from quantax.sites import Chain
from quantax.state import Variational, VS_TYPE
from quantax.model import SingleDense, RBM_Dense
from quantax.sampler import Samples
from quantax.utils import ints_to_array, to_distributed_array
from quantax.optimizer.qngd import _accepts_diag_preconditioner

from _dtype import use_dtype

# A genuinely non-holomorphic complex activation (the conj term breaks holomorphy).
NONHOLO_ACT = lambda z: jnp.cosh(z) + 0.3 * jnp.conj(z)
# A real-parameter network with complex output (a pure phase): real_to_complex.
PHASE_ACT = lambda x: jnp.exp(1j * x)


def make_state(kind, L=4, features=2):
    """
    A `Variational` of the requested vs_type on an ``L``-site chain, in 64-bit
    precision so the references below can be compared at tight tolerances.
    """
    if kind == "real":
        use_dtype(jnp.float64)
        Chain(L, boundary=1)
        model = SingleDense(features, jnp.cosh, dtype=jnp.float64)
    elif kind == "holomorphic":
        use_dtype(jnp.complex128)
        Chain(L, boundary=1)
        model = RBM_Dense(features, dtype=jnp.complex128)
    elif kind == "non_holomorphic":
        use_dtype(jnp.complex128)
        Chain(L, boundary=1)
        model = SingleDense(
            features, NONHOLO_ACT, holomorphic=False, dtype=jnp.complex128
        )
    elif kind == "real_to_complex":
        use_dtype(jnp.complex128)
        Chain(L, boundary=1)
        model = SingleDense(features, PHASE_ACT, dtype=jnp.float64)
    else:
        raise ValueError(kind)
    return Variational(model)


def make_samples(state, n, seed=0, reweight=False):
    """
    Deterministic ±1 samples (``n`` a multiple of device_count) with psi attached,
    mirroring what samplers produce. With ``reweight=True`` a non-trivial positive
    reweight factor normalized to mean 1 is attached.
    """
    assert n % jax.device_count() == 0
    rng = np.random.default_rng(seed)
    spins = rng.choice([-1, 1], size=(n, state.Nmodes)).astype(np.int8)
    spins = to_distributed_array(jnp.asarray(spins))
    psi = state(spins)
    if reweight:
        w = jnp.asarray(rng.uniform(0.5, 1.5, size=n))
        w = w / w.mean()
    else:
        w = None
    return Samples(spins, psi=psi, r=w)


def prob_only(samples, seed=1):
    """
    The samples carrying only a non-trivial ``prob``, as from a sampler dropping its
    psi, together with the samples carrying the psi and r expected from them.
    """
    rng = np.random.default_rng(seed)
    prob = rng.uniform(0.5, 1.5, size=samples.nsamples)
    w = np.abs(np.asarray(samples.psi)) ** 2 / prob
    r = to_distributed_array(jnp.asarray(w / w.mean()))
    expected = Samples(samples.spins, psi=samples.psi, r=r)
    prob = to_distributed_array(jnp.asarray(prob))
    return Samples(samples.spins, prob=prob), expected


def padded_basis(symm):
    """
    The basis states of ``symm`` distributed over the devices and their number
    ``Ns``. The first state is repeated to pad the batch to a multiple of
    device_count, so only the first ``Ns`` rows are the basis.
    """
    spins = ints_to_array(symm.basis.states)
    Ns = spins.shape[0]
    pad = np.repeat(spins[:1], (-Ns) % jax.device_count(), axis=0)
    return to_distributed_array(jnp.asarray(np.concatenate([spins, pad]))), Ns


def _weights(samples):
    r = samples.r
    if r is None:
        return np.ones(samples.nsamples)
    return np.asarray(r)


def ref_obar(state, samples):
    """StochasticQNGD.get_Obar: center by the unweighted mean, scale by sqrt(w/N)."""
    Omat = np.asarray(state.jacobian(samples.spins))
    factor = np.sqrt(_weights(samples) / samples.nsamples)[:, None]
    return (Omat - Omat.mean(axis=0, keepdims=True)) * factor


def _componentwise_clip(x, lo, hi):
    """Clip real/imag parts independently for complex x, plain clip otherwise."""
    if np.iscomplexobj(x):
        return np.clip(x.real, lo, hi) + 1j * np.clip(x.imag, lo, hi)
    return np.clip(x, lo, hi)


def ref_energy_ebar(state, H, samples, clip=None):
    """
    SR.get_Ebar and its statistics: returns (Ebar, energy, VarE). The energy and
    variance use the weighted mean of the UNCLIPPED local energies, while Ebar
    clips to Emean +- clip*sigma (real/imag parts independently, shared sigma)
    and centers at the unweighted mean. ``clip`` must match the EnergyGrad under
    test (default None = the EnergyGrad default).
    """
    dtype = np.dtype(qtx.get_default_dtype())
    Eloc = np.asarray(H.Oloc(state, samples))
    Eloc = Eloc.astype(dtype) if dtype.kind == "c" else Eloc.real.astype(dtype)
    w = _weights(samples)
    Emean = np.mean(Eloc * w)
    energy = Emean.real
    VarE = np.mean(np.abs(Eloc - Emean) ** 2 * w).real
    if clip is not None:
        sigma = np.sqrt(VarE)
        Eloc = Emean + _componentwise_clip(Eloc - Emean, -clip * sigma, clip * sigma)
    Ebar = (Eloc - Eloc.mean()) * np.sqrt(w / samples.nsamples)
    return Ebar, energy, VarE


def ref_overlap_ebar(state, target_state, samples, clip=None):
    """Supervised.get_Ebar."""
    psi = np.asarray(samples.psi if samples.psi is not None else state(samples.spins))
    w = _weights(samples)
    phi = np.asarray(target_state(samples.spins))
    ratio = phi / psi
    ratio = ratio / np.mean(ratio * w) - 1
    if clip is not None:
        ratio = _componentwise_clip(ratio, -clip, clip)
    return -ratio * np.sqrt(w / samples.nsamples)


def ref_pack(vs_type, imag_time, Obar, Ebar):
    """The vs_type packing of QNGD.solve, producing the (A, b) fed to the solver."""
    if vs_type == VS_TYPE.real_or_holomorphic:
        return Obar, (Ebar if imag_time else 1j * Ebar)
    A = np.concatenate([Obar.real, Obar.imag], axis=0)
    if imag_time:
        b = np.concatenate([Ebar.real, Ebar.imag])
    else:
        b = np.concatenate([-Ebar.imag, Ebar.real])
    return A, b


def ref_unpack(vs_type, step):
    if vs_type == VS_TYPE.non_holomorphic:
        step = step.reshape(2, -1)
        step = step[0] + 1j * step[1]
    return step.astype(np.dtype(qtx.get_default_dtype()))


def ref_solve(
    state, solver, Obar, Ebar, imag_time=True, diag_preconditioner=None, **solver_kwargs
):
    """
    QNGD.solve_equation: vs_type packing around an arbitrary solver callable,
    plus the diag_preconditioner handling. Solvers that declare the argument get
    it passed through; the rest get the emulated right preconditioning
    ``solve(A / d) / d`` on the packed system.
    """
    A, b = ref_pack(state.vs_type, imag_time, Obar, Ebar)
    A, b = jnp.asarray(A), jnp.asarray(b)
    if diag_preconditioner is None:
        step = np.asarray(solver(A, b, **solver_kwargs))
    elif _accepts_diag_preconditioner(solver):
        step = np.asarray(
            solver(A, b, diag_preconditioner=diag_preconditioner, **solver_kwargs)
        )
    else:
        d = np.asarray(diag_preconditioner)
        step = np.asarray(solver(A / d, b, **solver_kwargs)) / d
    return ref_unpack(state.vs_type, step)


def ref_spring_step(state, solver, Obar, Ebar, phi, mu):
    """One SPRING.solve; the returned step is also the new phi buffer."""
    Ebar = Ebar - mu * (Obar @ phi)
    step = ref_solve(state, solver, Obar, Ebar)
    return step + mu * phi


def ref_march_step(state, solver, Obar, Ebar, bufs, mu, beta):
    """One MARCH.solve; returns (step, new buffers)."""
    phi, v = bufs["phi"], bufs["v"]
    Ebar = Ebar - mu * (Obar @ phi)
    V = v**0.25 + 1e-8
    step = ref_solve(state, solver, Obar / V[None, :], Ebar) / V
    step = step + mu * phi
    return step, {"phi": step, "v": beta * v + np.abs(step - phi) ** 2}


def ref_adam_step(state, solver, Obar, Ebar, bufs, mu, beta):
    """
    One AdamSR.solve; returns (step, new buffers). The second solve applies the
    diagonal preconditioner V, passed natively to solvers that accept it and
    emulated as ``solve(Obar / V) / V`` otherwise (see ``ref_solve``).
    """
    g = ref_solve(state, solver, Obar, Ebar)
    t = bufs["t"] + 1
    m = mu * bufs["m"] + (1 - mu) * g
    v = beta * bufs["v"] + (1 - beta) * np.abs(g) ** 2
    mhat = m / (1 - mu**t)
    vhat = v / (1 - beta**t)
    V = vhat**0.25 + 1e-8
    Ebar = Ebar - Obar @ mhat
    step = ref_solve(state, solver, Obar, Ebar, diag_preconditioner=jnp.asarray(V))
    return step + mhat, {"m": m, "v": v, "t": t}


def ref_sgd_step(state, Obar, Ebar, imag_time=True):
    """Closed-form full pipeline with sgd_solver: x = A^† b / len(b)."""
    A, b = ref_pack(state.vs_type, imag_time, Obar, Ebar)
    return ref_unpack(state.vs_type, A.conj().T @ b / b.shape[0])
