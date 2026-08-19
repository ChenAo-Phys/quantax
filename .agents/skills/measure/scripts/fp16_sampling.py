"""Measure with fp16 sampling: reweight to the fp32 target, bin the error.

An fp16 copy of the trained model draws samples from q ∝ |psi16|^2 instead of
the target p ∝ |psi|^2. This is ordinary importance sampling: with weights
w = p/q = |psi/psi16|^2, the self-normalized estimate
<O> = <w Oloc>_q / <w>_q is unbiased. fp16 only shapes the sampled
distribution; weights and estimators are evaluated in high precision. The
price is the effective-sample-size fraction <w>^2 / <w^2>, in practice
indistinguishable from 1.

Uncertainty and autocorrelation with reweighting — two caveats beyond the
plain binning analysis of measure_binning.py:

- The estimate is self-normalized, so the weights must be normalized over the
  whole run, not sweep by sweep. The raw w = |psi/psi16|^2 ≈ 1 for fp16, so
  it can be accumulated directly without rescaling.
- Bin the *centered* weighted estimator z = w (Oloc - <O>). Binning w * Oloc
  itself adds <O>^2 Var(w) to the variance — for the energy, whose |<O>| is
  extensive while Var(Oloc) is suppressed by the zero-variance property, this
  can dominate and grossly overestimate the error. With exact centering
  mean(z) = 0, so the fluctuations of the denominator <w> also drop out of
  the error at first order: err(<O>) = err_bin(z) / <w>.

The binning streams through the BinningAccumulator of measure_binning.py
(copied verbatim; importing it would execute that script), fed per sweep with
the two columns [z, w]. One streaming-specific point: the exact global <O> is
unknown until the run ends, so z is centered with the first-sweep estimate
O0. The estimate <O> = O0 + <z>/<w> is exact for any O0, and the
mis-centering only enters the variance at second order, (O0 - <O>)^2 Var(w).

Adapt the lattice, Hamiltonian, model, and the quoted `param_file` to the
trained state; give the fp16 state the same `symm` and `max_parallel`. fp16
sampling only pays off when sampling dominates the total cost — large
networks, many chains, and cheap (e.g. diagonal) observables; benchmark on
the target GPU first (see the time-bench skill). When many off-diagonal
operators are measured (e.g. pair-pair correlations), the Oloc forward
passes cost far more than the sampling — sample in fp32 there for better
reliability.
"""

import numpy as np
import jax
import jax.numpy as jnp

# On multi-node jobs, initialize distributed JAX here, before importing
# quantax (see the hpc-usage skill for the slurm-adapted arguments):
# jax.distributed.initialize(...)

import quantax as qtx
from quantax.sampler import Samples

is_main = jax.process_index() == 0


class BinningAccumulator:
    """Streaming binning analysis of local estimators.

    Feed one batch of shape (nchains,) or (nchains, nobs) per sweep with
    `add`. Bins along the sweep axis double in size level by level; each
    level keeps only running sums (on device, sample-sharded), never the
    full series. Estimators are centered by the first-sweep mean so that
    float32 accumulation doesn't lose the variance to cancellation.
    """

    def __init__(self):
        self._levels = []
        self._offset = None

    def add(self, oloc):
        x = jnp.asarray(oloc)
        if x.ndim == 1:
            x = x[:, None]
        if self._offset is None:
            self._offset = x.mean(axis=0)
        x = x - self._offset
        lvl = 0
        while x is not None:
            if lvl == len(self._levels):
                self._levels.append({"buf": None, "s1": 0.0, "s2": 0.0, "n": 0})
            L = self._levels[lvl]
            L["s1"] = L["s1"] + x
            L["s2"] = L["s2"] + x**2
            L["n"] += 1
            if L["buf"] is None:  # wait for a partner to form the next-level bin
                L["buf"] = x
                x = None
            else:
                x = (L["buf"] + x) / 2
                L["buf"] = None
            lvl += 1

    def results(self, min_bins=8):
        """Return (mean, err, tau, bin_sizes, errors) as host arrays, each per
        observable.

        `errors[i]` is the error estimate at bin size `bin_sizes[i]`; it should
        plateau at large bins, and `err = errors[-1]` is the tau-corrected
        uncertainty of `mean`. `tau` is the integrated autocorrelation time.
        """
        bin_sizes, errors = [], []
        for lvl, L in enumerate(self._levels):
            if L["n"] < min_bins:
                break
            m = L["n"] * L["s1"].shape[0]  # bins * chains
            mean_l = L["s1"].sum(axis=0) / m
            var_l = L["s2"].sum(axis=0) / m - mean_l**2
            # reduced over samples -> replicated scalars, safe to bring to host
            errors.append(np.asarray(jnp.sqrt(var_l / (m - 1))))
            bin_sizes.append(2**lvl)
        errors = np.stack(errors)

        L0 = self._levels[0]
        mean = self._offset + L0["s1"].sum(axis=0) / (L0["n"] * L0["s1"].shape[0])
        tau = 0.5 * ((errors[-1] / errors[0]) ** 2 - 1)
        if is_main and np.any(errors[-1] > 1.2 * errors[-2]):
            print("WARNING: binning error not plateaued; increase NSWEEPS")
        return np.asarray(mean), errors[-1], tau, np.array(bin_sizes), errors


# --- system and state: adapt to the target simulation ---
lattice = qtx.sites.Square(4)
N = lattice.Nsites
H = qtx.operator.Ising(h=3.0)

model = qtx.model.RBM_Dense(features=16)
state = qtx.state.Variational(
    model,
    # param_file="params.eqx",  # well-trained weights for measurement
    max_parallel=16384,
)

# --- fp16 copy, used only inside the sampler ---
# cast only float parameters; integer buffers must keep their dtype
cast16 = lambda x: x.astype(jnp.float16) if jnp.issubdtype(x.dtype, jnp.floating) else x
state16 = qtx.state.Variational(
    qtx.utils.filter_tree_map(cast16, state.model), max_parallel=16384
)
sampler16 = qtx.sampler.LocalFlip(
    state16, nsamples=512, thermal_steps=50 * N, sweep_steps=2 * N
)

NSWEEPS = 128

acc = BinningAccumulator()  # columns [z, w], z = w * (Eloc - O0)
O0 = None
w2_sum = 0.0
for _ in range(NSWEEPS):
    s = sampler16.sweep().spins
    psi = state(s)  # fp32 psi, attached below so Oloc doesn't recompute it
    psi16 = sampler16._psi  # the sampler stores psi16 of its current spins
    assert psi16 is not None
    w = jnp.asarray(abs(psi / psi16) ** 2)
    Eloc = H.Oloc(state, Samples(s, psi)).real
    if O0 is None:
        O0 = float(jnp.mean(w * Eloc)) / float(jnp.mean(w))
    acc.add(jnp.stack([w * (Eloc - O0), w], axis=1))
    w2_sum = w2_sum + jnp.mean(w**2)

mean, err, tau, bin_sizes, errors = acc.results()
w_mean = mean[1]
E = O0 + mean[0] / w_mean  # exact self-normalized estimate for any O0
E_err = err[0] / w_mean
ess = w_mean**2 / float(w2_sum / NSWEEPS)

if is_main:
    print(
        f"E = {E:.6f} ± {E_err:.2e}   "
        f"(tau_int = {tau[0]:.2f}, ESS fraction = {ess:.4f})"
    )
    for b, e in zip(bin_sizes, errors):
        print(f"  bin size {b:4d}: error = {e[0] / w_mean:.3e}")
