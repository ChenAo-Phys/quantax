"""Production measurement of observables with binning error analysis.

Measures a diagonal observable (M^2) directly from the spins and off-diagonal
ones (H and local <sx_i> at every site) via `Oloc`. Local estimators are fed
per sweep into streaming binning accumulators, which give the mean, the
integrated autocorrelation time tau, and the uncertainty
sqrt(var * (1 + 2 tau) / nsamples_total) for every quantity separately.

Multi-node ready: local estimators and all accumulators stay on the GPUs with
their sample sharding — no host transfer or gather until the final small
reductions. Don't wrap distributed arrays in `np.asarray` (it fails for
non-addressable shards and would move data across nodes anyway).

Memory: storing the full per-site Oloc series for binning would take
nsweeps * nchains * N * 4 bytes per run — a GPU-memory OOM risk on long runs.
`BinningAccumulator` instead keeps O(log nsweeps) partial sums, i.e.
O(nchains_local * nobs) per device independent of the run length, while still
producing the exact binning curve and a separate tau for each observable.

Adapt the lattice, Hamiltonian, model, and the quoted `param_file` to the
trained state. Results are printed and saved to measurements.npz.
"""

import numpy as np
import jax
import jax.numpy as jnp

# On multi-node jobs, initialize distributed JAX here, before importing
# quantax (see the hpc-usage skill for the slurm-adapted arguments):
# jax.distributed.initialize(...)

import quantax as qtx
from quantax.operator import sigma_x, sigma_z, Operator

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

    def add(self, Oloc):
        x = jnp.asarray(Oloc)  # (nchains,) or (nchains, nobs)
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

sampler = qtx.sampler.LocalFlip(
    state, nsamples=512, thermal_steps=50 * N, sweep_steps=2 * N
)

# save thermalized chains
spins_full = qtx.utils.to_replicated_numpy(sampler._spins)
if is_main:
    np.save("spins.npy", spins_full)
del spins_full

# If the run collapses later, rebuild the sampler from the saved spins instead
# of paying thermalization again:
# sampler = qtx.sampler.LocalFlip(
#     state,
#     nsamples=512,
#     sweep_steps=2 * N,
#     thermal_steps=0,
#     initial_spins=jnp.load("spins.npy"),
# )

NSWEEPS = 128

M = sum((sigma_z(i) for i in range(N)), Operator([])) / N
M2 = M @ M
sx_ops = [sigma_x(i) for i in range(N)]

acc_E = BinningAccumulator()
acc_M2 = BinningAccumulator()
acc_sx = BinningAccumulator()

for _ in range(NSWEEPS):
    samples = sampler.sweep()
    s = samples.spins

    # diagonal observable: needs only the spins, no psi(s)
    acc_M2.add(M2.apply_diag(s).real)

    # off-diagonal: attach psi(s) once so the Oloc calls below don't recompute it
    samples = qtx.sampler.Samples(s, state(s))
    acc_E.add(H.Oloc(state, samples).real)
    sxloc = jnp.stack(
        [op.Oloc(state, samples).real for op in sx_ops], axis=1
    )  # (nchains, N), sample-sharded
    acc_sx.add(sxloc)

# --- results: uncertainty = sqrt(var * (1 + 2 tau) / ntotal) per observable ---
results = {}
for name, acc in [("E", acc_E), ("M2", acc_M2)]:
    mean, err, tau, bin_sizes, errors = acc.results()
    results[name] = (mean[0], err[0])
    if is_main:
        print(f"\n{name} = {mean[0]:.6f} ± {err[0]:.2e}   (tau_int = {tau[0]:.2f})")
        for b, e in zip(bin_sizes, errors):
            print(f"  bin size {b:4d}: error = {e[0]:.3e}")

sx_mean, sx_err, sx_tau, _, _ = acc_sx.results()
if is_main:
    print("\nsx per site:")
    for i in range(N):
        print(
            f"  site {i:3d}: {sx_mean[i]:.6f} ± {sx_err[i]:.2e}"
            f"   (tau_int = {sx_tau[i]:.2f})"
        )
    np.savez(
        "measurements.npz",
        E=results["E"][0],
        E_err=results["E"][1],
        M2=results["M2"][0],
        M2_err=results["M2"][1],
        sx=sx_mean,
        sx_err=sx_err,
        sx_tau=sx_tau,
    )
