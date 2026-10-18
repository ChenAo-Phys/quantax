"""Production measurement of observables with binning error analysis.

Measures a diagonal observable (M^2) directly from the spins and off-diagonal
ones: H via `Oloc`, and the local <sx_i> at every site packed into one
connected-config forward instead of one `Oloc` call per site. Local estimators
are fed per sweep into streaming binning accumulators, which give the mean, the
integrated autocorrelation time tau, and the uncertainty
sqrt(var * (1 + 2 tau) / nsamples_total) for every quantity separately.

Every estimator is weighted by the reweighting factor `samples.r` (mean 1 per
sweep) and centered by its first-sweep estimate O0 before binning: the
accumulators bin z = r (Oloc - O0), never the raw product r * Oloc, whose
variance carries an extra <O>^2 Var(r).

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

# On multi-node jobs, initialize distributed JAX here, before importing quantax,
# with the snippet of "Distributed initialization" in the hpc-usage skill.

import quantax as qtx
from quantax.operator import sigma_x, sigma_z, Operator

is_main = jax.process_index() == 0


class BinningAccumulator:
    """Streaming binning analysis of local estimators.

    Feed one batch of shape (nchains,) or (nchains, nobs) per sweep with
    `add`. Bins along the sweep axis double in size level by level; each
    level keeps only running sums (on device, sample-sharded), never the
    full series. Estimators are centered by the first-sweep estimate so that
    float32 accumulation doesn't lose the variance to cancellation.
    """

    def __init__(self):
        self._levels = []
        self._offset = None

    def add(self, Oloc, r=None):
        """Add one sweep of local estimators `Oloc`, weighted by the reweighting
        factor `r` = `samples.r` (mean 1 per sweep, None for r = 1)."""
        x = jnp.asarray(Oloc)  # (nchains,) or (nchains, nobs)
        if x.ndim == 1:
            x = x[:, None]
        if r is not None:
            r = jnp.asarray(r)[:, None]
        if self._offset is None:  # first-sweep estimate O0
            self._offset = (x if r is None else r * x).mean(axis=0)
        x = x - self._offset
        if r is not None:
            x = r * x  # z = r (Oloc - O0), so <O> = O0 + <z>
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
        Only bin sizes with at least `min_bins` bins are used.
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
        if not errors:
            raise ValueError(f"Binning needs NSWEEPS >= min_bins = {min_bins}.")
        errors = np.stack(errors)

        L0 = self._levels[0]
        mean = self._offset + L0["s1"].sum(axis=0) / (L0["n"] * L0["s1"].shape[0])
        tau = 0.5 * ((errors[-1] / errors[0]) ** 2 - 1)
        if len(errors) < 2:
            if is_main:
                print(
                    "WARNING: fewer than 2 binning levels, tau and plateau not "
                    f"checked; use NSWEEPS >= {2 * min_bins}"
                )
        elif is_main and np.any(errors[-1] > 1.2 * errors[-2]):
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

NCHAINS = 512
sampler = qtx.sampler.LocalFlip(
    state, nsamples=NCHAINS, thermal_steps=50 * N, sweep_steps=2 * N
)

# save thermalized chains
spins_full = qtx.utils.to_replicated_numpy(sampler._spins)
if is_main:
    np.save("spins.npy", spins_full)
del spins_full

# Prefer starting from the chains saved with the trained checkpoint: from random
# or stale spins, peaked or ordered states can need far more thermal steps than
# above. The same rebuild also restarts a collapsed run without paying
# thermalization again:
# sampler = qtx.sampler.LocalFlip(
#     state,
#     nsamples=NCHAINS,
#     sweep_steps=2 * N,
#     thermal_steps=0,
#     initial_spins=jnp.load("spins.npy"),
# )

NSWEEPS = 128

M = Operator([])  # in-place sum, see the caveats in SKILL.md
for i in range(N):
    M += sigma_z(i)
M /= N
M2 = M @ M

# all sx_i in one operator, measured by one packed forward per sweep
Sx = Operator([])
for i in range(N):
    Sx += sigma_x(i)
CHECK_SITES = np.arange(0, N, max(1, N // 8))  # sweep-0 check against per-term Oloc

acc_E = BinningAccumulator()
acc_M2 = BinningAccumulator()
acc_sx = BinningAccumulator()
E_sweep, VarE_sweep = [], []  # per-sweep scalars on device, for stationarity checks
n_zero_r = 0

for n in range(NSWEEPS):
    samples = sampler.sweep()
    s = samples.spins

    # Off-diagonal first: H.Oloc fills a missing psi(s) and the reweighting factor
    # r in place, so later calls on the same samples reuse them.
    Eloc = H.Oloc(state, samples).real
    r = samples.r  # mean 1 per sweep; None means r = 1
    w = jnp.ones_like(Eloc) if r is None else r
    acc_E.add(Eloc, r)
    E_sweep.append(jnp.mean(w * Eloc))
    VarE_sweep.append(jnp.mean(w * (Eloc - E_sweep[-1]) ** 2))
    n_zero_r = n_zero_r + jnp.sum(w == 0)  # non-finite weights are zeroed

    # diagonal observable: needs only the spins
    acc_M2.add(M2.apply_diag(s).real, r)

    # Packed off-diagonal: one connected-config pass and one forward for all sites,
    # instead of N Oloc calls that each pay at least one full forward chunk.
    ((_, s_conn, x_conn),) = Sx.apply_off_diag(s)  # (nchains, N, Nmodes), (nchains, N)
    psi_conn = state(s_conn.reshape(-1, s.shape[1])).reshape(x_conn.shape)
    # divide the LogArrays before converting; never jnp.abs/jnp.log a psi
    ratio = jnp.asarray(psi_conn / samples.psi[:, None])
    sxloc = jnp.where(jnp.isnan(x_conn), 0, x_conn * ratio).real  # (nchains, N)
    if n == 0:
        sx_ref = jnp.stack(
            [sigma_x(i).Oloc(state, samples).real for i in CHECK_SITES], axis=1
        )
        diff = jnp.max(jnp.abs(sxloc[:, CHECK_SITES] - sx_ref))
        if not diff <= 1e-3 * jnp.max(jnp.abs(sx_ref)):
            raise RuntimeError(f"packed sx differs from per-term Oloc by {diff}")
    acc_sx.add(sxloc, r)

# --- results: uncertainty = sqrt(var * (1 + 2 tau) / ntotal) per observable ---
results = {}
for name, acc in [("E", acc_E), ("M2", acc_M2)]:
    mean, err, tau, bin_sizes, errors = acc.results()
    results[name] = (mean[0], err[0])
    if is_main:
        print(f"\n{name} = {mean[0]:.6f} ± {err[0]:.2e}   (tau_int = {tau[0]:.2f})")
        for b, e in zip(bin_sizes, errors):
            print(f"  bin size {b:4d}: error = {e[0]:.3e}")

# Stationarity and frozen chains: the first sweeps should match the training E and
# VarE, and the scatter of sweep means should be ~ sqrt(VarE / nchains).
E_sweep = np.asarray(jnp.stack(E_sweep))
VarE_sweep = np.asarray(jnp.stack(VarE_sweep))
n_zero_r = int(n_zero_r)
if is_main:
    print(f"\nfirst sweeps: E {E_sweep[:4]}, VarE {VarE_sweep[:4]}")
    scatter, expected = E_sweep.std(), np.sqrt(VarE_sweep.mean() / NCHAINS)
    print(f"scatter of sweep means {scatter:.2e}, expected ~{expected:.2e}")
    if n_zero_r > 0:
        print(f"WARNING: {n_zero_r} samples with zero weight (non-finite psi)")

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
        E_sweep=E_sweep,
        VarE_sweep=VarE_sweep,
    )
