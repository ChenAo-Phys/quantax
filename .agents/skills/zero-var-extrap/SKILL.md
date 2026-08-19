---
name: zero-var-extrap
description: Zero-variance extrapolation of VMC results towards the exact ground state. Use when estimating the ground-state energy from a converged training curve or from a family of variational states - linear-in-variance extrapolation for the energy, sqrt- or linear-variance for other observables (with guidance on choosing), and ODR fitting with errors on both axes via odrpack.
---

# Principle

Write the variational state as $|\psi\rangle \propto |\psi_0\rangle + \varepsilon
|\psi_\perp\rangle$ with $|\psi_0\rangle$ the exact ground state. Then

- $E - E_0 = O(\varepsilon^2)$ and $\mathrm{Var}(E_\mathrm{loc}) =
  O(\varepsilon^2)$, so the **energy is linear in the energy variance**:
  $E \approx k \, \mathrm{Var}E + E_0$. Fitting a line and reading the
  intercept at $\mathrm{Var}E = 0$ removes the leading variational bias.
- A generic observable has a first-order error in the worst case,
  $\langle O \rangle - \langle O \rangle_0 = O(\varepsilon)$, so it is
  **linear in $\sqrt{\mathrm{Var}E}$**; when the first-order term is
  suppressed it becomes linear in $\mathrm{Var}E$ like the energy. See
  "Observables other than energy" below for how to choose.

Normalizing the variance (per site, or $\mathrm{Var}E/E^2$) only rescales the
slope; the intercept is unchanged. Don't mix differently normalized points in
one fit.

# Data sources

Each fit point is a pair (VarE, y) with error bars on both. Two ways to
generate them:

1. **Tail of a training curve** (cheap, energy only). After convergence a
   stochastic optimizer fluctuates around the optimum, so the per-step
   `(energy, VarE)` records scatter along the extrapolation line. Quantax
   optimizers expose `.energy` and `.VarE` after each step; record them
   during training, e.g. `np.savetxt` with one row `[step, energy, VarE]`
   per optimization step. Bin consecutive steps (mean ± std/sqrt(bin) per
   bin) to suppress autocorrelation.
2. **A family of converged states** (standard for observables): increasing
   bond dimension, network width/depth, or added symmetry projections. Each
   state contributes one point — measure both together via
   `H.expectation(state, samples, return_var=True)` — with error bars from
   the binning analysis of the [measure](../measure/SKILL.md) skill. This spans a wider variance
   range than a single training tail and gives a more trustworthy intercept.

# ODR fit with odrpack

Since both VarE and y carry uncertainty, use orthogonal distance regression,
not OLS (errors in x bias the OLS slope towards zero — regression dilution).
`scipy.odr` is deprecated and will be removed in SciPy 1.19.0; use
[odrpack](https://pypi.org/project/odrpack/) (`pip install odrpack`), which
wraps the same ODRPACK95 library:

```python
from odrpack import odr_fit
import numpy as np

res = odr_fit(
    lambda x, beta: beta[0] * x + beta[1],  # note: f(x, beta), not f(beta, x)
    x, y,
    beta0=np.polyfit(x, y, 1),          # OLS initial guess
    weight_x=1 / x_err**2,              # weights, not standard deviations
    weight_y=1 / y_err**2,
)
k, b = res.beta
k_err, b_err = res.sd_beta
```

Differences from the legacy `scipy.odr`:

- The model signature is `f(x, beta)` instead of `f(beta, x)`.
- Errors enter as weights `1/err**2` (`weight_x`/`weight_y`) instead of
  standard deviations (`sx`/`sy` in `RealData`).
- Use `res.sd_beta` for parameter errors — it is already scaled by
  `res.res_var`. The habit of taking `sqrt(diag(cov_beta))` underestimates
  the error by a factor `sqrt(res_var)` in both libraries.

Since both packages wrap the same Fortran library, the fitted parameters and
errors are identical up to numerical noise.

# Script

[scripts/zero_var_extrap.py](scripts/zero_var_extrap.py) implements the full
pipeline (binning, error propagation, ODR fit, plot) as importable functions
and a CLI. For a training-curve file with columns `[step, E, VarE, ...]`:

```bash
python zero_var_extrap.py training_curve.txt --start 1000 --end 10000 --bin 50 --plot fit.pdf
```

prints the intercept (the zero-variance estimate), slope, and `res_var`, and
saves a plot of the binned points, colored by training step, with the fitted
line extended to VarE = 0. Typical output:

```
intercept = -49.102346 ± 0.000556
slope = 0.124627 ± 0.001620
res_var = 1.114
```

Here `--start`/`--end` select the converged tail of the run (inspect the
energy curve first) and `--bin` sets how many consecutive steps form one fit
point.

For a generic observable stored in column `c`, extrapolate in
$\sqrt{\mathrm{Var}E}$ with `--ycol c --power 0.5`. For points from a family
of states with measured error bars, skip the binning and call
`zero_var_extrap(x, y, x_err, y_err)` directly.

# Observables other than energy

- Measure the observable and VarE **on the same state** (same checkpoint,
  same symmetry projection) so each point is consistent; reuse one set of
  samples for both.
- **Choosing sqrt vs linear.** The bare estimator of a generic observable has
  a first-order bias, $\langle O\rangle - \langle O\rangle_0 =
  2\varepsilon\,\mathrm{Re}\langle\psi_0|O|\psi_\perp\rangle + O(\varepsilon^2)$
  (Assaraf & Caffarel 2003), which gives the worst-case scaling
  $\langle O\rangle \approx b + k\sqrt{\mathrm{Var}E}$ — the conservative
  default (`power=0.5`). But the first-order coefficient can be strongly
  suppressed: when the residual $|\psi_\perp\rangle$ spreads over many states
  whose contributions cancel in sign (the argument used in the Monte Carlo
  shell model, where occupation numbers and quadrupole moments are
  extrapolated *linearly* in the variance, Shimizu et al. 2010), or when a
  symmetry of the ansatz and of $O$ forbids the cross term. Then the
  $O(\varepsilon^2) \propto \mathrm{Var}E$ term dominates and `power=1` is
  correct. In practice, fit **both powers** and let the data decide — the
  scaling with better linearity (lower `res_var`, no visible curvature) wins;
  when the data cannot discriminate, report the spread of the two intercepts
  as a systematic uncertainty. Note the two disagree most near the origin,
  where the sqrt form has infinite slope.
- **Expect much lower accuracy than for the energy.** (i) Observables lack
  the zero-variance property, so each point carries far larger statistical
  error bars at equal statistics — the training-tail source is usually too
  noisy, and the multi-state approach with long measurement runs is
  preferred. (ii) The lever arm is weaker under sqrt scaling: reducing VarE
  by 100x shrinks the remaining bias only 10x. (iii) The functional-form
  ambiguity above adds a systematic error the energy doesn't have. Quote
  extrapolated observables with these systematics, not just the fit error.
- The slope can have either sign, and different observables extrapolate
  independently — there is no reason for one state's bias to cancel across
  quantities.

# Pitfalls

- **Fit window**: start after the energy has converged — points from the
  descending phase don't fluctuate around one optimum and bend the line.
  The linear law is only asymptotic in $\varepsilon$; if the plot shows
  curvature, drop the largest-variance points.
- **Systematic vs statistical error**: `sd_beta` is the statistical error of
  the fit only. Vary `start` and the fit window and take the scatter of the
  intercept as the systematic uncertainty; it often dominates.
- **Sanity check `res_var`** (printed by the script): should be ~1. Much
  larger means underestimated error bars (autocorrelation — increase the bin
  size until the intercept error stabilizes, like a binning analysis) or a
  nonlinear window.
- **Variance spikes**: occasional sampling spikes give bins with huge VarE.
  They stay in the fit (ODR downweights them through their large `x_err`),
  and the script's plot uses a robust quantile x-limit so they don't blow up
  the figure.
- **Extrapolation distance**: the intercept is only as good as the smallest
  variance reached. A training tail clusters in a narrow VarE range, so the
  extrapolated error bar shrinks but the leverage is small; points spanning a
  decade of variance (multi-state) constrain the line far better.
- **Wide variance windows**: the linear law holds near convergence. If the
  points span a large variance range, shell-model practice fits the energy
  with a second-order polynomial in the variance instead (Shimizu et al.
  2010); prefer restricting the window to the linear regime.
