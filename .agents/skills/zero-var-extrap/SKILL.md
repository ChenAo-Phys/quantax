---
name: zero-var-extrap
description: Zero-variance extrapolation of VMC results towards the exact ground state. Use when estimating the ground-state energy from the tail of a training curve or from a family of variational states - linear-in-variance extrapolation for the energy, sqrt- or linear-variance for other observables (with guidance on choosing), ODR fitting with errors on both axes via odrpack, a window-grid error estimate, and lever-arm and variational-bound checks for unidentified slopes.
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

1. **Tail of a training curve** (cheap, energy only). Late in training,
   (energy, VarE) drift together towards VarE = 0 along the extrapolation
   line as the state slowly improves. The slope usually comes from this slow
   drift, not from step-to-step noise (check: the rank correlation of E and
   VarE after subtracting a ~100-step running mean is ~0), so the intercept
   depends on the fit window (see the window grid in Pitfalls). Quantax
   optimizers expose `.energy` and `.VarE` after each step; record them
   during training, e.g. `np.savetxt` with one row `[step, energy, VarE]` per
   optimization step. Bin consecutive steps (mean ± std/sqrt(bin) per bin) to
   suppress autocorrelation.
2. **A family of converged states** (standard for observables): a
   *systematic* sequence within one ansatz family, such as increasing bond
   dimension, or width/depth at fixed architecture. Each state contributes
   one point — measure both together via
   `H.expectation(state, samples, return_var=True)` on chains thermalized
   until VarE stops drifting — with error bars from the binning analysis of
   the [measure](../measure/SKILL.md) skill. This spans a wider variance
   range than a single training tail and gives a more trustworthy intercept.
   Different architectures, or projected vs unprojected versions of a state,
   need not share a slope: a symmetry projection can lower E at nearly
   unchanged VarE and move the state off its parent's line. Fit each family
   separately and compare the intercepts; pool them only if the joint fit has
   `res_var` ~ 1 and unstructured residuals. A pooled `res_var` >> 1 means
   the families don't share one line, and more statistics won't fix it.

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
    maxit=300,                          # default 50 stops early on weak data
)
assert res.success, res.stopreason
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

odrpack is not a quantax dependency (not even in the `[full]` extra), so
cluster environments often lack it: install it there, or copy the training
log, after the job's last write, to a machine that has it.

# Script

[scripts/zero_var_extrap.py](scripts/zero_var_extrap.py) implements the full
pipeline (binning, error propagation, ODR fit, plot, window grid) as
importable functions and a CLI. For a training-curve file with columns
`[step, E, VarE, ...]`:

```bash
python zero_var_extrap.py training_curve.txt --start 1000 --end 10000 --bin 50 --plot fit.pdf
```

prints the intercept (the zero-variance estimate), slope, `res_var` and the
lever arm `R` (see Pitfalls), warns when the window does not identify the
slope, and saves a plot of the binned points, colored by training step, with
the fitted line extended to VarE = 0. Typical output:

```
intercept = -49.993614 ± 0.006705
slope = 0.394088 ± 0.005693
res_var = 1.053 (should be ~1; larger means underestimated errors or nonlinearity)
lever arm R = 2.4 (should be a few; intercept distance in units of the VarE drift)
```

`--start`/`--end` are values of the step column (`--stepcol`, default 0; `-1`
uses the row index), start inclusive and end exclusive; the default end is
the end of the file. Choose both from the energy AND a running median of VarE
(see Pitfalls). `--bin` sets how many consecutive steps form one fit point.
Several values for `--start`, `--end` and `--bin` scan the window grid
(`grid_extrap` in Python) and print every window plus the grid estimate with
its statistical and window errors.

For a generic observable stored in column `c`, extrapolate in
$\sqrt{\mathrm{Var}E}$ with `--ycol c --power 0.5`. For points from a family
of states with measured error bars, skip the binning and call
`zero_var_extrap(x, y, x_err, y_err)` directly; it warns instead of raising
when ODR doesn't converge, so check `res.success`, and `check_fit` runs the
identifiability checks. `check_fit` requires `energy=`: pass `energy=True` for
the energy, which enables the variational-bound check, and `energy=False` for
other observables, whose slope may have either sign. `grid_extrap` infers it
from `ycol` and `power` like the CLI unless `energy` is given.

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

- **Fit window**: exclude the fast early descent and any restart transient
  (they bend the line), but don't restrict the fit to a fully flat tail: a
  window without VarE drift has no lever arm. Choose the window from a
  running median of VarE over a few hundred steps, not only from the energy.
  After a warm restart from stale or unthermalized chains (`thermal_steps=0`,
  an old saved spin configuration) or when a symmetry projection is switched
  on mid-run, VarE first rises (tens of percent while E still falls, or
  several-fold with E jumping up too when the chains are stale), and
  optimizer moments need ~100s of steps to recover; start after that. Choose
  the end deliberately too: if VarE flattens or rises again late in the run,
  stop before that, since running to the end of the file can shift the
  intercept by more than its error. The linear law is only asymptotic in
  $\varepsilon$; if the plot shows curvature, drop the largest-variance
  points.
- **Systematic vs statistical error**: `sd_beta` is the statistical error of
  one fit only. Scan a grid of windows, start × end × bin, with >= 4 binned
  points per fit (several CLI values, or `grid_extrap`). Drop non-converged
  fits, fits violating the variational bound, and fits whose `sd_beta[1]` is
  far above the grid's typical value, and report how many were dropped. Quote
  the grid mean as E0, the **mean** `sd_beta[1]` as the statistical error,
  and the std of the intercepts as the systematic error of the window choice.
  All windows reuse the same steps, so never divide by sqrt(#windows). Don't
  filter on slope significance (windows whose `sd_beta[1]` rivals the
  extrapolation distance $|k|\,\overline{x}$): that keeps the windows whose
  noise happened to line up and quotes a falsely small error. Keep the weak
  windows and count them (the CLI does); if many are weak, call the estimate
  weakly constrained.
- **Sanity check `res_var`** (printed by the script): should be ~1. Much
  larger means underestimated error bars (autocorrelation — increase the bin
  size until the intercept error stabilizes, like a binning analysis), a
  nonlinear window, or points from different ansatz families (see Data
  sources). But `res_var` ~ 1 does not certify the slope (see Lever arm).
- **Lever arm / identifiability**: the script prints
  $R = \mathrm{median}(\mathrm{Var}E) / \Delta$, with $\Delta$ the VarE drift
  across the window (spread of the medians of 4 consecutive chunks). The
  intercept sits $R$ drift-spans from the data, so its error grows like $R$
  times the slope error. A few spans is fine. At $R \gtrsim 10$ (VarE drifts
  by about 10% or less across the window) the slope comes from a tiny drift,
  so treat stat ⊕ syst as a lower bound. For the energy, also require
  $k > 0$ and an intercept below every fitted energy (variational bound; the
  script warns): a flat tail can give a converged fit with `res_var` ~ 1 and
  a negative slope whose "E0" lies above the variational energy. Points
  spanning a decade of VarE (multi-state, one ansatz family) constrain the
  line far better.
- **Too-low training VarE**: the logged (E, VarE) come from the training
  chains, and three things make VarE look too small and drag the intercept:
  frozen chains (e.g. stuck on a configuration with $\psi = \infty$, which
  Metropolis never leaves), a biased sampler, and a heavy-tailed
  $E_\mathrm{loc}$ (near-singular determinant or Pfaffian amplitudes) that a
  training batch under-samples. Samples with non-finite $E_\mathrm{loc}$ are
  dropped from both E and VarE; only the warning "N sample(s) with non-finite
  local energy excluded." shows it. Before trusting a training-tail fit,
  re-measure one or two checkpoints in the window with freshly thermalized
  chains (measure skill): VarE should match the training value (much larger
  means unthermalized measurement chains or a training artifact), and the
  detrended step-to-step scatter of the training E should be about
  sqrt(VarE / nsamples). Lower VarE at higher E than sibling runs is a red
  flag.
- **Variance spikes**: occasional sampling spikes give bins with huge VarE.
  They stay in the fit (ODR downweights them through their large `x_err`),
  and the script's plot uses a robust quantile x-limit so they don't blow up
  the figure.
- **Wide variance windows**: the linear law holds near convergence. If the
  points span a large variance range, shell-model practice fits the energy
  with a second-order polynomial in the variance instead (Shimizu et al.
  2010) or as a cross-check of the linear intercept; prefer restricting the
  window to the linear regime. On a narrow training tail the quadratic term
  is unconstrained.
