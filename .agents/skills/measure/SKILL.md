---
name: measure
description: Measure observables from NQS samples. Use when estimating expectation values with reliable error bars - uncertainty and binning analysis (autocorrelation), efficient measurement of diagonal vs off-diagonal operators, and fp16 sampling corrected by reweighting.
---

# Basic measurement

`op.expectation(state, samples, return_var=True)` returns the estimates of
$\langle O \rangle = \langle O_\mathrm{loc} \rangle$ and
$\mathrm{Var}(O_\mathrm{loc})$, with the local estimator
$O_\mathrm{loc}(s) = \sum_{s'} \langle s|O|s' \rangle \, \psi(s')/\psi(s)$
averaged over samples. For the basics, see `tutorials/samples.ipynb`: it
introduces `ExactSampler` and Metropolis samplers (thermalization, sweep
settings, custom proposals and mixed samplers), the sample PyTree (`spins`,
`psi`, ...), measuring expectation values with uncertainty, and sampled
estimators for quantities beyond operators — state overlap (fidelity) and
Rényi-2 entanglement entropy.

The naive uncertainty `sqrt(var / nsamples)` is valid only for **uncorrelated**
samples. Chains in one sweep are independent (after thermalization), but
consecutive sweeps of the same chain are autocorrelated — for multi-sweep
statistics use binning (below).

The energy has the zero-variance property: `Var(Hloc) → 0` as the state
approaches an eigenstate, so energy error bars are far smaller than those of
other observables at equal statistics. Don't extrapolate energy error bars to
other quantities.

A caveat: In `MixSampler`, `nsamples` is the sum of all `nsamples` in sub-samplers.

# Uncertainty and binning analysis

Feed the local estimators sweep by sweep into a binning analysis along the
sweep axis: the error estimate grows with bin size until it plateaus.
[scripts/measure_binning.py](scripts/measure_binning.py) is a production
template (diagonal + off-diagonal measurement); its `BinningAccumulator`
streams the binning with O(log nsweeps) partial sums, so many observables
(e.g. one per site) each get their own binning curve and tau without storing
the full `(nsweeps, nchains, nobs)` series — a host-memory OOM risk in large
runs.

- The plateau value is the true error; the naive estimate (bin size 1)
  underestimates it by `sqrt(1 + 2 tau)`. If the curve is still rising at the
  largest bin, run more sweeps — the error bar is not converged.
- Integrated autocorrelation time: `tau = ((err_plateau / err_1)**2 - 1) / 2`;
  effective sample count is `nsamples_total / (1 + 2 tau)`.
- Prefer many chains over many sweeps when memory allows: chains are
  independent, so they need no binning; thermalization is paid once at sampler
  construction. When `Oloc` is expensive (off-diagonal `op`), raising
  `sweep_steps` buys less-correlated samples at unchanged sampling cost but
  fewer `Oloc` evaluations.

# Diagonal vs off-diagonal operators

**Diagonal** (only `sigma_z` / number operators): `Oloc(s) = <s|O|s>` is a pure
function of the spins — no forward pass at all. Diagonal
measurements are essentially free — measure as many as useful, and consider
saving `spins` to disk for post-hoc analysis.

**Off-diagonal**: `Oloc` needs forward passes on all connected configurations
`s'` — this dominates the measurement cost. To keep it cheap:

- Set `max_parallel` in `Variational`; connected configs are padded to the
  forward chunk, avoiding memory overflow and re-jitting.
- Metropolis samples carry `psi=None`; `psi(s)` is computed inside the same
  connected-config forward pass, so don't precompute it for a single operator.
  When measuring **many** operators on the same sweep, compute `psi = state(s)`
  once and attach it via `Samples(s, psi)` so every `Oloc` call reuses it.
- Reuse one set of samples for all observables; loop sweeps only to grow
  statistics.
- Models with fast local updates (`RefModel`, e.g. fermion determinants) use
  them in `Oloc` automatically when the operator provides the required update
  modes. Consider computing `psi, internal = state.init_internal(s)` once and attaching
  it via `Samples(s, psi, internal)` so every `Oloc` call reuses it.
- Whenever many off-diagonal operators are measured together, consider
  measuring smaller pieces to save time — e.g. for pair-pair correlations,
  decompose into unique two- and four-fermion terms and assemble the
  correlation afterwards instead of measuring one operator per displacement
  (see [dwave/dwave_pairing.md](dwave/dwave_pairing.md) for the d-wave case,
  split into a GPU measurement stage and a CPU assembly stage so large
  multi-node runs aren't slowed by per-term host syncs).

# Mixed-precision (fp16) sampling

Sampling — often a major cost — can run in half precision without giving up
exactness: an fp16 copy of the model samples the slightly wrong distribution
`q ∝ |psi16|^2`, and reweighting corrects it to the target `p ∝ |psi|^2`.
This is ordinary importance sampling, already built into quantax: attach
`r = (p/q) / <p/q>_q` as `Samples.reweight_factor` (see its docstring) and
every consumer — `op.expectation`, the optimizer gradient estimators —
computes the unbiased `<O>_p = <r * Oloc>_q`. fp16 only shapes the proposal
distribution; the weights and all estimators are evaluated in high precision.
To avoid data overflow, consider using fp16 in neural networks, while transforming outputs back to fp32 before final layers like `exp` or tensor contractions.

Use fp16 sampling only when sampling dominates the total cost — most notably
when measuring diagonal operators, whose `Oloc` needs no forward pass at all.
When many off-diagonal operators are measured on the same samples (e.g. the
pair-pair correlations above), the `Oloc` forward passes cost far more than
the sampling, so the fp16 speedup is marginal — sample in fp32 there for
better reliability.

Sweep with the fp16 sampler and build the reweighting factor from the
ratio of the two wavefunctions on the sampled spins, `psi = state(spins)`
and `psi16 = sampler._psi` (the sampler keeps the wavefunction of its
current spins, so no extra fp16 forward is needed):
`r = abs(psi / psi16) ** 2; r = jnp.asarray(r / r.mean())` — normalize
before leaving the log representation. The factor returned by `sweep()`
is trivial here — it corrects `|psi16|^n → |psi16|^2`, not fp16 → fp32 —
so replace it: `samples = Samples(spins, psi, None, r)`. Attaching the
fp32 `psi` lets every measurement reuse it, so the reweighting adds no
forward pass at all.

## Uncertainty and autocorrelation with reweighting

A single `expectation` call on one sweep gives the naive error
`sqrt(var / Ns_eff)` with `Ns_eff = Ns / <r^2>` — valid because chains are
independent. Multi-sweep runs need the binning analysis (above), with two
caveats specific to reweighting
([scripts/fp16_sampling.py](scripts/fp16_sampling.py)):

- The reweighted mean is self-normalized, `<O> = <w Oloc> / <w>` with the raw
  weight `w = |psi/psi16|^2`. Normalize the weights over the **whole run**,
  not sweep by sweep, which would make the sweeps inconsistently normalized.
  For fp16 `w ≈ 1`, so the raw ratio accumulates safely without rescaling.
- Bin the **centered** weighted estimator `z = w * (Oloc - <O>)` and take
  `err = err_bin(z) / <w>`. Binning the raw product `w * Oloc` inflates the
  variance by `<O>^2 Var(w)` — for the energy, whose `|<O>|` is extensive
  while `Var(Oloc)` is suppressed by the zero-variance property, this false
  term can dominate the true error. Exact centering also makes `mean(z) = 0`,
  so the fluctuations of the denominator `<w>` drop out of the error at first
  order.
- The script streams the two columns `[z, w]` per sweep through the
  `BinningAccumulator` of
  [scripts/measure_binning.py](scripts/measure_binning.py). Streaming can't
  center with the exact global `<O>` (known only at the end), so `z` is
  centered with the first-sweep estimate `O0`: the estimate
  `<O> = O0 + <z>/<w>` is exact for any `O0`, and the mis-centering enters
  the variance only at second order, `(O0 - <O>)^2 Var(w)`.

## Cost and validity

- The reweighted estimator is unbiased (up to a negligible `O(1/Ns)`
  self-normalization bias). The cost is a reduced effective sample size,
  `Ns_eff = Ns * <w>^2 / <w^2>`. For fp16 the weights stay close to 1, so `Ns_eff ≈ Ns`.
- The fp16 wavefunction must still track the high-precision one closely:
  otherwise the fp16 Markov chain itself degrades (acceptance collapses,
  ergodicity lost) and the weights grow heavy tails. Monitor the ESS
  fraction — a visible drop below 1 signals trouble.

Cautions:

- Models with complex parameters cannot be cast (JAX has no complex32).
- Fermionic determinant `RefModel`s are poor fp16 candidates: linear-algebra
  kernels and accumulated local-update error need higher precision.
- Prefer fp16 over bf16: similar speed, ~10x finer precision.
