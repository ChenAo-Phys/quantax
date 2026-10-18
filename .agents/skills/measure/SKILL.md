---
name: measure
description: Measure observables from NQS samples. Use when estimating expectation values with reliable error bars - uncertainty and binning analysis (autocorrelation), and efficient measurement of diagonal vs off-diagonal operators.
---

# Basic measurement

`op.expectation(state, samples, return_var=True)` returns the estimates of
$\langle O \rangle = \langle O_\mathrm{loc} \rangle$ and
$\mathrm{Var}(O_\mathrm{loc})$, with the local estimator
$O_\mathrm{loc}(s) = \sum_{s'} \langle s|O|s' \rangle \, \psi(s')/\psi(s)$
averaged over samples. For the basics, see `tutorials/samples.ipynb`: it
introduces `ExactSampler` and Metropolis samplers (thermalization, sweep
settings, custom proposals and mixed samplers), the mutable `Samples` (`spins`,
`psi`, `prob`, `state_internal`, `r`, not a PyTree), measuring expectation
values with uncertainty, and sampled
estimators for quantities beyond operators — state overlap (fidelity) and
Rényi-2 entanglement entropy.

The naive uncertainty `sqrt(var / nsamples)` is valid only for **uncorrelated**
samples. Chains in one sweep are independent (after thermalization, see the
checks below), but consecutive sweeps of the same chain are autocorrelated —
for multi-sweep statistics use binning (below).

The energy has the zero-variance property: `Var(Hloc) → 0` as the state
approaches an eigenstate, so energy error bars are far smaller than those of
other observables at equal statistics. Don't extrapolate energy error bars to
other quantities.

# Uncertainty and binning analysis

Feed the local estimators sweep by sweep into a binning analysis along the
sweep axis: the error estimate grows with bin size until it plateaus.
[scripts/measure_binning.py](scripts/measure_binning.py) is a production
template (diagonal, Hamiltonian and packed off-diagonal measurement, with
reweighting and stationarity checks); its `BinningAccumulator`
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
- Weight before binning. `Oloc` returns the raw local estimator, while
  `expectation` weights it by `samples.r` (mean 1 per sweep). `r != 1` for
  `reweight != 2`, and at rounding level whenever `psi` is recomputed after
  sampling (fast local updates, or low precision with the chunk above the
  per-device sample count). Bin the centered
  `z = r * (Oloc - O0)` with `O0` the first-sweep estimate, so
  `<O> = O0 + <z>` (`BinningAccumulator.add(Oloc, r)` does this). Binning
  `r * Oloc` adds `<O>^2 Var(r)`, which can dominate the error of the extensive
  energy unless `Var(r)` is tiny.
- Thermalize first: binning assumes stationary chains. For a trained state,
  start from the chains saved with the checkpoint (save
  `qtx.utils.to_replicated_numpy(sampler._spins)` next to every parameter file
  during training and pass it as `initial_spins`, with exactly the measurement
  sampler's `nsamples` rows; subsample or tile otherwise). From random or stale
  spins, peaked or ordered states can need far more than the default
  `thermal_steps = 20 * Nmodes` (e.g. 5-10x for stripe order). Check that
  the first sweeps reproduce the training E and VarE, and discard leading
  sweeps while they drift; a VarE far above the training value means the
  chains are not thermalized.
- Detect frozen chains: the scatter of the per-sweep mean should be
  `~sqrt(VarE / nchains)`. Zero scatter, bit-identical sweeps, or a VarE far
  below the training value mean stuck chains (a chain at `psi = inf` never
  leaves it). `update_r` silently sets non-finite weights to 0, so also count
  `samples.r == 0`. Re-measure with fresh chains, e.g. `sampler.reset()`.

# Diagonal vs off-diagonal operators

**Diagonal** (only `sigma_z` / number operators): `Oloc(s) = <s|O|s>` is a pure
function of the spins — no forward pass at all. Diagonal
measurements are essentially free — measure as many as useful, and consider
saving `spins` to disk for post-hoc analysis. The only exception is the
reweighting factor `r`: if the samples carry `prob` but no `psi`, a diagonal
`Oloc` / `expectation` spends a separate forward pass on `psi(s)` to update `r`,
see the ordering note below. In a diagonal-only measurement, avoid it by setting
`samples.psi = sampler._psi` and calling `samples.update_r()`. Never do this
when off-diagonal operators follow on the same samples: the sampler drops `psi`
when its forward can differ materially from the padded `state(s)` forward (local
updates, or low precision with a chunk above the per-device sample count), and
`Oloc` would divide numerators from `state(s_conn)` by it, biasing the result.
Measure those first instead; any `psi` you attach yourself must come from
`state(s)`.

**Off-diagonal**: `Oloc` needs forward passes on all connected configurations
`s'` — this dominates the measurement cost. To keep it cheap:

- Set `max_parallel` in `Variational`. It is a per-device chunk of *input*
  configurations: under `symm=`, every forward evaluates chunk x `symm.nsymm`
  images, so budget memory for that and never multiply the chunk by `nsymm`.
  Sampling never pads a batch below the chunk, but every `Oloc` call rounds its
  per-device connected configs (plus the `psi(s)` rows if still missing) up to
  a multiple of the chunk, so each call costs at least one full chunk. Choose
  by regime:
  - Hamiltonian-like operators, with many connected configs per sample: set the
    chunk as large as memory allows, for fewer, bigger forwards, coarser
    rounding and fewer re-jits. The scripts here use 16384, sized for small
    nets; measure the per-config forward memory of heavy models first.
  - Many small operators (a single `sigma_x(i)`, hopping or pair term, a few
    connected configs per sample): a large chunk is pure padding, e.g. 32x the
    needed forward at 512 samples/device and chunk 16384. A symmetry-projected
    measurement of ~4700 such terms ran 7x faster at chunk = samples/device,
    with identical results. Pack the terms (next bullet), or measure them with
    a second state on its own samples, whose fresh `psi` keeps the denominator
    and the numerators in one executable. Pass `symm` again, since
    `state.model` is unprojected:
    ```python
    small = qtx.state.Variational(
        state.model, symm=state.symm, max_parallel=nsamples // ndevices
    )
    small_samples = qtx.sampler.Samples(samples.spins, prob=samples.prob)
    sx0 = sigma_x(0).Oloc(small, small_samples)
    ```
- Pack many small operators into one forward: sum them with `+=` into one
  `Operator` (see Caveats), call `apply_off_diag` once, and evaluate all
  connected configs together. Per update-mode group, it returns configs
  `(nsamples, nterms, Nmodes)` and strengths `(nsamples, nterms)`, with NaN
  marking invalid connections and columns ordered by `OpTerm` (one per
  operator string), in insertion order within each:
  ```python
  ((_, s_conn, x_conn),) = Sx.apply_off_diag(s)  # Sx = sum of sigma_x(i)
  psi_conn = state(s_conn.reshape(-1, s.shape[1])).reshape(x_conn.shape)
  ratio = jnp.asarray(psi_conn / samples.psi[:, None])  # divide LogArrays first
  sxloc = jnp.where(jnp.isnan(x_conn), 0, x_conn * ratio)  # (nsamples, N)
  ```
  `samples.psi` must be filled first: measure the Hamiltonian before, or set
  `samples.psi = state(s)` and call `samples.update_r()`. Keep psi as the
  `LogArray` that `state` returns; `jnp.abs` / `jnp.log` of `state(s)`
  materializes psi and under/overflows on large lattices. `s_conn` holds
  nsamples x nterms configs, invalid ones included, so pack in groups when it
  gets large; packing also forgoes the fast local updates of `RefModel` states.
  Cross-check a few columns against per-term `Oloc` on the first sweep, as
  [scripts/measure_binning.py](scripts/measure_binning.py) does.
- Metropolis samples carry the sweep `psi` and `r`, except for states with fast
  local updates, or in low precision (TF32 / half) when the forward chunk exceeds
  the per-device sample count. A missing `psi(s)` is computed inside the
  connected-config forward pass of the first `Oloc` call and written into the
  samples in place together with `r`, so every later `Oloc` call on the same
  samples reuses it. Don't precompute it when an `Oloc` call comes first (packed
  forwards are the exception, see above), and keep passing the same `Samples`
  object instead of rebuilding it from `samples.spins`.
- Measure the off-diagonal operators first and the diagonal ones afterwards.
  The first off-diagonal `Oloc` fills in `psi` and updates `r` for free, while a
  diagonal operator measured first has to run a separate forward pass for them.
- Reuse one set of samples for all observables; loop sweeps only to grow
  statistics.
- Models with fast local updates (`RefModel`, e.g. fermion determinants) use
  them in `Oloc` automatically when the operator provides the required update
  modes. `Oloc` doesn't keep their internal values, so consider setting
  `samples.psi, samples.state_internal = state.init_internal(samples.spins)` once,
  followed by `samples.update_r()`, so every `Oloc` call reuses them.
- Whenever many off-diagonal operators are measured together, consider
  measuring smaller pieces to save time — e.g. for pair-pair correlations,
  decompose into unique two- and four-fermion terms and assemble the
  correlation afterwards instead of measuring one operator per displacement
  (see [dwave/dwave_pairing.md](dwave/dwave_pairing.md) for the d-wave case,
  split into a GPU measurement stage and a CPU assembly stage so the term
  values stay on device until the periodic saves; `Oloc` itself still syncs
  with the host once per call).

# Caveats

- In `MixSampler`, `nsamples` is the sum of all `nsamples` in sub-samplers.
- Build operators from many terms with an in-place sum, never
  `functools.reduce(operator.add, ops)` or `sum(ops)`: `Operator.__add__`
  deep-copies the growing left operand at every step (quadratic in the number
  of terms), while `+=` appends in place. On a 32x32 lattice this took startup
  from ~46 s to 0.05 s, with an identical result. `+=` mutates its left
  operand, so start from an empty operator:
  ```python
  op = Operator([])
  for term in terms:
      op += term
  ```
