---
name: symm-test
description: Measure the overlap between a variational state and its symmetry-transformed copies to decide whether a symmetry projection is worth applying. Use when choosing projection symmetries for a trained state, diagnosing symmetry breaking (AFM/stripe order), or verifying architecture-enforced symmetries.
---

# Symmetry overlap test

For each group element $T$ of a `Symmetry`, estimate on samples from $|\psi|^2$

$$
O_T = \frac{\langle\psi|T|\psi\rangle \langle\psi|T^{-1}|\psi\rangle}
           {\langle\psi|\psi\rangle^2}
    = \mathrm{mean}\!\left[\sigma_T(s)\frac{\psi(Ts)}{\psi(s)}\right]
      \mathrm{mean}\!\left[\sigma_{T^{-1}}(s)\frac{\psi(T^{-1}s)}{\psi(s)}\right],
$$

the **squared** normalized overlap ($\sigma$ = fermionic permutation sign, 1 for
spins). Interpretation:

- $O_T \approx 0$: the state breaks $T$ — projection onto this element is
  pointless (the projected state is essentially orthogonal or ill-defined).
- $O_T \sim 0.9$: approximate symmetry — projection helps restore the
  symmetric ground state. This is the regime worth projecting.
- $O_T \approx 1$: already symmetrized — projection won't improve anything.

Reference implementation: [scripts/symm_overlap.py](scripts/symm_overlap.py)
(`get_overlap(state, samples, symm)`, handles fermion signs and spin-system
Z2 inversion). Set `samples.psi = state(samples.spins)`. Samples should come
from $|\psi|^2$ (`reweight=2`); otherwise also set `prob` and call
`samples.update_r()`, and `get_overlap` weights by `samples.r`.

## Estimator facts (don't mistake these for bugs)

- $O_T = O_{T^{-1}}$ **exactly** by construction, and elements related by an
  architecture-exact symmetry give exactly equal $O_T$ — repeated values in
  the output are consistency, not a copy-paste error.
- The identity element gives $1 \pm 0$; any element enforced exactly by the
  network (CNN translations; sublattice-embedding periodicity, e.g.
  `sublattice=(Lx, 4)` makes y-translation by 4 exact) gives
  $1.0000 \pm 0.0000$ with strictly zero variance. Both are free sanity checks
  that the permutation/sign machinery is right.
- Strictly zero variance needs a forward equivariant up to summation order
  (plain CNN). Ill-conditioned steps amplify rounding — the condition number of
  a det/Pf occupied block, or the cancellation factor
  $\max_t|\psi(T_t s)|/|\Psi(s)|$ of an in-model symmetry average — so under
  TF32 (`quantax.utils.is_low_precision`) exact elements can read
  $O_T \approx 1$ with ratios $x = \sigma_T\psi(Ts)/\psi(s)$ off 1 by
  $10^{-2}$ or more. Print $\max|x-1|$ and rerun in a fresh process with
  `jax.config.update("jax_default_matmul_precision", "highest")` before building
  the state: rounding shrinks by orders of magnitude, a defect stays (index/sign
  bug; per-site `Embedding(sublattice=...)` under an in-model translation
  average — `state.nparams` is the tell).
- Error bar: $O_T$ is a product of two means estimated on the **same** samples
  — propagate with the covariance term,
  $\mathrm{Var}(O) = \bar y^2\mathrm{Var}(\bar x) + \bar x^2\mathrm{Var}(\bar y) + 2\bar x\bar y\,\mathrm{Cov}(\bar x,\bar y)$. One sweep of many chains →
  chains independent → naive $/n$ errors valid **if equilibrated**. The
  default start is random configs plus `thermal_steps = 20 * Nmodes` Metropolis
  steps. Domains of strongly ordered states (stripes, AFM) coarsen diffusively
  under local moves, with a relaxation time growing ~$L^2$ — at large $L$ many
  times the default — so pass the training chains as `initial_spins=` or raise
  `thermal_steps`. $O_T$ (or $E$, VarE) still drifting between successive
  sweeps means unthermalized.
- One sample set serves all symmetries and elements (1 forward per element
  per sample; $O_{T^{-1}}$ reuses the $T^{-1}$ column); sampling dominates the
  cost, the overlaps themselves are cheap.

## Quantax transformation mechanics

- Transformed config: `spins[:, symm._perm[g]]`. The inverse $T^{-1}$ is the
  group element $g'$ with `perm[g'] == argsort(perm[g])`; reuse its sign
  column and forward. With anti-periodic boundaries the fermion sign of
  $T^{-1}$ differs from that of $T$ (`_perm_sign` rows belong to their own
  element), so pairing `argsort(perm[g])` with `_perm_sign[g]` is wrong.
  Characters are ignored — the diagnostic assumes the trivial sector.
- `state(s)` returns the **projected** amplitude of
  `Variational(..., symm=S)`: the projection lives in Variational's forward,
  not in `state.model`. Elements of S give $O_T = 1$ by construction, another
  free check. To decide on symmetries beyond S, test the projected state; to
  test the bare network, load the same `param_file` into a `Variational` with
  the default `symm`. Never shortcut with `jax.vmap(state.model)`: it silently
  skips `Variational.symm`.
- Form $\psi(Ts)/\psi(s)$ as a `LogArray` ratio, then `jnp.asarray(ratio)`
  (also fine for plain-array models). `jnp.abs`/`jnp.log`/`.value()` on
  $\psi$ itself underflows in fp32 once $\log|\psi| \lesssim -100$.
- Fermion sign: use `quantax.symmetry.symmetry._permutation_sign` (private).
  It is vmapped **over group elements** for a single configuration — vmap it
  over samples yourself: `jax.vmap(_permutation_sign, in_axes=(0, None, None))`.
- `SpinInverse()` is two different objects: for spinful **fermions** it is a
  mode permutation (lives in `_perm`); for **spin** systems it is a Z2
  inversion — `_perm` holds only the identity and the flip lives in
  `symm.Z2_inversion`, so the transformed config is `-s`. A `_perm`-only
  implementation silently measures the identity and reports overlap 1.
- Composite generators (e.g. translate-then-spin-flip for fermions): compose
  permutations as `g1[g2]` and wrap in `Symmetry(g_composed)`. This drops the
  boundary signs, so under anti-periodic boundaries test the `@` product group
  instead: it keeps `_perm_sign` and contains every product element.
- With `Heisenberg(msr=True)` the state lives in the MSR-rotated basis. Any
  lattice isometry maps the two sublattices uniformly, so at fixed even
  $N_\downarrow$ the MSR sign cancels in the ratio — no correction needed.

## Choosing elements to test

- Test products, not only generators. Magnetically ordered states often break
  each spatial operation and spin inversion separately ($O \approx 0$) yet
  keep their products with spin inversion ($O \sim 0.9$). Build product groups
  with `@`, e.g. `Flip(axis=0) @ SpinInverse()` or
  `Translation((1, 0)) @ SpinInverse()`; this works for spins (Z2 inversion)
  and spinful fermions (mode permutation), and `get_overlap` reports every
  product element.
- Scan reflection/rotation centers before calling one broken. `Flip` and
  `Rotation` default to the lattice mean coordinate: on even $L$ that is
  bond-centred for `Flip` and plaquette-centred for `Rotation`. Order pinned off-center can give $O \approx 0$ about one center
  and $O \sim 0.9$ about another. `Flip(axis) @ Translation(<unit vector along
  axis>)` contains the reflections about every center along that axis;
  otherwise pass `center=` explicitly.
- Report per group element, never per group: with period-2 order, odd
  translation powers give $O \approx 0$ and even powers $O \approx 1$ in the
  same group.
