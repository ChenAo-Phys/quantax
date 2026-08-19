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
Z2 inversion).

## Estimator facts (don't mistake these for bugs)

- $O_T = O_{T^{-1}}$ **exactly** by construction, and elements related by an
  architecture-exact symmetry give exactly equal $O_T$ — repeated values in
  the output are consistency, not a copy-paste error.
- The identity element gives $1 \pm 0$; any element enforced exactly by the
  network (CNN translations; sublattice-embedding periodicity, e.g.
  `sublattice=(16,4)` on a 16×8 lattice makes y-translation by 4 exact) gives
  $1.0000 \pm 0.0000$ with strictly zero variance. Both are free sanity checks
  that the permutation/sign machinery is right.
- Error bar: $O_T$ is a product of two means estimated on the **same** samples
  — propagate with the covariance term,
  $\mathrm{Var}(O) = \bar y^2\mathrm{Var}(\bar x) + \bar x^2\mathrm{Var}(\bar y) + 2\bar x\bar y\,\mathrm{Cov}(\bar x,\bar y)$. One sweep of many chains →
  chains independent → naive $/n$ errors valid.
- One sample set serves all symmetries and elements (2 forwards per element
  per sample); sampling dominates the cost, the overlaps themselves are cheap.

## Quantax transformation mechanics

- Transformed config: `spins[:, symm._perm[g]]`; inverse via
  `jnp.argsort(perm, axis=1)`. Characters are ignored — the diagnostic assumes
  the trivial sector.
- Fermion sign: use `quantax.symmetry.symmetry._permutation_sign` (private).
  It is vmapped **over group elements** for a single configuration — vmap it
  over samples yourself: `jax.vmap(_permutation_sign, in_axes=(0, None, None))`.
- `SpinInverse()` is two different objects: for spinful **fermions** it is a
  mode permutation (lives in `_perm`); for **spin** systems it is a Z2
  inversion — `_perm` holds only the identity and the flip lives in
  `symm.Z2_inversion`, so the transformed config is `-s`. A `_perm`-only
  implementation silently measures the identity and reports overlap 1.
- Composite generators (e.g. translate-then-spin-flip for fermions): compose
  permutations as `g1[g2]` and wrap in `Symmetry(g_composed)`.
- With `Heisenberg(msr=True)` the state lives in the MSR-rotated basis. Any
  lattice isometry maps the two sublattices uniformly, so at fixed even
  $N_\downarrow$ the MSR sign cancels in the ratio — no correction needed.

## Reading the numbers (16×8 Hubbard stripe state, what actually happened)

- Every individual symmetry (reflections, odd translations, spin inversion)
  gave $O \approx 0$, yet their **products with spin inversion** gave
  $O \sim 0.88$–$0.93$. Magnetically ordered states typically preserve
  spatial-op × spin-flip combinations while breaking each factor — always test
  products, not just the generators, before concluding nothing can be
  projected.
- Reflection overlap is extremely sensitive to the **center**: $0.002(1)$
  about one axis vs $0.88(2)$ about the bond-centered axis of the same state.
  Scan centers before declaring a reflection broken — the order parameter may
  just be pinned off-center.
- Report **per group element**, never per group: a translation group of a
  period-2 stripe gives $O \approx 0$ for odd powers and $O \approx 1$ for
  even powers in the same group.
