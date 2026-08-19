# D-wave pair-pair correlation

Goal: the raw pair-pair correlation $P(dr) = \left< \Delta^\dagger(r) \Delta(r+dr) \right>$ and its
*connected* part $P_c(dr)$ — the correlation with the Wick/Gaussian-factorizable
background subtracted, analogous to a connected correlation function (cumulant)
but relative to the Wick reference rather than a plain product of means — for
one reference site `r` and all displacements `dr`. The d-wave pair operator is
built from bond singlets

$$
\Delta^\dagger(r) = \frac{1}{4} \sum_\delta f(\delta)\, B^\dagger(r, r+\delta), \quad
f(\pm x) = +1,\ f(\pm y) = -1,
$$

$$
B^\dagger(a, b) = \frac{1}{\sqrt{2}} \left(c^\dagger_{a\uparrow} c^\dagger_{b\downarrow}
- c^\dagger_{a\downarrow} c^\dagger_{b\uparrow}\right), \quad B^\dagger(a,b) = B^\dagger(b, a).
$$

Fully expanded in $c^\dagger c^\dagger c c$ strings, $P$ carries the global prefactor $1/32$.
The $1/4$ normalizes $P$ to the standard convention of the Hubbard-model
literature: at long distance $P$ saturates to the correlator
$\langle B^\dagger B \rangle$ of a single bond-singlet pair, so
$\Delta_\mathrm{SC}$ (below) is directly comparable to the AFQMC/DMRG
benchmarks (Qin et al., PRX 10, 031016 (2020), Simons Collaboration;
Xu et al., Science 384, eadh7691 (2024), AFQMC). Beware that other
conventions coexist: mVMC papers (e.g. Ido et al., PRB 97, 045138 (2018))
omit the $1/4$, giving $4\times$ this $\Delta_\mathrm{SC}$.

**Don't** build $\Delta^\dagger(r) \Delta(r+dr)$ as one `Operator` per `dr`: expanding all `dr`
gives $16N$ products of bond singlets, but only $4 \times 2N$ are distinct (each
lattice bond is reached from both endpoints), so every underlying
$c^\dagger c^\dagger c c$ term is measured twice — and the naive route yields no
hopping terms to subtract. Instead, the calculation splits into two stages:

1. **Define unique terms.** Pair terms $F(b_L, b_R) = \left< B^\dagger_{b_L} B_{b_R} \right>$,
   one per pair of the 4 bonds $b_L$ through $r$ and the $2N$ lattice bonds
   $b_R$ — each a single `Operator` of four $c^\dagger c^\dagger c c$ strings.
   Key each bond by (lower endpoint, axis) so $B^\dagger(a,b) = B^\dagger(b,a)$
   dedupes it automatically. Also define hopping terms
   $G_\sigma(i,j) = \left< c^\dagger_{i\sigma} c_{j\sigma} \right>$ between the
   5 sites touching $b_L$ and every site $j$. With APBC, additional boundary
   signs apply. Measured on GPU by [dwave_measure.py](dwave_measure.py) (see
   its docstring for the JAX/host-sync details).
2. **Assemble.**
   $$
   P(dr) = \frac{1}{16} \sum_{\delta, \delta'} f(\delta) f(\delta')\, F\bigl(\mathrm{bond}(r,\delta), \mathrm{bond}(r+dr,\delta')\bigr)
   $$
   (the $1/16$ from the two $\Delta$ normalizations; the $1/\sqrt{2}$'s live in the $F$ terms).
   Subtracting the non-pairing background — Wick-decoupling
   $\left< B^\dagger(a,b)\, B(c,d) \right>$ into hopping terms gives (Sz-conserving state)
   $$
   F_\mathrm{wick} = \frac{1}{2}\Bigl(G_\uparrow(a,c) G_\downarrow(b,d) + G_\uparrow(a,d) G_\downarrow(b,c)
   + G_\downarrow(a,d) G_\uparrow(b,c) + G_\downarrow(a,c) G_\uparrow(b,d)\Bigr)
   $$
   — gives the connected correlation $P_c(dr) = \frac{1}{16} \sum f f' (F - F_\mathrm{wick})$, which
   reflects genuine pairing rather than uncorrelated hopping. It is only
   meaningful when the two pairs don't share sites ($|dr|$ beyond ~2 lattice
   spacings). Error bars are jackknifed over sweeps, since $P_c$ is nonlinear
   in the measured terms. Assembled from the saved terms on CPU by
   [dwave_assembly.py](dwave_assembly.py) (see its docstring for the
   jackknife and the leave-one-block-out caveat under autocorrelation).

## Overall order parameter

A single number summarizing long-range pairing is the d-wave order parameter

$$
\Delta_\mathrm{SC}^2 = \frac{1}{\mathcal{M}} \sum_{|dr| \geq r_\mathrm{max}} P_c(dr),
$$

the connected correlation averaged over its saturation region: all
displacement vectors (not distance bins) with minimum-image distance
$|dr| \geq r_\mathrm{max}$, with $r_\mathrm{max} = d_\mathrm{max}/2$ where
$d_\mathrm{max}$ is the largest distance between two sites of the cluster
(for an $L \times L$ torus, $d_\mathrm{max} = L/\sqrt{2}$), and $\mathcal{M}$
the number of such vectors. The quantity usually plotted (e.g. for
finite-size extrapolation in $1/L$) is $\Delta_\mathrm{SC}$, the square root.
Computed at the end of [dwave_assembly.py](dwave_assembly.py), with errors
taken over the same jackknife replicates as $P_c$ (never propagated from the
per-$dr$ error bars, which are strongly correlated); the square root is
skipped when replicates of $\Delta_\mathrm{SC}^2$ dip $\leq 0$, i.e. when the
result is consistent with no long-range pairing.
