"""D-wave pair-pair correlation: CPU assembly stage.

Loads the raw per-sweep F/G term series saved by dwave_measure.py and
computes the raw P(dr) and connected P_c(dr) with jackknife error bars,
entirely with numpy on the CPU. Also reports the overall order parameter
Δ_SC — the average of P_c over the saturation region |dr| >= r_max (see
dwave_pairing.md) — with its error taken over the same jackknife replicates.

Why a separate script: the jackknife needs one full re-assembly per block —
a python loop combining the same ~200 saved numbers many times. That's
negligible CPU work, but running it inside the measurement job would hold a
GPU allocation idle for it. Copy dwave_terms.npz to your workstation after the
job has finished writing it, and run the assembly there. Don't run analysis on
shared login nodes, and never cp/stat/read a running job's output files on a
parallel filesystem: a hung read can hold a lock and stall the job's next save.

`canon_bond`/`bond_sites`/DIRS/FF/SITES are duplicated from dwave_measure.py
— pure Python, no JAX dependency — and must produce the identical bond keys,
since those keys are how F/G values loaded from the npz get mapped back onto
sites and displacements. If you change the bond convention in one script,
change it in the other.

The jackknife (`block_jackknife`) averages the F/G series over blocks of b
consecutive sweeps, computes leave-one-block-out estimates of every term,
re-runs `assemble` on each, and takes SE = sqrt(nb-1) * std(theta_1, ...,
theta_nb) over the nb replicates — this correctly propagates uncertainty
through the nonlinear (bilinear-in-G) Wick subtraction without needing a
hand-derived error-propagation formula. Consecutive sweeps are autocorrelated
(see the binning analysis in SKILL.md), so b = 1 (leave-one-sweep-out)
underestimates the error by ~sqrt(1 + 2 tau). The script scans b = 1, 2, 4, ...
while at least 8 blocks remain, prints the Δ²_SC error vs b, and quotes all
errors at the largest b; check that the error has plateaued there, else run
more sweeps.
"""

import numpy as np

data = np.load("dwave_terms.npz", allow_pickle=True)
F_series, F_keys = data["F"], [tuple(k) for k in data["F_keys"]]
G_series, G_keys = data["G"], [tuple(k) for k in data["G_keys"]]
LX, LY = int(data["LX"]), int(data["LY"])
R = tuple(int(x) for x in data["R"])
NSWEEPS = F_series.shape[0]

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
FF = {(1, 0): 1.0, (-1, 0): 1.0, (0, 1): -1.0, (0, -1): -1.0}  # d-wave form factor
SITES = [(x, y) for x in range(LX) for y in range(LY)]


def wrap(site):
    return (site[0] % LX, site[1] % LY)


def canon_bond(site, d):
    """Must match the identical helper in dwave_measure.py."""
    if d[0] + d[1] < 0:
        site = (site[0] + d[0], site[1] + d[1])
    x, y = wrap(site)
    return (x, y, 0 if d[0] != 0 else 1)


def bond_sites(bond):
    x, y, axis = bond
    return (x, y), wrap((x + 1, y) if axis == 0 else (x, y + 1))


def assemble(F, G):
    """Raw P(dr) and connected P_c(dr) for all dr from term values F and G."""
    P, P_c = {}, {}
    for dr in SITES:
        raw = conn = 0.0
        for dl in DIRS:
            bl = canon_bond(R, dl)
            a, b = bond_sites(bl)
            for d2 in DIRS:
                br = canon_bond(wrap((R[0] + dr[0], R[1] + dr[1])), d2)
                c, d = bond_sites(br)
                wick = 0.5 * (
                    G[a, c, "u"] * G[b, d, "d"]
                    + G[a, d, "u"] * G[b, c, "d"]
                    + G[a, d, "d"] * G[b, c, "u"]
                    + G[a, c, "d"] * G[b, d, "u"]
                )
                f = FF[dl] * FF[d2]
                raw += f * F[bl, br]
                conn += f * (F[bl, br] - wick)
        # 1/16 from the 1/4 normalization of each Δ (the 1/√2's live in B†);
        # normalizes P to the per-bond-singlet convention of the AFQMC/DMRG
        # benchmarks (Qin et al., PRX 2020; Xu et al., Science 2024)
        P[dr], P_c[dr] = raw / 16, conn / 16
    return P, P_c


# overall d-wave order parameter: Δ²_SC = mean of P_c over the saturation
# region |dr| >= r_max = d_max/2 (minimum-image distances, see dwave_pairing.md)
def min_image_dist(dr):
    dx = min(dr[0] % LX, LX - dr[0] % LX)
    dy = min(dr[1] % LY, LY - dr[1] % LY)
    return np.hypot(dx, dy)


d_max = max(min_image_dist(dr) for dr in SITES)
FAR = [dr for dr in SITES if min_image_dist(dr) >= d_max / 2]


def delta2(Pc):
    return sum(Pc[dr] for dr in FAR) / len(FAR)


def block_jackknife(b):
    """Leave-one-block-out replicates of (P, P_c) for blocks of b sweeps.

    Returns the number of blocks nb and the replicates; the jackknife error of
    any function of (P, P_c) is sqrt(nb - 1) * std over the replicates."""
    nb = NSWEEPS // b
    F_blk = F_series[: nb * b].reshape(nb, b, -1).mean(axis=1)
    G_blk = G_series[: nb * b].reshape(nb, b, -1).mean(axis=1)
    F_sum, G_sum = F_blk.sum(axis=0), G_blk.sum(axis=0)
    replicates = []
    for k in range(nb):
        F_k = dict(zip(F_keys, (F_sum - F_blk[k]) / (nb - 1)))
        G_k = dict(zip(G_keys, (G_sum - G_blk[k]) / (nb - 1)))
        replicates.append(assemble(F_k, G_k))
    return nb, replicates


def jk_err(nb, values):
    return np.sqrt(nb - 1) * np.std(values)


F_mc = dict(zip(F_keys, F_series.mean(axis=0)))
G_mc = dict(zip(G_keys, G_series.mean(axis=0)))
P_mc, Pc_mc = assemble(F_mc, G_mc)
d2_mc = delta2(Pc_mc)

# block-size scan of the jackknife error (handles the nonlinear Wick products)
MIN_BLOCKS = 8  # block sizes beyond b = 1 must leave at least this many blocks
if NSWEEPS < 2:
    raise ValueError("The jackknife needs NSWEEPS >= 2.")
print("\nΔ²_SC jackknife error vs block size:")
b, d2_errs = 1, []
while b == 1 or NSWEEPS // b >= MIN_BLOCKS:
    nb, replicates = block_jackknife(b)
    d2_errs.append(jk_err(nb, [delta2(Pc) for _, Pc in replicates]))
    print(f"  block {b:4d} ({nb:4d} blocks): {d2_errs[-1]:.6f}")
    b *= 2
if len(d2_errs) < 2:
    print("WARNING: one block size only, plateau not checked; run more sweeps")
elif d2_errs[-1] > 1.2 * d2_errs[-2]:
    print("WARNING: jackknife error not plateaued; run more sweeps")
P_jk = [P for P, _ in replicates]  # replicates at the largest block size
Pc_jk = [Pc for _, Pc in replicates]
b //= 2
dropped = NSWEEPS - nb * b  # trailing sweeps left out of the blocks (means use all)
note = f", last {dropped} sweep(s) not in the error estimate" if dropped else ""
print(f"errors below: leave-one-block-out, block {b}, {nb} blocks{note}")

print(f"\n{'dr':>8} {'P':>20} {'P_c':>20}")
for dr in SITES:
    err = jk_err(nb, [P[dr] for P in P_jk])
    cerr = jk_err(nb, [Pc[dr] for Pc in Pc_jk])
    print(f"{str(dr):>8} {P_mc[dr]:>12.5f} ± {err:.5f} {Pc_mc[dr]:>12.5f} ± {cerr:.5f}")

d2_jk = np.array([delta2(Pc) for Pc in Pc_jk])
print(
    f"\nΔ²_SC = {d2_mc:.6f} ± {jk_err(nb, d2_jk):.6f}"
    f"  ({len(FAR)} vectors with |dr| >= {d_max / 2:.3f})"
)
if d2_jk.min() > 0:
    print(f"Δ_SC  = {np.sqrt(d2_mc):.6f} ± {jk_err(nb, np.sqrt(d2_jk)):.6f}")
else:
    print(
        "Δ_SC not reported: some jackknife replicates of Δ²_SC are <= 0, "
        "i.e. consistent with no long-range pairing — quote Δ²_SC instead."
    )
