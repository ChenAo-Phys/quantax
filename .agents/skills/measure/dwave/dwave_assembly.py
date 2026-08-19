"""D-wave pair-pair correlation: CPU assembly stage.

Loads the raw per-sweep F/G term series saved by dwave_measure.py and
computes the raw P(dr) and connected P_c(dr) with jackknife error bars,
entirely with numpy on the CPU. Also reports the overall order parameter
Δ_SC — the average of P_c over the saturation region |dr| >= r_max (see
dwave_pairing.md) — with its error taken over the same jackknife replicates.

Why a separate script: the jackknife needs one full re-assembly per sweep —
a python loop combining the same ~200 saved numbers NSWEEPS times. That's
negligible CPU work, but running it inside the measurement job would hold a
GPU allocation idle for it. Copy the small dwave_terms.npz off the cluster
(or run on a login node) and do this step there instead.

`canon_bond`/`bond_sites`/DIRS/FF/SITES are duplicated from dwave_measure.py
— pure Python, no JAX dependency — and must produce the identical bond keys,
since those keys are how F/G values loaded from the npz get mapped back onto
sites and displacements. If you change the bond convention in one script,
change it in the other.

The jackknife (`jk_err`) computes leave-one-sweep-out estimates of every F/G
term, re-runs `assemble` on each, and takes
SE = sqrt(n-1) * std(theta_1, ..., theta_n) over the resulting replicates —
this correctly propagates uncertainty through the nonlinear (bilinear-in-G)
Wick subtraction without needing a hand-derived error-propagation formula.

Caveat: this treats sweeps as independent draws. If they're autocorrelated
(see the binning analysis in SKILL.md), the jackknife error is an
underestimate — for a production run, check by replacing the
leave-one-sweep-out jackknife with a leave-one-block-out version (group
consecutive sweeps into blocks before jackknifing) and confirm the error has
plateaued as block size grows.
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


F_mc = dict(zip(F_keys, F_series.mean(axis=0)))
G_mc = dict(zip(G_keys, G_series.mean(axis=0)))
P_mc, Pc_mc = assemble(F_mc, G_mc)

# jackknife over sweeps (handles the nonlinear Wick products in P_c)
F_sum, G_sum = F_series.sum(axis=0), G_series.sum(axis=0)
P_jk, Pc_jk = [], []
for n in range(NSWEEPS):
    F_n = dict(zip(F_keys, (F_sum - F_series[n]) / (NSWEEPS - 1)))
    G_n = dict(zip(G_keys, (G_sum - G_series[n]) / (NSWEEPS - 1)))
    P_n, Pc_n = assemble(F_n, G_n)
    P_jk.append(P_n)
    Pc_jk.append(Pc_n)


def jk_err(samples_by_dr, dr):
    vals = np.array([s[dr] for s in samples_by_dr])
    return np.sqrt(NSWEEPS - 1) * vals.std()


print(f"\n{'dr':>8} {'P':>20} {'P_c':>20}")
for dr in SITES:
    err, cerr = jk_err(P_jk, dr), jk_err(Pc_jk, dr)
    print(f"{str(dr):>8} {P_mc[dr]:>12.5f} ± {err:.5f} {Pc_mc[dr]:>12.5f} ± {cerr:.5f}")


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


d2_mc = delta2(Pc_mc)
d2_jk = np.array([delta2(Pc_n) for Pc_n in Pc_jk])
d2_err = np.sqrt(NSWEEPS - 1) * d2_jk.std()
print(
    f"\nΔ²_SC = {d2_mc:.6f} ± {d2_err:.6f}"
    f"  ({len(FAR)} vectors with |dr| >= {d_max / 2:.3f})"
)
if d2_jk.min() > 0:
    dsc_jk = np.sqrt(d2_jk)
    dsc_err = np.sqrt(NSWEEPS - 1) * dsc_jk.std()
    print(f"Δ_SC  = {np.sqrt(d2_mc):.6f} ± {dsc_err:.6f}")
else:
    print(
        "Δ_SC not reported: some jackknife replicates of Δ²_SC are <= 0, "
        "i.e. consistent with no long-range pairing — quote Δ²_SC instead."
    )
