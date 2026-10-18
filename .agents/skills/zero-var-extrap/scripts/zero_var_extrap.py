"""Zero-variance extrapolation of VMC data via orthogonal distance regression.

Fit y = k * x + b with uncertainties on both x and y, where x is the energy
variance (or its square root for generic observables) and b is the
zero-variance estimate. Uses `odrpack` (https://pypi.org/project/odrpack/);
`scipy.odr` is deprecated and will be removed in SciPy 1.19.0.

CLI usage (training-curve file with columns [step, E, VarE, ...]). The window
selects rows by the value in the step column, start <= step < end:

    python zero_var_extrap.py training_curve.txt --start 1000 --end 10000 --bin 50

Several values for --start/--end/--bin scan the grid of windows and print the
grid mean with its statistical and window (systematic) errors:

    python zero_var_extrap.py training_curve.txt --start 1000 1500 2000 \\
        --end 8000 10000 --bin 20 50 100

For a generic observable stored in another column, extrapolate in sqrt(VarE):

    python zero_var_extrap.py obs.txt --ycol 3 --power 0.5
"""

import itertools
import warnings

import numpy as np
from odrpack import odr_fit

R_MAX = 10.0  # lever arm above which the intercept is weakly constrained


def bin_series(series, bin_size):
    """Bin a 1D series of consecutive (autocorrelated) VMC steps.

    Returns (mean, err) per bin, with err = std / sqrt(bin_size). Increase
    `bin_size` until the fit's res_var stops growing; a naive bin error on
    autocorrelated data underestimates the true uncertainty.
    """
    series = np.asarray(series)
    n = series.size // bin_size * bin_size
    binned = series[:n].reshape(-1, bin_size)
    return binned.mean(axis=1), binned.std(axis=1) / np.sqrt(bin_size)


def zero_var_extrap(x, y, x_err, y_err, beta0=None, maxit=300):
    """ODR fit of y = k * x + b, weighting both x and y errors.

    Returns the `odrpack.OdrResult`; `res.beta = [k, b]` and
    `res.sd_beta = [k_err, b_err]` (already scaled by res_var).

    Check `res.success`: a non-converged fit only warns, so window scans don't
    crash. A non-converged fit, or a converged one whose sd_beta[1] is
    comparable to the extrapolation distance |k| * mean(x), means the window
    does not identify the slope; raising `maxit` makes ODR converge but does
    not fix such a window. Drop non-converged fits. In a window grid, keep the
    converged weak ones and count them (dropping them filters on slope
    significance and understates the error; see `grid_extrap`); if many are
    weak, call the estimate weakly constrained.
    """
    if beta0 is None:
        beta0 = np.polyfit(x, y, 1)  # OLS initial guess

    res = odr_fit(
        lambda x, beta: beta[0] * x + beta[1],
        x,
        y,
        beta0=beta0,
        weight_x=1 / np.asarray(x_err) ** 2,
        weight_y=1 / np.asarray(y_err) ** 2,
        maxit=maxit,
    )
    if not res.success:
        warnings.warn(f"ODR did not converge: {res.stopreason}")
    return res


def select_window(data, start=-np.inf, end=np.inf, stepcol=0):
    """Rows of `data` with start <= step < end.

    stepcol: column of the optimization step; None uses the row index.
    """
    data = np.asarray(data)
    steps = np.arange(len(data)) if stepcol is None else data[:, stepcol]
    return data[(steps >= start) & (steps < end)]


def check_steps(data, stepcol=0):
    """Warn if the step column is not increasing (appended restarts?)."""
    if stepcol is not None and np.any(np.diff(np.asarray(data)[:, stepcol]) <= 0):
        warnings.warn("step column is not increasing (appended restarts?)")


def lever_arm(v, nchunks=4):
    """Lever arm R = median(v) / (VarE drift across the window).

    v: raw per-step VarE**power in the window. The drift is the max - min of
    the medians of `nchunks` consecutive chunks of v (medians resist variance
    spikes). The intercept sits R drift-spans away from the data. Returns nan
    for fewer than `nchunks` steps and inf for a constant v.
    """
    v = np.asarray(v)
    if len(v) < nchunks:
        return np.nan
    medians = [np.median(c) for c in np.array_split(v, nchunks)]
    with np.errstate(divide="ignore"):
        return np.median(v) / np.ptp(medians)


def check_fit(res, x, y, R=None, *, energy):
    """Warnings for a fit whose intercept the data may not identify.

    x, y: fitted points. R: lever arm from `lever_arm`. energy: whether y is the
    energy, which also checks the variational bound, k > 0 and an intercept below
    every fitted energy. Pass energy=False for other observables, whose slope may
    have either sign.
    """
    k, b = res.beta
    msgs = []
    if not res.success:
        msgs.append(f"ODR did not converge ({res.stopreason}): drop this window")
    if res.sd_beta[1] > 0.5 * abs(k) * np.mean(x):
        msgs.append(
            "intercept error comparable to the extrapolation distance "
            "|k| * mean(x): the window does not identify the slope"
        )
    if energy and (k <= 0 or b > np.min(y)):
        msgs.append(
            "slope <= 0 or intercept above the lowest fitted energy: violates the "
            "variational bound, the window does not identify the slope"
        )
    if R is not None and R > R_MAX:
        msgs.append(
            f"lever arm R = {R:.0f}: the slope rests on a small VarE drift, "
            "quote more than the fit and window errors"
        )
    return msgs


def extrap_training_curve(
    data, start, end, bin_size, ycol=1, xcol=2, power=1.0, stepcol=0, maxit=300
):
    """Zero-variance extrapolation on the tail of a training curve.

    data: array with the training curve, one optimization step per row.
    start, end: step window, start <= step < end (end=None: to the end of the
        data). Exclude the fast early descent and restart transients.
    ycol: column of the extrapolated quantity (1 = energy).
    xcol: column of the energy variance.
    power: 1 for the energy, 0.5 for generic observables (x = VarE**power).
    stepcol: column of the optimization step; None uses the row index.

    Returns (res, (x, y, x_err, y_err, step)).
    """
    end = np.inf if end is None else end
    check_steps(data, stepcol)
    d = select_window(data, start, end, stepcol)
    if d.shape[0] // bin_size < 3:
        raise ValueError(
            f"{d.shape[0]} steps in [{start:g}, {end:g}) give fewer than 3 bins of "
            f"{bin_size}; start/end are step values, not row indices"
        )
    y, y_err = bin_series(d[:, ycol], bin_size)
    v, v_err = bin_series(d[:, xcol], bin_size)
    # error propagation through x = v**power
    x = v**power
    x_err = np.abs(power) * v ** (power - 1) * v_err
    steps = np.arange(len(d)) if stepcol is None else d[:, stepcol]
    step, _ = bin_series(steps, bin_size)

    res = zero_var_extrap(x, y, x_err, y_err, maxit=maxit)
    return res, (x, y, x_err, y_err, step)


def grid_extrap(
    data,
    starts,
    ends,
    bins,
    ycol=1,
    xcol=2,
    power=1.0,
    stepcol=0,
    maxit=300,
    energy=None,
    sd_rel_max=5.0,
):
    """Zero-variance extrapolation over a start x end x bin grid of windows.

    Windows with fewer than 4 bins are skipped. A fit is dropped if it did not
    converge, violates the variational bound (energy only), or has
    sd_beta[1] > sd_rel_max * (median sd_beta[1] of the fits passing the first
    two checks). energy: whether column ycol is the energy, which enables the
    variational-bound check; None infers it as ycol == 1 and power == 1, like the
    CLI.

    Returns (b_mean, stat, syst, table): the mean intercept of the kept fits,
    their mean sd_beta[1] (all windows reuse the same steps, so it is not
    divided by sqrt(#windows)) and the std of their intercepts. `table` has
    one row per fitted window: [start, end, bin, k, b, sd_b, res_var, R, D,
    kept], with R the lever arm and D = |k| * mean(x) the extrapolation
    distance.
    """
    if energy is None:
        energy = ycol == 1 and power == 1
    check_steps(data, stepcol)
    rows = []
    for start, end, bin_size in itertools.product(starts, ends, bins):
        end = np.inf if end is None else end
        d = select_window(data, start, end, stepcol)
        if d.shape[0] // bin_size < 4:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # failed fits are dropped and counted
            res, (x, y, *_) = extrap_training_curve(
                data, start, end, bin_size, ycol, xcol, power, stepcol, maxit
            )
        k, b = res.beta
        R = lever_arm(d[:, xcol] ** power)
        ok = res.success and not (energy and (k <= 0 or b > y.min()))
        sd_b, D = res.sd_beta[1], abs(k) * x.mean()
        rows.append([start, end, bin_size, k, b, sd_b, res.res_var, R, D, ok])

    table = np.array(rows, dtype=float).reshape(-1, 10)
    kept = table[:, 9].astype(bool)
    if kept.any():
        kept &= table[:, 5] <= sd_rel_max * np.median(table[kept, 5])
        table[:, 9] = kept
    if kept.sum() < 2:
        warnings.warn(f"only {kept.sum()} usable window(s) out of {len(table)}")
        return np.nan, np.nan, np.nan, table
    b_kept = table[kept, 4]
    return b_kept.mean(), table[kept, 5].mean(), b_kept.std(ddof=1), table


def plot_extrap(res, x, y, x_err, y_err, step, ax=None, label="E0"):
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()
    k, b = res.beta
    k_err, b_err = res.sd_beta

    ax.errorbar(
        x,
        y,
        xerr=x_err,
        yerr=y_err,
        fmt="none",
        ecolor="gray",
        alpha=0.5,
        capsize=2,
        zorder=1,
    )
    sc = ax.scatter(x, y, s=10, c=step, cmap="viridis", zorder=2)
    plt.colorbar(sc, ax=ax, label="step")

    # robust upper limit: occasional variance spikes would blow up the range
    # (spiked bins stay in the fit; ODR downweights them via their large x_err)
    x_max = 1.1 * np.quantile(x, 0.97)
    x_plot = np.linspace(0, x_max, 200)
    ax.plot(x_plot, k * x_plot + b, "k--", label=f"{label}={b:.4f} ± {b_err:.4f}")
    ax.set_xlim(0, x_max)
    ax.legend()
    return ax


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("file", help="training curve txt, one step per row")
    parser.add_argument(
        "--start", type=float, nargs="+", default=[0], help="first step (inclusive)"
    )
    parser.add_argument(
        "--end",
        type=float,
        nargs="+",
        default=[np.inf],
        help="end step (exclusive; default: end of file)",
    )
    parser.add_argument(
        "--bin", type=int, nargs="+", default=[50], help="steps per bin"
    )
    parser.add_argument(
        "--ycol",
        type=int,
        default=1,
        help="column of the extrapolated quantity (1 = energy; the variational-bound "
        "check runs for ycol 1 at power 1 only)",
    )
    parser.add_argument(
        "--xcol", type=int, default=2, help="column of the energy variance"
    )
    parser.add_argument(
        "--stepcol", type=int, default=0, help="column of the step (-1: row index)"
    )
    parser.add_argument(
        "--power", type=float, default=1.0, help="x = VarE**power (0.5 for observables)"
    )
    parser.add_argument("--maxit", type=int, default=300, help="ODR iteration limit")
    parser.add_argument(
        "--plot", default=None, help="output figure path (single window)"
    )
    args = parser.parse_args()

    data = np.loadtxt(args.file)
    stepcol = None if args.stepcol < 0 else args.stepcol
    energy = args.ycol == 1 and args.power == 1

    if len(args.start) * len(args.end) * len(args.bin) > 1:
        if args.plot is not None:
            parser.error("--plot needs a single window")
        b, stat, syst, table = grid_extrap(
            data,
            args.start,
            args.end,
            args.bin,
            args.ycol,
            args.xcol,
            args.power,
            stepcol,
            args.maxit,
            energy,
        )
        print("start end bin: intercept ± sd (slope, res_var, R)")
        for s, e, bs, k, b_w, sd, rv, R, D, kept in table:
            print(
                f"{s:g} {e:g} {bs:g}: {b_w:.6f} ± {sd:.6f} ({k:.4f}, {rv:.2f}, "
                f"{R:.1f}){'' if kept else ' dropped'}"
            )
        kept = table[:, 9].astype(bool)
        print(f"{kept.sum()} of {len(table)} windows kept")
        print(f"intercept = {b:.6f} ± {stat:.6f} (stat) ± {syst:.6f} (window)")
        n_weak = np.sum(kept & (table[:, 5] > 0.5 * table[:, 8]))
        if n_weak:
            print(
                f"WARNING: {n_weak} kept windows have an intercept error above half "
                "the extrapolation distance |k| * mean(x) (slope not identified)"
            )
        R_med = np.median(table[:, 7]) if len(table) else np.nan
        if R_med > R_MAX:
            print(
                f"WARNING: median lever arm R = {R_med:.0f}: the slope rests on a "
                "small VarE drift, quote more than the stat and window errors"
            )
    else:
        start, end, bin_size = args.start[0], args.end[0], args.bin[0]
        res, (x, y, x_err, y_err, step) = extrap_training_curve(
            data,
            start,
            end,
            bin_size,
            args.ycol,
            args.xcol,
            args.power,
            stepcol,
            args.maxit,
        )
        d = select_window(data, start, end, stepcol)
        R = lever_arm(d[:, args.xcol] ** args.power)

        k, b = res.beta
        k_err, b_err = res.sd_beta
        print(f"intercept = {b:.6f} ± {b_err:.6f}")
        print(f"slope = {k:.6f} ± {k_err:.6f}")
        print(
            f"res_var = {res.res_var:.3f} (should be ~1; larger means "
            "underestimated errors or nonlinearity)"
        )
        print(
            f"lever arm R = {R:.1f} (should be a few; intercept distance in "
            "units of the VarE drift)"
        )
        for msg in check_fit(res, x, y, R, energy=energy):
            print(f"WARNING: {msg}")

        if args.plot is not None:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plot_extrap(res, x, y, x_err, y_err, step)
            plt.xlabel("VarE" if args.power == 1 else f"VarE^{args.power}")
            plt.ylabel(f"column {args.ycol}")
            plt.savefig(args.plot, bbox_inches="tight")
            print(f"figure saved to {args.plot}")
