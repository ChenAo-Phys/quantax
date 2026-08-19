"""Zero-variance extrapolation of VMC data via orthogonal distance regression.

Fit y = k * x + b with uncertainties on both x and y, where x is the energy
variance (or its square root for generic observables) and b is the
zero-variance estimate. Uses `odrpack` (https://pypi.org/project/odrpack/);
`scipy.odr` is deprecated and will be removed in SciPy 1.19.0.

CLI usage (training-curve file with columns [step, E, VarE, ...]):

    python zero_var_extrap.py training_curve.txt --start 1000 --end 10000 --bin 50

For a generic observable stored in another column, extrapolate in sqrt(VarE):

    python zero_var_extrap.py obs.txt --ycol 3 --power 0.5
"""

import numpy as np
from odrpack import odr_fit


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


def zero_var_extrap(x, y, x_err, y_err, beta0=None):
    """ODR fit of y = k * x + b, weighting both x and y errors.

    Returns the `odrpack.OdrResult`; `res.beta = [k, b]` and
    `res.sd_beta = [k_err, b_err]` (already scaled by res_var).
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
    )
    if not res.success:
        raise RuntimeError(f"ODR failed: {res.stopreason}")
    return res


def extrap_training_curve(data, start, end, bin_size, ycol=1, xcol=2, power=1.0):
    """Zero-variance extrapolation on the tail of a training curve.

    data: array with the training curve, one optimization step per row.
    start, end: step window; must begin after the energy has converged.
    ycol: column of the extrapolated quantity (1 = energy).
    xcol: column of the energy variance.
    power: 1 for the energy, 0.5 for generic observables (x = VarE**power).

    Returns (res, (x, y, x_err, y_err, step)).
    """
    y, y_err = bin_series(data[start:end, ycol], bin_size)
    v, v_err = bin_series(data[start:end, xcol], bin_size)
    # error propagation through x = v**power
    x = v**power
    x_err = np.abs(power) * v ** (power - 1) * v_err
    step, _ = bin_series(np.arange(start, end), bin_size)

    res = zero_var_extrap(x, y, x_err, y_err)
    return res, (x, y, x_err, y_err, step)


def plot_extrap(res, x, y, x_err, y_err, step, ax=None, label="E0"):
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()
    k, b = res.beta
    k_err, b_err = res.sd_beta

    ax.errorbar(
        x, y, xerr=x_err, yerr=y_err, fmt="none",
        ecolor="gray", alpha=0.5, capsize=2, zorder=1,
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

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="training curve txt, one step per row")
    parser.add_argument("--start", type=int, default=0, help="first step of the fit window")
    parser.add_argument("--end", type=int, default=None, help="last step (default: end of file)")
    parser.add_argument("--bin", type=int, default=50, help="steps per bin")
    parser.add_argument("--ycol", type=int, default=1, help="column of the extrapolated quantity")
    parser.add_argument("--xcol", type=int, default=2, help="column of the energy variance")
    parser.add_argument("--power", type=float, default=1.0, help="x = VarE**power (0.5 for observables)")
    parser.add_argument("--plot", default=None, help="output figure path")
    args = parser.parse_args()

    data = np.loadtxt(args.file)
    end = args.end if args.end is not None else data.shape[0]
    res, (x, y, x_err, y_err, step) = extrap_training_curve(
        data, args.start, end, args.bin, args.ycol, args.xcol, args.power
    )

    k, b = res.beta
    k_err, b_err = res.sd_beta
    print(f"intercept = {b:.6f} ± {b_err:.6f}")
    print(f"slope = {k:.6f} ± {k_err:.6f}")
    print(f"res_var = {res.res_var:.3f} (should be ~1; larger means "
          "underestimated errors or nonlinearity)")

    if args.plot is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_extrap(res, x, y, x_err, y_err, step)
        plt.xlabel("VarE" if args.power == 1 else f"VarE^{args.power}")
        plt.ylabel(f"column {args.ycol}")
        plt.savefig(args.plot, bbox_inches="tight")
        print(f"figure saved to {args.plot}")
