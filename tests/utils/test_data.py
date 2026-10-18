import sys
import types
import numpy as np
from quantax.utils import DataTracer


def _tracer(values, **kwargs):
    t = DataTracer()
    for v in values:
        t.append(v, **kwargs)
    return t


def test_append():
    t = _tracer([1.0, None, 2.0, 3.0])  # None is ignored: neither data nor time advance
    np.testing.assert_array_equal(t.data, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(np.asarray(t), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(t.time, [0, 1, 2])  # time auto-increments by 1
    t.append(4.0, time=5.5)
    t.append(5.0)  # increments from the last explicit time
    np.testing.assert_array_equal(t.time, [0, 1, 2, 5.5, 6.5])
    assert t[-1] == 5.0


def test_mean_and_uncertainty():
    t = _tracer([1.0, 2.0, 3.0])
    assert t.mean() == 2.0
    # SEM = sqrt(sum((x-mean)^2) / n / (n-1)) = sqrt(2 / 3 / 2) = sqrt(1/3)
    np.testing.assert_allclose(t.uncertainty(), np.sqrt(1 / 3))
    assert DataTracer().uncertainty() is None
    assert _tracer([1.0]).uncertainty() is None  # still < 2 points


def _fake_pyplot(monkeypatch):
    # records the plotted curves without importing matplotlib
    calls = []
    plt = types.ModuleType("matplotlib.pyplot")
    for name in ("plot", "semilogx", "semilogy", "loglog", "hlines"):
        setattr(plt, name, lambda *a, name=name, **kw: calls.append((name, a, kw)))
    mpl = types.ModuleType("matplotlib")
    mpl.pyplot = plt
    monkeypatch.setitem(sys.modules, "matplotlib", mpl)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", plt)
    return calls


def test_plot_batches_and_baseline(monkeypatch):
    calls = _fake_pyplot(monkeypatch)
    t = _tracer(np.arange(1.0, 9.0))  # data 1..8 at time 0..7

    # [start, end) = data 2..8, batches of 3 averaged, the incomplete last one dropped
    t.plot(start=1, batch=3, baseline=1.0)
    (hline, hargs, hkw), (name, (time, data), _) = calls
    assert (
        hline == "hlines" and hargs == (1.0,) and (hkw["xmin"], hkw["xmax"]) == (2, 5)
    )
    assert name == "plot"
    np.testing.assert_allclose(time, [2.0, 5.0])
    np.testing.assert_allclose(data, [3.0, 6.0])

    # with logy, the baseline turns the data into the relative error
    calls.clear()
    t.plot(end=4, batch=2, logy=True, baseline=-2.0)
    ((name, (time, data), _),) = calls
    assert name == "semilogy"
    np.testing.assert_allclose(time, [0.5, 2.5])
    np.testing.assert_allclose(data, [(1.5 + 2) / 2, (3.5 + 2) / 2])

    calls.clear()
    t.plot(logx=True)
    t.plot(logx=True, logy=True)
    t.plot(start=7, batch=2)  # less than one batch: nothing to plot
    assert [c[0] for c in calls] == ["semilogx", "loglog"]
