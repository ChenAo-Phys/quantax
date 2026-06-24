import numpy as np
from quantax.utils import DataTracer


def test_append_and_data():
    t = DataTracer()
    for v in (1.0, 2.0, 3.0):
        t.append(v)
    assert np.array_equal(t.data, [1.0, 2.0, 3.0])
    assert np.array_equal(t.time, [0, 1, 2])  # time auto-increments by 1


def test_append_none_is_ignored():
    t = DataTracer()
    t.append(1.0)
    t.append(None)  # ignored: neither data nor time advance
    t.append(2.0)
    assert np.array_equal(t.data, [1.0, 2.0])
    assert np.array_equal(t.time, [0, 1])


def test_explicit_time():
    t = DataTracer()
    t.append(1.0, time=0.5)
    t.append(2.0, time=1.5)
    assert np.array_equal(t.time, [0.5, 1.5])


def test_mean_indexing_and_array():
    t = DataTracer()
    for v in (2.0, 4.0, 6.0):
        t.append(v)
    assert t.mean() == 4.0
    assert t[1] == 4.0
    assert np.array_equal(np.asarray(t), [2.0, 4.0, 6.0])


def test_uncertainty_none_below_two_points():
    t = DataTracer()
    assert t.uncertainty() is None
    t.append(1.0)
    assert t.uncertainty() is None  # still < 2 points


def test_uncertainty_is_standard_error_of_mean():
    t = DataTracer()
    for v in (1.0, 2.0, 3.0):
        t.append(v)
    # SEM = sqrt(sum((x-mean)^2) / n / (n-1)) = sqrt(2 / 3 / 2) = sqrt(1/3)
    np.testing.assert_allclose(t.uncertainty(), np.sqrt(1 / 3))
