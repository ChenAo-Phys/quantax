from os import PathLike
from typing import Any, List
from typing import BinaryIO
from numpy.typing import NDArray, ArrayLike
import jax
import numpy as np


class DataTracer:
    """
    Keeps track of a scalar time series, typically an observable (e.g. energy)
    recorded once per optimization or time-evolution step.

    The data points are stored in a growing list and exposed as numpy arrays
    through :attr:`data` and :attr:`time`.
    """

    def __init__(self):
        self._data: List[Any] = []
        self._time: List[Any] = []

    @property
    def data(self) -> NDArray[np.floating]:
        """The data stored in the DataTracer"""
        return np.asarray(self._data)

    @property
    def time(self) -> NDArray[np.floating]:
        """The time stored in the DataTracer"""
        return np.asarray(self._time)

    def append(self, data: ArrayLike | None, time: ArrayLike | None = None) -> None:
        """
        Append a new data point.

        :param data:
            The data to be appended, expected to be a scalar. ``None`` is
            ignored, so optional quantities can be appended unconditionally.

        :param time:
            The time of the data point, default to be incremental by 1 in each
            append.
        """
        if data is None:
            return

        if time is None:
            time = 0 if len(self._time) == 0 else self._time[-1] + 1

        self._data.append(data)
        self._time.append(time)

    def __getitem__(self, idx) -> NDArray[np.floating]:
        """Get data by indexing."""
        return self.data[idx]

    def __array__(self) -> NDArray[np.floating]:
        """Return data as a numpy array."""
        return self.data

    def __repr__(self) -> str:
        return self.data.__repr__()

    def mean(self) -> np.floating:
        """Mean value of the data"""
        return np.mean(self.data)

    def uncertainty(self) -> np.floating | None:
        """
        Standard error of the mean, ``None`` if fewer than 2 data points are
        stored.
        """
        data = self.data
        n = data.size
        if n < 2:
            return None
        diff = np.abs(data - data.mean()) ** 2
        return np.sqrt(np.sum(diff) / n / (n - 1))

    def save(self, file: str | PathLike[str] | BinaryIO) -> None:
        """Save data to file"""
        if jax.process_index() == 0:
            np.save(file, self.data)

    def save_time(self, file: str | PathLike[str] | BinaryIO) -> None:
        """Save time to file"""
        if jax.process_index() == 0:
            np.save(file, self.time)

    def plot(
        self,
        start: int | None = None,
        end: int | None = None,
        batch: int = 1,
        logx: bool = False,
        logy: bool = False,
        baseline: ArrayLike | None = None,
    ) -> None:
        """
        Plot the data

        :param start:
            Starting index

        :param end:
            Ending index

        :param batch:
            Batch size. The mean value in a whole batch is one data point in the plot

        :param logx:
            Whether to use log scale in x-axis

        :param logy:
            Whether to use log scale in y-axis

        :param baseline:
            Show a dashed line y=baseline
        """
        import matplotlib.pyplot as plt

        time = self.time
        data = self.data
        if start is None:
            start = 0
        if end is None:
            end = data.size
        time = time[start:end]
        data = data[start:end]

        num_redundant = data.size % batch
        if num_redundant > 0:
            time = time[:-num_redundant]
            data = data[:-num_redundant]
        if data.size == 0:
            return

        time = np.mean(time.reshape(-1, batch), axis=1)
        data = np.mean(data.reshape(-1, batch), axis=1)
        if baseline is not None:
            baseline = np.asarray(baseline)
            if logy:
                data = (data - baseline) / np.abs(baseline)
            else:
                plt.hlines(
                    baseline,
                    xmin=time[0],
                    xmax=time[-1],
                    colors="k",
                    linestyles="dashed",
                )

        if logx and not logy:
            plt.semilogx(time, data)
        elif logy and not logx:
            plt.semilogy(time, data)
        elif logx and logy:
            plt.loglog(time, data)
        else:
            plt.plot(time, data)
