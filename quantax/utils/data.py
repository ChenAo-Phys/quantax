from os import PathLike
from typing import BinaryIO
from numpy.typing import NDArray, ArrayLike
import jax
import numpy as np


class DataTracer:
    """
    The structure used to keep track of the data updates
    """

    def __init__(self):
        self._data_array = np.array([])
        self._time_array = np.array([])
        self._ax = None

    @property
    def data(self) -> NDArray[np.floating]:
        """The data stored in the DataTracer"""
        return self._data_array

    @property
    def time(self) -> NDArray[np.floating]:
        """The time stored in the DataTracer"""
        return self._time_array

    def append(self, data: ArrayLike | None, time: ArrayLike | None = None):
        """
        Append new data

        :param data:
            The data to be appended

        :param time:
            The time of the data, default to be incremental by 1 in each append
        """
        if data is None:
            return

        self._data_array = np.append(self._data_array, data)

        if time is None:
            t = 0 if self._time_array.size == 0 else self._time_array[-1] + 1
        else:
            t = time
        self._time_array = np.append(self._time_array, t)

    def __getitem__(self, idx):
        """Get data by indexing."""
        return self._data_array[idx]

    def __array__(self):
        """Return data as a numpy array."""
        return self.data

    def __repr__(self):
        return self.data.__repr__()

    def mean(self) -> np.floating:
        """Mean value of the data"""
        return np.mean(self.data)

    def uncertainty(self) -> np.floating | None:
        """Uncertainty of the data"""
        n = self.data.size
        if n < 2:
            return None
        mean = self.mean()
        diff = np.abs(self.data - mean) ** 2
        return np.sqrt(np.sum(diff) / n / (n - 1))

    def save(self, file: str | PathLike[str] | BinaryIO) -> None:
        """Save data to file"""
        if jax.process_index() == 0:
            np.save(file, self._data_array)

    def save_time(self, file: str | PathLike[str] | BinaryIO) -> None:
        """Save time to file"""
        if jax.process_index() == 0:
            np.save(file, self._time_array)

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

        time = self._time_array
        data = self._data_array
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
