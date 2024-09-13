from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from numbers import Number


class DataTracer:
    def __init__(self):
        self._data_list = []
        self._data_array = np.array([])
        self._time_list = []
        self._time_array = np.array([])
        self._ax = None

    @property
    def data(self) -> np.ndarray:
        return self._data_array

    @property
    def time(self) -> np.ndarray:
        return self._time_array

    @property
    def ax(self) -> Optional[plt.Axes]:
        return self._ax

    def append(self, data: Number, time: Optional[float] = None):
        self._data_list.append(data)
        self._data_array = np.asarray(self._data_list)

        if time is None:
            if len(self._time_list):
                self._time_list.append(self._time_list[-1] + 1)
            else:
                self._time_list.append(0)
        else:
            self._time_list.append(time)
        self._time_array = np.asarray(self._time_list)

    def __getitem__(self, idx):
        return self._data_array[idx]

    def __array__(self):
        return self.data

    def __repr__(self):
        return self.data.__repr__()

    def mean(self) -> Number:
        return np.mean(self.data)

    def uncertainty(self) -> Optional[Number]:
        n = self.data.size
        if n < 2:
            return None
        mean = self.mean()
        diff = np.abs(self.data - mean) ** 2
        return np.sqrt(np.sum(diff) / n / (n - 1))

    def save(self, file: str):
        np.save(file, self._data_array)

    def save_time(self, file: str):
        np.save(file, self._time_array)

    def plot(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        batch: int = 1,
        logx: bool = False,
        logy: bool = False,
        baseline: Optional[Number] = None,
    ) -> None:
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
                data = (data - baseline) / abs(baseline)
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
