from __future__ import annotations
from typing import Sequence, Union
from numbers import Number
from . import Operator, sigma_x, sigma_p, sigma_m, sigma_z
from ..global_defs import get_sites


def Heisenberg(
    J: Union[Number, Sequence[Number]] = 1.0,
    n_neighbor: Union[int, Sequence[int]] = 1,
    msr: bool = False,
) -> Operator:
    sites = get_sites()
    if isinstance(J, Number):
        J = [J]
    if isinstance(n_neighbor, Number):
        n_neighbor = [n_neighbor]
    if len(J) != len(n_neighbor):
        raise ValueError("The 'J' and 'n_neighbor' should have the same length.")
    neighbors = sites.get_neighbor(n_neighbor)

    def hij(i, j, sign):
        hx = 2 * sign * (sigma_p(i) * sigma_m(j) + sigma_m(i) * sigma_p(j))
        hz = sigma_z(i) * sigma_z(j)
        return hx + hz

    H = 0
    for idx, neighbors_i in enumerate(neighbors):
        sign = -1 if msr and n_neighbor[idx] == 1 else 1
        H = H + J[idx] * sum(hij(i, j, sign) for i, j in neighbors_i)
    return H


def Ising(
    h: Number = 0.0,
    J: Number = 1.0,
) -> Operator:
    sites = get_sites()
    H = -h * sum(sigma_x(i) for i in range(sites.nsites))
    neighbors = sites.get_neighbor()
    H += -J * sum(sigma_z(i) * sigma_z(j) for i, j in neighbors)
    return H
