from __future__ import annotations
from typing import Sequence, Union
from numbers import Number
from . import (
    Operator,
    sigma_x,
    sigma_p,
    sigma_m,
    sigma_z,
    create_u,
    create_d,
    annihilate_u,
    annihilate_d,
    number_u,
    number_d,
)
from ..global_defs import get_sites


def Heisenberg(
    J: Union[Number, Sequence[Number]] = 1.0,
    n_neighbor: Union[int, Sequence[int]] = 1,
    msr: bool = False,
) -> Operator:
    """
    Heisenberg Hamiltonian
    :math:`H = J_n \sum_{<ij>_n} \mathbf{\sigma}_i \cdot \mathbf{\sigma}_j`
    """
    sites = get_sites()
    if isinstance(J, Number):
        J = [J]
    if isinstance(n_neighbor, Number):
        n_neighbor = [n_neighbor]
    if len(J) != len(n_neighbor):
        raise ValueError("'J' and 'n_neighbor' should have the same length.")
    neighbors = sites.get_neighbor(n_neighbor)

    def hij(i, j, sign):
        hx = 2 * sign * (sigma_p(i) * sigma_m(j) + sigma_m(i) * sigma_p(j))
        hz = sigma_z(i) * sigma_z(j)
        return hx + hz

    H = 0
    for idx, neighbors_i in enumerate(neighbors):
        sign = -1 if msr and n_neighbor[idx] == 1 else 1
        H = H + J[idx] * sum(hij(i.item(), j.item(), sign) for i, j in neighbors_i)
    return H


def Ising(
    h: Number = 0.0,
    J: Number = 1.0,
) -> Operator:
    """
    Transverse-field Ising Hamiltonian
    :math:`H = -J \sum_{<ij>} \sigma^z_i \sigma^z_j - h \sum_i \sigma^x_i`
    """
    sites = get_sites()
    H = -h * sum(sigma_x(i) for i in range(sites.nstates))
    neighbors = sites.get_neighbor()
    H += -J * sum(sigma_z(i) * sigma_z(j) for i, j in neighbors)
    return H


def Hubbard(
    U: Number,
    t: Union[Number, Sequence[Number]] = 1.0,
    n_neighbor: Union[int, Sequence[int]] = 1,
) -> Operator:
    r"""
    Hubbard Hamiltonian
    :math:`H = -t_n \sum_{<ij>_n} \sum_{s \in \{↑,↓\}} (c_{i,s}^† c_{j,s} + c_{j,s}^† c_{i,s}) + U \sum_i n_{i↑} n_{i↓}`
    """
    sites = get_sites()
    if isinstance(t, Number):
        t = [t]
    if isinstance(n_neighbor, Number):
        n_neighbor = [n_neighbor]
    if len(t) != len(n_neighbor):
        raise ValueError("'t' and 'n_neighbor' should have the same length.")
    neighbors, signs = sites.get_neighbor(n_neighbor, return_sign=True)

    def hop(i, j):
        hop_up = create_u(i) * annihilate_u(j) + create_u(j) * annihilate_u(i)
        hop_down = create_d(i) * annihilate_d(j) + create_d(j) * annihilate_d(i)
        return hop_up + hop_down

    H = 0
    for neighbor, sign, tn in zip(neighbors, signs, t):
        for (i, j), s in zip(neighbor, sign):
            H += -s.item() * tn * hop(i, j)

    H += U * sum(number_u(i) * number_d(i) for i in range(sites.nsites))
    return H
