from __future__ import annotations
import numpy as np
from . import Operator
from ..sites import Lattice
from ..global_defs import is_default_cpl, get_sites


def _get_site_operator(index: tuple, opstr: str, strength: float = 1.) -> Operator:
    sites = get_sites()
    if len(index) == 1 and 0 < index < sites.nsites:
        index = index[0]
    else:
        if not isinstance(sites, Lattice):
            raise ValueError(
                "The sites must be lattice when the index is given by coordinate"
            )
        shape = sites.shape

        if len(index) == len(shape):
            xyz = [index[0]]
            index = index[1:]
        elif len(index) == len(shape) - 1 and shape[0] == 1:
            xyz = [0]
        else:
            raise ValueError("The input index doesn't match the shape of lattice.")
        sign = []
        for x, l, bc in zip(index, shape[1:], sites.boundary):
            xyz.append(x % l)
            sign.append(bc ** abs(x // l))
        index = sites.index_from_xyz[tuple(xyz)].item()
        strength *= np.prod(sign).item()
    return Operator([[opstr, [[strength, index]]]])


def sigma_x(*index) -> Operator:
    return _get_site_operator(index, "x")


def sigma_y(*index) -> Operator:
    if not is_default_cpl():
        raise RuntimeError(
            "'sigma_y' operator is not supported for real default data types,"
            "try `quantax.set_default_dtype(np.complex128)` before calling `sigma_y`,"
            "or use `sigma_p` and `sigma_m` instead."
        )
    return _get_site_operator(index, "y")


def sigma_z(*index) -> Operator:
    return _get_site_operator(index, "z")


def sigma_p(*index) -> Operator:
    return _get_site_operator(index, "+")


def sigma_m(*index) -> Operator:
    return _get_site_operator(index, "-")


def S_x(*index) -> Operator:
    return _get_site_operator(index, "x", 0.5)


def S_y(*index) -> Operator:
    if not is_default_cpl():
        raise RuntimeError(
            "'S_y' operator is not supported for real default data types,"
            "try `quantax.set_default_dtype(np.complex128)` before calling `S_y`,"
            "or use `S_p` and `S_m` instead."
        )
    return _get_site_operator(index, "y", 0.5)


def S_z(*index) -> Operator:
    return _get_site_operator(index, "z", 0.5)


def S_p(*index) -> Operator:
    return _get_site_operator(index, "+")


def S_m(*index) -> Operator:
    return _get_site_operator(index, "-")
