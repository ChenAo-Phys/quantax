from __future__ import annotations
import numpy as np
from . import Operator
from ..sites import Lattice
from ..global_defs import is_default_cpl, get_sites


def _get_site_operator(index: tuple, opstr: str, strength: float = 1.) -> Operator:
    sites = get_sites()
    if not isinstance(index, int):
        index = np.asarray(index)
        if index.size > 1:
            if not isinstance(sites, Lattice):
                raise ValueError(
                    "The sites must be lattice when the index is given by coordinate"
                )
            shape = sites.shape
            xyz = tuple(x % shape[i] for i, x in enumerate(index))
            if len(xyz) == len(shape):
                index = sites.index_from_xyz[xyz]
            elif len(xyz) == len(shape) - 1 and shape[-1] == 1:
                index = sites.index_from_xyz[xyz][0]
            else:
                raise ValueError("The input index doesn't match the shape of sites")
        index = index.item()
    index = index % sites.nsites
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
