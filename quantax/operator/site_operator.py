from __future__ import annotations
import numpy as np
from . import Operator
from ..sites import Lattice
from ..global_defs import PARTICLE_TYPE, is_default_cpl, get_sites


def _get_site_operator(
    index: tuple, opstr: str, strength: float = 1.0, is_fermion_down: bool = False
) -> Operator:
    sites = get_sites()
    if len(index) == 1 and 0 <= index[0] < sites.Nsites:
        index = int(index[0])
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
    if is_fermion_down:
        index += sites.Nsites
    return Operator([[opstr, [[strength, index]]]])


def sigma_x(*index) -> Operator:
    r"""
    :math:`\sigma^x` operator for spin and fermion systems.

    For spin systems,

    .. math::

        \sigma^x =
        \begin{pmatrix}
            0 & 1 \\
            1 & 0
        \end{pmatrix}
    """
    return _get_site_operator(index, "x", 2.0)


def sigma_y(*index) -> Operator:
    r"""
    :math:`\sigma^y` operator for spin and fermion systems.

    For spin systems,

    .. math::

        \sigma^y =
        \begin{pmatrix}
            0 & -i \\
            i & 0
        \end{pmatrix}

    .. warning::

        The :math:`\sigma^y` operator works only if the default data type is set to
        complex in `quantax.set_default_dtype`.
    """
    if not is_default_cpl():
        raise RuntimeError(
            "`sigma_y` operator is not supported for real default data types, "
            "try `quantax.set_default_dtype(np.complex128)` before calling `sigma_y`, "
            "or use `sigma_p` and `sigma_m` instead."
        )
    return _get_site_operator(index, "y", 2.0)


def sigma_z(*index) -> Operator:
    r"""
    :math:`\sigma^z` operator for spin and fermion systems.

    For spin systems,

    .. math::

        \sigma^z =
        \begin{pmatrix}
            1 & 0 \\
            0 & -1
        \end{pmatrix}
    """
    return _get_site_operator(index, "z", 2.0)


def sigma_p(*index) -> Operator:
    r"""
    :math:`\sigma^+` operator for spin systems.

    .. math::

        \sigma^+ =
        \begin{pmatrix}
            0 & 1 \\
            0 & 0
        \end{pmatrix}
    """
    if get_sites().is_fermion:
        raise RuntimeError("`sigma_p` works for spin systems instead of fermions")
    return _get_site_operator(index, "+")


def sigma_m(*index) -> Operator:
    r"""
    :math:`\sigma^-` operator for spin systems.

    .. math::

        \sigma^- =
        \begin{pmatrix}
            0 & 0 \\
            1 & 0
        \end{pmatrix}
    """
    if get_sites().is_fermion:
        raise RuntimeError("`sigma_m` works for spin systems instead of fermions")
    return _get_site_operator(index, "-")


def S_x(*index) -> Operator:
    r"""
    :math:`S^x` operator for spin and fermion systems.

    .. math::

        S^x =
        \begin{pmatrix}
            0 & 1/2 \\
            1/2 & 0
        \end{pmatrix}
    """
    return _get_site_operator(index, "x")


def S_y(*index) -> Operator:
    r"""
    :math:`S^y` operator

    .. math::

        S^y =
        \begin{pmatrix}
            0 & -i/2 \\
            i/2 & 0
        \end{pmatrix}

    .. warning::

        The :math:`S^y` operator works only if the default data type is set to
        complex in `quantax.set_default_dtype`.
    """
    if not is_default_cpl():
        raise RuntimeError(
            "'S_y' operator is not supported for real default data types, "
            "try `quantax.set_default_dtype(np.complex128)` before calling `S_y`, "
            "or use `S_p` and `S_m` instead."
        )
    return _get_site_operator(index, "y")


def S_z(*index) -> Operator:
    r"""
    :math:`S^z` operator

    .. math::

        S^z =
        \begin{pmatrix}
            1/2 & 0 \\
            0 & -1/2
        \end{pmatrix}
    """
    return _get_site_operator(index, "z")


def S_p(*index) -> Operator:
    r"""
    :math:`S^+` operator

    .. math::

        S^+ =
        \begin{pmatrix}
            0 & 1 \\
            0 & 0
        \end{pmatrix}
    """
    return sigma_p(*index)


def S_m(*index) -> Operator:
    r"""
    :math:`S^-` operator

    .. math::

        S^- =
        \begin{pmatrix}
            0 & 0 \\
            1 & 0
        \end{pmatrix}
    """
    return sigma_m(*index)


def create(*index) -> Operator:
    r"""
    :math:`c^†` operator
    """
    if not get_sites().particle_type == PARTICLE_TYPE.spinless_fermion:
        raise RuntimeError("`create` only works for spinless fermions")
    return _get_site_operator(index, "+")


def create_u(*index) -> Operator:
    r"""
    :math:`c_↑^†` operator
    """
    if not get_sites().particle_type == PARTICLE_TYPE.spinful_fermion:
        raise RuntimeError("`create_u` only works for spinful fermions")
    return _get_site_operator(index, "+")


def create_d(*index) -> Operator:
    r"""
    :math:`c_↓^†` operator
    """
    if not get_sites().particle_type == PARTICLE_TYPE.spinful_fermion:
        raise RuntimeError("`create_d` only works for spinful fermions")
    return _get_site_operator(index, "+", is_fermion_down=True)


def annihilate(*index) -> Operator:
    r"""
    :math:`c` operator
    """
    if not get_sites().particle_type == PARTICLE_TYPE.spinless_fermion:
        raise RuntimeError("`annihilate` only works for spinless fermions")
    return _get_site_operator(index, "-")


def annihilate_u(*index) -> Operator:
    r"""
    :math:`c_↑` operator
    """
    if not get_sites().particle_type == PARTICLE_TYPE.spinful_fermion:
        raise RuntimeError("`annihilate_u` only works for spinful fermions")
    return _get_site_operator(index, "-")


def annihilate_d(*index) -> Operator:
    r"""
    :math:`c_↓` operator
    """
    if not get_sites().particle_type == PARTICLE_TYPE.spinful_fermion:
        raise RuntimeError("`annihilate_d` only works for spinful fermions")
    return _get_site_operator(index, "-", is_fermion_down=True)


def number(*index) -> Operator:
    r"""
    :math:`n = c^† c` operator
    """
    if not get_sites().particle_type == PARTICLE_TYPE.spinless_fermion:
        raise RuntimeError("`number` only works for spinless fermions")
    return _get_site_operator(index, "n")


def number_u(*index) -> Operator:
    r"""
    :math:`n_↑ = c_↑^† c_↑` operator
    """
    if not get_sites().particle_type == PARTICLE_TYPE.spinful_fermion:
        raise RuntimeError("`number_u` only works for spinful fermions")
    return _get_site_operator(index, "n")


def number_d(*index) -> Operator:
    r"""
    :math:`n_↓ = c_↓^† c_↓` operator
    """
    if not get_sites().particle_type == PARTICLE_TYPE.spinful_fermion:
        raise RuntimeError("`number_d` only works for spinful fermions")
    return _get_site_operator(index, "n", is_fermion_down=True)
