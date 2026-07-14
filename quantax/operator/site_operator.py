r"""
Single-site operators.

Every constructor in this module returns an :class:`~quantax.operator.Operator`
acting on one site, and accepts the site through a common ``*index`` convention:

- **Flat index:** a single integer ``i`` with ``0 <= i < Nsites`` selects site ``i``
  directly. This is the only form allowed for a generic (non-lattice) ``Sites``.

- **Lattice coordinates:** for a :class:`~quantax.sites.Lattice`, the site may be
  given as ``(sublattice, x, y, ...)``, i.e. the in-cell index followed by one cell
  coordinate per spatial axis. The sublattice index may be omitted when there is a
  single site per unit cell.

  Out-of-range coordinates wrap around according to the boundary condition of each
  axis, and every wrap multiplies the operator strength by the boundary sign
  (``+1`` for PBC, ``-1`` for APBC, ``0`` for OBC). For example, on an anti-periodic
  chain ``sigma_z(-1)`` addresses the last site with a flipped sign.
"""

from __future__ import annotations
import numpy as np
from . import Operator, OpTerm
from ..sites import Lattice
from ..global_defs import PARTICLE_TYPE, is_default_cpl, get_sites


def _get_site_operator(
    index: tuple[int, ...],
    opstr: str,
    strength: float = 1.0,
    is_fermion_down: bool = False,
) -> Operator:
    """
    Build a single-site :class:`Operator`. See the module docstring for the
    ``index`` convention (flat index vs. lattice coordinates with boundary sign).

    :param index: The site, as a single flat index or as lattice coordinates.
    :param opstr: The QuSpin operator string, e.g. ``"x"``, ``"z"``, ``"+"``, ``"n"``.
    :param strength: A prefactor multiplying the operator.
    :param is_fermion_down: Whether to target the spin-down mode of a spinful fermion,
        which lives at ``idx + Nsites`` in the configuration array.
    """
    sites = get_sites()
    if len(index) == 1 and 0 <= index[0] < sites.Nsites:
        idx = int(index[0])
    else:
        if not isinstance(sites, Lattice):
            raise ValueError(
                f"For non-lattice `Sites`, the index must be a single integer in "
                f"[0, {sites.Nsites}), got {index}."
            )
        shape = sites.shape

        if len(index) == len(shape):
            xyz = [index[0]]
            index = index[1:]
        elif len(index) == len(shape) - 1 and shape[0] == 1:
            xyz = [0]
        else:
            raise ValueError(
                f"The index {index} doesn't match the lattice shape {shape}. Provide "
                f"either a single flat site index, or one coordinate per spatial axis "
                f"(optionally preceded by the sublattice index)."
            )
        sign = []
        for x, l, bc in zip(index, shape[1:], sites.boundary):
            xyz.append(x % l)
            sign.append(bc ** abs(x // l))
        idx = sites.index_from_xyz[tuple(xyz)].item()
        strength *= np.prod(sign).item()
    if is_fermion_down:
        idx += sites.Nsites

    op_term = OpTerm(opstr, [strength], [[idx]])
    return Operator([op_term])


def sigma_x(*index: int) -> Operator:
    r"""
    :math:`\sigma^x` operator for spin and fermion systems.

    For spin systems,

    .. math::

        \sigma^x =
        \begin{pmatrix}
            0 & 1 \\
            1 & 0
        \end{pmatrix}

    :param index: Site to act on, as a flat index or lattice coordinates; see the
        module docstring for the convention.
    """
    return _get_site_operator(index, "x", 2.0)


def sigma_y(*index: int) -> Operator:
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

    :param index: Site to act on, as a flat index or lattice coordinates; see the
        module docstring for the convention.
    """
    if not is_default_cpl():
        raise RuntimeError(
            "`sigma_y` operator is not supported for real default data types, "
            "try `quantax.set_default_dtype(np.complex128)` before calling `sigma_y`, "
            "or use `sigma_p` and `sigma_m` instead."
        )
    return _get_site_operator(index, "y", 2.0)


def sigma_z(*index: int) -> Operator:
    r"""
    :math:`\sigma^z` operator for spin and fermion systems.

    For spin systems,

    .. math::

        \sigma^z =
        \begin{pmatrix}
            1 & 0 \\
            0 & -1
        \end{pmatrix}

    :param index: Site to act on, as a flat index or lattice coordinates; see the
        module docstring for the convention.
    """
    return _get_site_operator(index, "z", 2.0)


def sigma_p(*index: int) -> Operator:
    r"""
    :math:`\sigma^+` operator for spin systems.

    .. math::

        \sigma^+ =
        \begin{pmatrix}
            0 & 1 \\
            0 & 0
        \end{pmatrix}

    :param index: Site to act on, as a flat index or lattice coordinates; see the
        module docstring for the convention.
    """
    if get_sites().is_fermion:
        raise RuntimeError("`sigma_p` works for spin systems instead of fermions")
    return _get_site_operator(index, "+")


def sigma_m(*index: int) -> Operator:
    r"""
    :math:`\sigma^-` operator for spin systems.

    .. math::

        \sigma^- =
        \begin{pmatrix}
            0 & 0 \\
            1 & 0
        \end{pmatrix}

    :param index: Site to act on, as a flat index or lattice coordinates; see the
        module docstring for the convention.
    """
    if get_sites().is_fermion:
        raise RuntimeError("`sigma_m` works for spin systems instead of fermions")
    return _get_site_operator(index, "-")


def S_x(*index: int) -> Operator:
    r"""
    :math:`S^x` operator for spin and fermion systems.

    .. math::

        S^x =
        \begin{pmatrix}
            0 & 1/2 \\
            1/2 & 0
        \end{pmatrix}

    :param index: Site to act on, as a flat index or lattice coordinates; see the
        module docstring for the convention.
    """
    return _get_site_operator(index, "x")


def S_y(*index: int) -> Operator:
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

    :param index: Site to act on, as a flat index or lattice coordinates; see the
        module docstring for the convention.
    """
    if not is_default_cpl():
        raise RuntimeError(
            "'S_y' operator is not supported for real default data types, "
            "try `quantax.set_default_dtype(np.complex128)` before calling `S_y`, "
            "or use `S_p` and `S_m` instead."
        )
    return _get_site_operator(index, "y")


def S_z(*index: int) -> Operator:
    r"""
    :math:`S^z` operator

    .. math::

        S^z =
        \begin{pmatrix}
            1/2 & 0 \\
            0 & -1/2
        \end{pmatrix}

    :param index: Site to act on, as a flat index or lattice coordinates; see the
        module docstring for the convention.
    """
    return _get_site_operator(index, "z")


def S_p(*index: int) -> Operator:
    r"""
    :math:`S^+` operator

    .. math::

        S^+ =
        \begin{pmatrix}
            0 & 1 \\
            0 & 0
        \end{pmatrix}

    :param index: Site to act on, as a flat index or lattice coordinates; see the
        module docstring for the convention.
    """
    return sigma_p(*index)


def S_m(*index: int) -> Operator:
    r"""
    :math:`S^-` operator

    .. math::

        S^- =
        \begin{pmatrix}
            0 & 0 \\
            1 & 0
        \end{pmatrix}

    :param index: Site to act on, as a flat index or lattice coordinates; see the
        module docstring for the convention.
    """
    return sigma_m(*index)


def create(*index: int) -> Operator:
    r"""
    :math:`c^†` operator

    :param index: Site to act on, as a flat index or lattice coordinates; see the
        module docstring for the convention.
    """
    if not get_sites().particle_type == PARTICLE_TYPE.spinless_fermion:
        raise RuntimeError("`create` only works for spinless fermions")
    return _get_site_operator(index, "+")


def create_u(*index: int) -> Operator:
    r"""
    :math:`c_↑^†` operator

    :param index: Site to act on, as a flat index or lattice coordinates; see the
        module docstring for the convention.
    """
    if not get_sites().particle_type == PARTICLE_TYPE.spinful_fermion:
        raise RuntimeError("`create_u` only works for spinful fermions")
    return _get_site_operator(index, "+")


def create_d(*index: int) -> Operator:
    r"""
    :math:`c_↓^†` operator

    :param index: Site to act on, as a flat index or lattice coordinates; see the
        module docstring for the convention.
    """
    if not get_sites().particle_type == PARTICLE_TYPE.spinful_fermion:
        raise RuntimeError("`create_d` only works for spinful fermions")
    return _get_site_operator(index, "+", is_fermion_down=True)


def annihilate(*index: int) -> Operator:
    r"""
    :math:`c` operator

    :param index: Site to act on, as a flat index or lattice coordinates; see the
        module docstring for the convention.
    """
    if not get_sites().particle_type == PARTICLE_TYPE.spinless_fermion:
        raise RuntimeError("`annihilate` only works for spinless fermions")
    return _get_site_operator(index, "-")


def annihilate_u(*index: int) -> Operator:
    r"""
    :math:`c_↑` operator

    :param index: Site to act on, as a flat index or lattice coordinates; see the
        module docstring for the convention.
    """
    if not get_sites().particle_type == PARTICLE_TYPE.spinful_fermion:
        raise RuntimeError("`annihilate_u` only works for spinful fermions")
    return _get_site_operator(index, "-")


def annihilate_d(*index: int) -> Operator:
    r"""
    :math:`c_↓` operator

    :param index: Site to act on, as a flat index or lattice coordinates; see the
        module docstring for the convention.
    """
    if not get_sites().particle_type == PARTICLE_TYPE.spinful_fermion:
        raise RuntimeError("`annihilate_d` only works for spinful fermions")
    return _get_site_operator(index, "-", is_fermion_down=True)


def number(*index: int) -> Operator:
    r"""
    :math:`n = c^† c` operator

    :param index: Site to act on, as a flat index or lattice coordinates; see the
        module docstring for the convention.
    """
    if not get_sites().particle_type == PARTICLE_TYPE.spinless_fermion:
        raise RuntimeError("`number` only works for spinless fermions")
    return _get_site_operator(index, "n")


def number_u(*index: int) -> Operator:
    r"""
    :math:`n_↑ = c_↑^† c_↑` operator

    :param index: Site to act on, as a flat index or lattice coordinates; see the
        module docstring for the convention.
    """
    if not get_sites().particle_type == PARTICLE_TYPE.spinful_fermion:
        raise RuntimeError("`number_u` only works for spinful fermions")
    return _get_site_operator(index, "n")


def number_d(*index: int) -> Operator:
    r"""
    :math:`n_↓ = c_↓^† c_↓` operator

    :param index: Site to act on, as a flat index or lattice coordinates; see the
        module docstring for the convention.
    """
    if not get_sites().particle_type == PARTICLE_TYPE.spinful_fermion:
        raise RuntimeError("`number_d` only works for spinful fermions")
    return _get_site_operator(index, "n", is_fermion_down=True)
