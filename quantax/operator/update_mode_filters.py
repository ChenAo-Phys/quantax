from typing import Sequence
from ..global_defs import get_sites, PARTICLE_TYPE

#: Operator characters that act diagonally in the computational basis, i.e. they
#: leave the configuration unchanged. Every other character ("+", "-", "x", "y")
#: flips the site it acts on. This is the single source of truth shared with the
#: operator application in :mod:`quantax.operator.operator`.
DIAGONAL_OPS = frozenset({"I", "n", "z"})


def none_filter(opstr: str, indices: Sequence[int]) -> dict[str, int]:
    """
    A filter that assigns no update mode.

    All operator terms are grouped together regardless of how many sites they
    change, so the connected configurations are evaluated with a direct forward
    pass rather than a reference (low-rank) update.

    :param opstr:
        The operator string.

    :param indices:
        The site indices the operator acts on.

    :return:
        An empty dictionary.
    """
    return {}


def nflips_filter(opstr: str, indices: Sequence[int]) -> dict[str, int]:
    """
    A filter that assigns the number of flipped sites (``nflips``) as the update
    mode.

    ``nflips`` is the number of off-diagonal characters in ``opstr`` (everything
    not in :data:`DIAGONAL_OPS`), which equals the number of single-site flips the
    term applies to a configuration. Operator terms are then grouped by ``nflips``
    so that a model supporting reference updates can size its low-rank update
    accordingly. The ``"nflips"`` key is the update mode requested by such models
    via ``required_update_modes``.

    :param opstr:
        The operator string.

    :param indices:
        The site indices the operator acts on.

    :return:
        A dictionary ``{"nflips": <number of flipped sites>}``.
    """
    return {"nflips": sum(1 for op in opstr if op not in DIAGONAL_OPS)}


def nflips_up_dn_filter(opstr: str, indices: Sequence[int]) -> dict[str, int]:
    """
    A filter that assigns the numbers of flipped spin-up and spin-down modes
    (``nflips_up`` and ``nflips_dn``) as the update mode.

    ``nflips_up`` counts the off-diagonal characters in ``opstr`` (everything not
    in :data:`DIAGONAL_OPS`) acting on spin-up modes (index below ``Nsites``), and
    ``nflips_dn`` the ones acting on spin-down modes. The two keys are the update
    modes requested via ``required_update_modes`` by models whose low-rank updates
    treat the two spin sectors separately (e.g. `~quantax.model.SingletPair`).
    This filter is only meaningful for spinful fermion systems.

    :param opstr:
        The operator string.

    :param indices:
        The site indices the operator acts on.

    :return:
        A dictionary
        ``{"nflips_up": <flipped spin-up modes>, "nflips_dn": <flipped spin-down modes>}``.
    """
    sites = get_sites()
    if sites.particle_type != PARTICLE_TYPE.spinful_fermion:
        raise ValueError(
            "`nflips_up_dn_filter` is only defined for spinful fermion systems, "
            f"got particle type {sites.particle_type.name}."
        )

    flip_inds = [i for op, i in zip(opstr, indices) if op not in DIAGONAL_OPS]
    nflips_up = sum(1 for i in flip_inds if i < sites.Nsites)
    return {"nflips_up": nflips_up, "nflips_dn": len(flip_inds) - nflips_up}
