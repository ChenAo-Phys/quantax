from typing import Sequence

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
