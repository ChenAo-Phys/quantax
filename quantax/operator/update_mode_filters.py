def nflips_filter(opstr: str, indices: tuple[int, ...]) -> dict[str, int]:
    """
    A filter function that selects the number of flips (nflips) update mode.
    
    :param opstr:
        The operator string.

    :param indices:
        The indices the operator acts on.

    :return:
        A dictionary with key "nflips" and its corresponding value.
    """
    return {"nflips": sum(1 for s in opstr if s not in ("I", "n", "z"))}
