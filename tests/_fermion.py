"""Test-only references for fermionic wavefunctions."""

import numpy as np


def pfaffian(A):
    """Pfaffian by expansion along the first row (small matrices only)."""
    n = A.shape[0]
    if n % 2 == 1:
        return 0.0
    out = 1.0 if n == 0 else 0.0
    for j in range(1, n):
        rest = [k for k in range(1, n) if k != j]
        out += (-1) ** (j - 1) * A[0, j] * pfaffian(A[np.ix_(rest, rest)])
    return out
