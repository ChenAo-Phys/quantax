"""Helpers shared by the operator tests."""

import numpy as np


def dense(op) -> np.ndarray:
    """Dense matrix of an operator as a plain numpy array."""
    return np.asarray(op.todense())
