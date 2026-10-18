"""Exact references and histograms shared by the sampler tests."""

import numpy as np
import jax.numpy as jnp
from quantax.sites import Chain
from quantax.state import DenseState
from quantax.symmetry import Identity
from quantax.utils import ints_to_array


def sector_configs() -> np.ndarray:
    """All configurations of the current (sector-restricted) basis, in basis order."""
    symm = Identity()
    symm.basis_make()
    return np.asarray(ints_to_array(symm.basis.states))


def skewed_dense_state(
    L: int, up: float = 2.0
) -> tuple[DenseState, np.ndarray, np.ndarray]:
    """
    A real, positive `DenseState` on a length-``L`` spin chain with amplitude ``up``
    for ``s[0] == +1`` and 1 otherwise, with no Sz conservation so the full ``2**L``
    Hilbert space is reachable by single-spin flips. Returns the state together with
    the basis-ordered configurations and the (unnormalized) wave function, so a test
    can compute exact expectations.
    """
    Chain(L, boundary=1)
    symm = Identity()
    symm.basis_make()
    configs = np.asarray(ints_to_array(symm.basis.states))
    psi = np.where(configs[:, 0] > 0, up, 1.0)
    return DenseState(jnp.asarray(psi), symm), configs, psi


def config_codes(spins) -> np.ndarray:
    """Bijective integer code of +-1 configurations, for histograms and lookups."""
    spins = np.asarray(spins)
    return np.sum((spins > 0).astype(np.int64) << np.arange(spins.shape[1]), axis=1)


def config_index(spins, configs: np.ndarray) -> np.ndarray:
    """Index of every configuration in ``configs``, asserting it is one of them."""
    lookup = np.full(1 << configs.shape[1], -1)
    lookup[config_codes(configs)] = np.arange(len(configs))
    idx = lookup[config_codes(spins)]
    assert np.all(idx >= 0), "configurations outside of the basis"
    return idx


def histogram(spins, configs: np.ndarray) -> np.ndarray:
    """Counts of every configuration of ``configs`` among ``spins``."""
    return np.bincount(config_index(spins, configs), minlength=len(configs))
