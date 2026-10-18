"""
End-to-end VMC test in the default single precision: Metropolis sampling, local
energies, SR steps and parameter updates of a variational state, checked against
exact diagonalization.
"""

import numpy as np
import jax
from quantax.sites import Chain
from quantax.state import Variational
from quantax.model import Jastrow
from quantax.operator import Ising
from quantax.sampler import LocalFlip
from quantax.optimizer import SR
from quantax.symmetry import TransND

L = 6
NDEV = jax.device_count()
NS = -(-32 // NDEV) * NDEV  # 32 rounded up to a multiple of NDEV


def _ising_dense(h):
    """
    All configurations of the periodic chain, their zz energy and the dense matrix of
    ``Ising(h)``, built in numpy independently of the operator under test.
    """
    configs = 1 - 2 * ((np.arange(2**L)[:, None] >> np.arange(L)) & 1)
    zz = -np.sum(configs * np.roll(configs, 1, axis=1), axis=1)
    single_flip = np.sum(configs != configs[:, None], axis=2) == 1
    return configs.astype(np.int8), zz, np.diag(zz) - h * single_flip


def test_vmc_loop_converges_and_samples_psi_squared():
    # A Jastrow factor with all couplings free, projected onto zero momentum, has
    # more parameters (36) than samples (up to 5 devices), so the default solver
    # takes the MinSR branch. The chains start from the ferromagnetic state, which
    # is cheaper to compile than random states, and are thermalized by the sampler.
    Chain(L)
    H = Ising(h=1.0)
    state = Variational(Jastrow(), symm=TransND())
    sampler = LocalFlip(state, NS, initial_spins=np.ones((NS, L), np.int8))
    opt = SR(state, H)
    for _ in range(20):
        step = opt.get_step(sampler.sweep())
        state.update(step * 0.1)

    # The exact energy of the final state is free of sampling noise. Over 20 seeds
    # its relative error is at most 1.4e-4, starting from 13-44%.
    configs, zz, Hd = _ising_dense(1.0)
    psi = np.asarray(state(configs), dtype=np.float64)
    E = psi @ Hd @ psi / (psi @ psi)
    E0 = np.linalg.eigvalsh(Hd)[0]
    assert E < E0 + 1e-3 * abs(E0)

    # The sampler draws |psi|^2 of the trained state: the sampled mean of the zz
    # energy agrees with its exact value within the error bar of the mean over the
    # independent chains. The zz energy is bounded, unlike the local energy, whose
    # heavy tail makes such an error bar unreliable. Over 20 seeds the deviation is
    # at most 2.2 error bars, while sampling |psi| instead gives more than 10.
    spins = np.stack([np.asarray(sampler.sweep().spins) for _ in range(8)])
    zz_chains = -np.mean(np.sum(spins * np.roll(spins, 1, axis=2), axis=2), axis=0)
    zz_exact = psi**2 @ zz / np.sum(psi**2)
    err = zz_chains.std(ddof=1) / np.sqrt(NS)
    assert abs(zz_chains.mean() - zz_exact) < 5 * err
