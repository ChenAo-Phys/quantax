from .qngd import QNGD, StochasticQNGD, ExactQNGD
from .sr import SR, SPRING, MARCH, ER, AdamSR
from .driver import Driver, Euler, AdaptiveHeunEvolution
from .supervised import Supervised, SupervisedAdam, SupervisedExact
from .solver import (
    pinvh_solve,
    lstsq_shift_cg,
    minnorm_shift_eig,
    lstsq_shift_eig,
    auto_shift_eig,
    pinvh_solve,
    minnorm_pinv_eig,
    lstsq_pinv_eig,
    auto_pinv_eig,
    block_pinv_eig,
    minsr_pinv_eig,
    sgd_solver,
)
from .time_evol import TimeEvol
