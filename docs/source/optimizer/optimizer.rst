optimizer
=========

.. currentmodule:: quantax.optimizer

Quantum natural gradient descent
--------------------------------

.. autosummary::
    :nosignatures:
    :toctree:

    QNGD
    StochasticQNGD
    ExactQNGD
    SR
    SPRING
    MARCH
    AdamSR
    ER
    TimeEvol

Supervised learning
--------------------------------

.. autosummary::
    :nosignatures:
    :toctree:

    Supervised
    SupervisedAdam
    SupervisedExact

Gradient sources
--------------------------------

.. autosummary::
    :nosignatures:
    :toctree:

    EnergyGrad
    OverlapGrad

Update strategies
--------------------------------

.. autosummary::
    :nosignatures:
    :toctree:

    Updater
    PlainUpdater
    SpringUpdater
    MarchUpdater
    AdamUpdater

Solvers
--------------------------------

Iterative solvers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree:

    lstsq_shift_cg
    minnorm_shift_cg
    lsmr

Non-iterative solvers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree:

    minnorm_shift_eig
    lstsq_shift_eig
    auto_shift_eig
    pinvh_solve
    minnorm_pinv_eig
    lstsq_pinv_eig
    auto_pinv_eig
    block_pinv_eig
    sgd_solver
