model
========

.. currentmodule:: quantax.model


Neural quantum states
---------------------
It's recommended to use `~ResConv` when one needs a deep NQS.

.. autosummary::
    :toctree:

    SingleDense
    RBM_Dense
    SingleConv
    RBM_Conv
    ResConv


Fermionic Mean-field
--------------------

.. autosummary::
    :toctree:

    GeneralDet
    RestrictedDet
    UnrestrictedDet
    MultiDet
    GeneralPf
    SingletPair
    MultiPf
    PartialPair


Backflow
--------

The backflow models make the mean-field orbitals configuration-dependent through a
neural network ``net``.

.. autosummary::
    :toctree:

    DetBackflow
    PfBackflow


Jastrow factors
---------------

.. autosummary::
    :toctree:

    Jastrow
    GeneralJastrow
