symmetry
========

``symmetry`` specifies the symmetry sector of the studied system. It can be used to
reduce the cost of exact diagonalization and project the wave function into certain
symmetry sectors.

.. currentmodule:: quantax.symmetry

Main class
------------

.. autosummary::
    :nosignatures:
    :toctree:

    Symmetry


Common symmetries
-----------------

.. autosummary::
    :toctree:

    Identity
    SpinInverse
    ParticleHole
    Translation
    TransND
    LinearTransform
    Flip
    Rotation
    C4v
    D6
