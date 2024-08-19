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
    ParticleConserve
    SpinInverse
    ParticleHole
    Translation
    TransND
    Trans1D
    Trans2D
    Trans3D
    LinearTransform
    Flip
    Rotation
    C4v
    D6
