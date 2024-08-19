sites
=====

``sites`` contains the basic information of the system, including spatial positions 
and the particle type.

.. currentmodule:: quantax.sites

.. note::
    Currently quantax only supports spin-1/2 systems or spinful fermion systems.

.. warning::
    Unlike other NQS packages, in quantax the spatial geometry and hilbert space 
    information provided by ``Sites`` is a **global constant**.
    There should only be one ``Sites`` or ``Lattice`` defined in the beginning of a 
    program and kept fixed for the whole program.

Main classes
------------

.. autosummary::
    :nosignatures:
    :toctree:

    Sites
    Lattice


Common lattices
---------------

.. autosummary::
    :toctree:

    Grid
    Chain
    Square
    Cube
    Pyrochlore
    Triangular
    TriangularB