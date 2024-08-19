operator
========

.. currentmodule:: quantax.operator

Main class
------------

.. autosummary::
    :nosignatures:
    :toctree:

    Operator


Site operators
-----------------

The customized operators are supported by simple operations of site operators.
See the definition of transverse-field Ising Hamiltonian below as an example.

.. code-block:: python

   from quantax.sites import Square
   from quantax.operator import sigma_x, sigma_z

   lattice = Square(4)

   TFIsing = -sum(sigma_x(i) for i in range(lattice.nstates))
   TFIsing += -sum(sigma_z(i) * sigma_z(j) for i, j in lattice.get_neighbor())


The index of site operators can be a site index or a site coordinate.
In the latter case, the boundary condition is taken into account automatically.
For example,

.. code-block:: python

    from quantax.sites import Square
    from quantax.operator import create_u

    lattice = Square(4, boundary=-1, is_fermion=True)  # Anti-periodic boundary
    
    # The two following definitions are equivalent
    op1 = -create_u(0, 0)
    op2 = create_u(0, 4)


.. autosummary::
    :toctree:

    sigma_x
    sigma_y
    sigma_z
    sigma_p
    sigma_m
    S_x
    S_y
    S_z
    S_p
    S_m
    create_u
    create_d
    annihilate_u
    annihilate_d
    number_u
    number_d


Hamiltonians
---------------

.. autosummary::
    :toctree:

    Heisenberg
    Ising
    Hubbard
