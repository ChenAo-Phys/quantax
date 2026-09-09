operator
========

.. currentmodule:: quantax.operator

Main classes
------------

.. autosummary::
    :nosignatures:
    :toctree:

    Operator
    OpTerm
    OpTermJAX


Site operators
-----------------

The customized operators are supported by simple operations of site operators.
See the definition of transverse-field Ising Hamiltonian below as an example.

.. code-block:: python

   from quantax.sites import Square
   from quantax.operator import sigma_x, sigma_z

   lattice = Square(4)

   TFIsing = -sum(sigma_x(i) for i in range(lattice.Nsites))
   TFIsing += -sum(sigma_z(i) @ sigma_z(j) for i, j in lattice.get_neighbor())


The index of site operators can be a site index or a site coordinate.
In the latter case, the boundary condition is taken into account automatically.
For example,

.. code-block:: python

    from quantax import PARTICLE_TYPE
    from quantax.sites import Square
    from quantax.operator import create

    # Anti-periodic boundary
    lattice = Square(4, boundary=-1, particle_type=PARTICLE_TYPE.spinless_fermion)
    
    # The two following definitions are equivalent
    op1 = -create(0, 0)
    op2 = create(0, 4)


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
    create
    create_u
    create_d
    annihilate
    annihilate_u
    annihilate_d
    number
    number_u
    number_d


Hamiltonians
---------------

.. autosummary::
    :toctree:

    Heisenberg
    Ising
    Hubbard
    tJ
    tV


Update mode filters
-------------------

When an operator is applied to a batch of configurations, its terms are grouped
according to an *update mode*, a dictionary of static integers attached to each group.
Models supporting low-rank reference updates (`~quantax.nn.RefModel`) declare the keys
they need in ``required_update_modes``, and `~Operator.apply_update_mode_filter`
regroups the operator terms accordingly. The filters below cover the common choices,
and a custom filter is any callable ``(opstr, indices) -> dict[str, int]``.

.. autosummary::
    :toctree:

    none_filter
    nflips_filter
    nflips_up_dn_filter
