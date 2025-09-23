.. global_defs:

global_defs
================

In `global_defs`, users can define and check global constants used in the simulation, 
including datatypes, random keys, and Hilbert space information. The settings should
be done right after importing Quantax, and before defining any model or state.

.. autofunction:: quantax.set_default_dtype
.. autofunction:: quantax.get_default_dtype
.. autofunction:: quantax.get_real_dtype
.. autofunction:: quantax.is_default_cpl
.. autofunction:: quantax.set_random_seed
.. autofunction:: quantax.get_subkeys
.. autofunction:: quantax.PARTICLE_TYPE
.. autofunction:: quantax.get_sites
.. autofunction:: quantax.get_lattice
