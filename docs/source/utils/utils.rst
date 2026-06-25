utils
=========

.. currentmodule:: quantax.utils

Data
--------------------------------

.. autosummary::
    :nosignatures:
    :toctree:

    DataTracer


Sharding
--------------------------------

.. autosummary::
    :nosignatures:
    :toctree:

    make_mesh
    get_distributed_P
    get_distributed_sharding
    get_replicated_sharding


Array
--------------------------------

.. autosummary::
    :nosignatures:
    :toctree:

    is_sharded_array
    to_distributed_array
    to_replicated_array
    global_to_local
    local_to_global
    local_to_replicated
    to_replicated_numpy
    array_extend
    array_set


Pytree
--------------------------------

.. autosummary::
    :nosignatures:
    :toctree:

    tree_fully_flatten
    filter_tree_map
    tree_split_cpl
    tree_combine_cpl
    apply_updates


Manipulating functions
------------------------------

.. autosummary::
    :nosignatures:
    :toctree:

    jit_chunk_vmap
    chunk_map


Customized arrays for large numbers
-----------------------------------
.. autosummary::
    :nosignatures:
    :toctree:

    LogArray
    ScaleArray


Basis states
--------------------------------

.. autosummary::
    :nosignatures:
    :toctree:

    ints_to_array
    array_to_ints
    neel
    stripe
    Sqz_factor
    rand_states
