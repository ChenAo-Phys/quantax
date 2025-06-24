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

    get_global_sharding
    get_replicate_sharding

Array
--------------------------------

.. autosummary::
    :nosignatures:
    :toctree:

    is_sharded_array
    to_global_array
    to_replicate_array
    global_to_local
    local_to_global
    local_to_replicate
    to_replicate_numpy
    array_extend
    array_set
    sharded_segment_sum

Spins
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
    
Pytree
--------------------------------

.. autosummary::
    :nosignatures:
    :toctree:

    tree_fully_flatten
    filter_replicate
    filter_tree_map
    tree_split_cpl
    tree_combine_cpl
    apply_updates

Function
------------------------------

.. autosummary::
    :nosignatures:
    :toctree:

    chunk_shard_vmap
    chunk_map
    shmap

Linear algebra
------------------------------

.. autosummary::
    :nosignatures:
    :toctree:

    det
    pfaffian
    det_update_rows
    det_update_gen
    pfa_eye
    pfa_update
