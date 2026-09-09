nn
========

.. currentmodule:: quantax.nn


Modules
-----------

.. autosummary::
    :toctree:

    Sequential
    RawInputLayer
    RefModel


Activation function
-------------------

.. autosummary::
    :toctree:

    sinhp1_by_scale
    prod_by_log
    exp_by_scale
    exp_by_log
    crelu
    cardioid
    pair_cpl


Initializers
---------------

The functions below take an `equinox.nn.Linear <https://docs.kidger.site/equinox/api/nn/linear/#equinox.nn.Linear>`__
or `equinox.nn.Conv <https://docs.kidger.site/equinox/api/nn/conv/#equinox.nn.Conv>`__ layer,
initialize its weight with the given scheme and set its bias to 0.

.. autosummary::
    :toctree:

    apply_lecun_normal
    apply_he_normal
    apply_glorot_normal

Quantax also re-exports the JAX initializers with their axes fixed to the convention of
`equinox.nn.Linear <https://docs.kidger.site/equinox/api/nn/linear/#equinox.nn.Linear>`__
and `equinox.nn.Conv <https://docs.kidger.site/equinox/api/nn/conv/#equinox.nn.Conv>`__ weights (``in_axis=1``, ``out_axis=0``,
``batch_axis=()``), namely `~jax.nn.initializers.variance_scaling`,
`~jax.nn.initializers.lecun_normal`, `~jax.nn.initializers.lecun_uniform`,
`~jax.nn.initializers.glorot_normal`, `~jax.nn.initializers.glorot_uniform`,
`~jax.nn.initializers.he_normal` and `~jax.nn.initializers.he_uniform`.
They are called as ``initializer(key, shape, dtype)`` and are useful when a weight
is generated directly instead of through an Equinox layer.

.. code-block:: python

    import jax.numpy as jnp
    import quantax as qtx

    W = qtx.nn.lecun_normal(qtx.get_subkeys(), (16, 8), jnp.float32)


Sign structures
----------------------
.. autosummary::
    :toctree:

    compute_sign
    marshall_sign
    stripe_sign
    neel120_phase


Conv layers
-------------------

.. autosummary::
    :toctree:

    ReshapeConv
    Conv
    ConvSymmetrize
    circular_pad


Embedding
-------------------

.. autosummary::
    :toctree:

    input_to_index
    Embedding

Fermions
--------------------------------

.. autosummary::
    :nosignatures:
    :toctree:

    fermion_idx
    changed_inds
    permute_sign
    fermion_inverse_sign
    fermion_reorder_sign
