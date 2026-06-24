"""Test-only dtype helper.

``use_dtype`` sets Quantax's default dtype *without* the global
``jax_enable_x64`` side effect that ``qtx.set_default_dtype`` has (see
``quantax.global_defs.set_default_dtype``). 64-bit tests scope precision with
the ``x64`` fixture (see ``conftest.py``) instead, so the global flag never
leaks across tests.
"""

import jax.numpy as jnp
import quantax as qtx


def use_dtype(dtype):
    qtx.global_defs.DTYPE = jnp.dtype(dtype)
