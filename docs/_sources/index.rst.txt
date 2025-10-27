.. Quantax documentation master file, created by
   sphinx-quickstart on Thu Aug  1 15:06:03 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the **Quantax** documentation!

**Quantax** is a Python package for flexible neural quantum states based on
`QuSpin <https://github.com/QuSpin/QuSpin/tree/dev_0.3.8>`_,
`JAX <https://github.com/google/jax>`_, and
`Equinox <https://github.com/patrick-kidger/equinox>`_.


Installation
============

Requires Python 3.10+, JAX 0.4.34+

``pip install quantax``


Tutorials
===========
The tutorials below guide you through the main features of Quantax.
You don't have to follow them one by one. Feel free to jump to the ones that interest you most.

.. toctree::
   :maxdepth: 1

   tutorials/0-quick_start.ipynb
   tutorials/1-exact_diag.ipynb
   tutorials/2-build_net.ipynb
   tutorials/3-samples.ipynb
   tutorials/4-fermion_mf.ipynb
   tutorials/5-J1J2_vmc.ipynb


Examples
==========
The examples below reproduce the main results of several important NQS papers.

.. toctree::
   :maxdepth: 1

   examples/1-RBM.ipynb
   examples/2-TDVP.ipynb
   examples/3-MinSR.ipynb
   examples/4-Backflow.ipynb

API
===

.. toctree::
   :maxdepth: 1

   global_defs
   sites/sites
   symmetry/symmetry
   operator/operator
   nn/nn
   model/model
   state/state
   sampler/sampler
   optimizer/optimizer
   utils/utils


Papers
======

.. toctree::
   :maxdepth: 1

   papers/packages
   papers/publications
