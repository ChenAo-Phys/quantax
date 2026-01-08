.. Quantax documentation master file, created by
   sphinx-quickstart on Thu Aug  1 15:06:03 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the **Quantax** documentation!

**Quantax** is a Python package for flexible neural quantum states based on `JAX <https://github.com/google/jax>`_


Installation
============

Requires Python 3.10+, JAX 0.4.34+

First, ensure that a correct JAX version is installed.
For details, check `JAX Installation <https://docs.jax.dev/en/latest/installation.html>`_.

For a direct installation of full functionality (recommended in most cases),

``pip install quantax[full]``

For a minimal installation,

``pip install quantax``


Tutorials
===========
The tutorials below guide you through the main features of Quantax.
You don't have to follow them one by one. Feel free to jump to the ones that interest you most.

.. toctree::
   :maxdepth: 1

   tutorials/quick_start
   tutorials/exact_diag
   tutorials/build_net
   tutorials/samples
   tutorials/J1J2
   tutorials/triangular
   tutorials/fermion_mf
   tutorials/neural_jastrow
   tutorials/dynamics
   tutorials/local_updates


Examples
==========
The examples below reproduce the main results of several important NQS papers.

.. toctree::
   :maxdepth: 1

   examples/RBM
   examples/TDVP
   examples/MinSR
   examples/Backflow

API
===

.. list-table::
   :widths: 25 75

   * - :doc:`global_defs`
     - Modifiers and utility functions of global constants
   * - :doc:`sites/sites`
     - Geometry and particles of the quantum system
   * - :doc:`symmetry/symmetry`
     - Symmetry sector of the studied system
   * - :doc:`operator/operator`
     - Quantum operators on the Hilbert space
   * - :doc:`nn/nn`
     - Network components
   * - :doc:`model/model`
     - Variational wavefunctions
   * - :doc:`state/state`
     - Quantum states
   * - :doc:`sampler/sampler`
     - Samplers for generating configurations
   * - :doc:`optimizer/optimizer`
     - Optimizers for training wavefunctions
   * - :doc:`utils/utils`
     - Utility functions

.. toctree::
   :hidden:
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
