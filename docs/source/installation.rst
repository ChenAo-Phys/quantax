Installation
============

Step 1 - Create a conda environment
---------------------------------------

*Step 1.1* -  
Download quspin-linux_reqs.txt

*Step 1.2* - 
In the folder of ``quspin-linux_reqs.txt``, run command 
``conda create -n quantax --file quspin-linux_reqs.txt``
(you are free to choose other environment names)

*Step 1.3* - 
``conda activate quantax``

Step 2 - Install QuSpin
---------------------------

Quantax relies on the dev_0.3.8 branch of QuSpin, which can't be easily installed
through pip or conda. Follow the instruction below for manual installation.

*Step 2.1* - 
Choose a folder where you want to clone the repository, and run command
``git clone -b dev_0.3.8 https://github.com/QuSpin/QuSpin.git``, and ``cd QuSpin``

*Step 2.2* -
Run command ``python setup.py install build_ext -i --omp``.
this will take some time and show many harmless warnings.

*Step 2.3* -
Try ``import quspin`` in a the created python environment. 
The manually installed QuSpin may raise circular import error. 
This problem can be solved by copying the quspin source code into ``site-packages`` 
of the conda environment and replacing the installed QuSpin files.

Step 3 - Install Quantax
---------------------------

*Step 3.1* -
Install JAX according to the `installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_

*Step 3.2* -
Choose a folder where you want to clone the repository, and run command
``git clone https://github.com/ChenAo-Phys/quantax.git``, and ``cd quantax``

*Step 3.3* - 
Install Quantax by ``pip install .``


Supported platform
------------------
CPU, NVIDIA GPU

Currently Quantax only supports single-node parallelization.