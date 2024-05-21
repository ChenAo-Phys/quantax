# quantax
Flexible neural quantum states based on [QuSpin](https://github.com/QuSpin/QuSpin/tree/dev_0.3.8), [JAX](https://github.com/google/jax), and [Equinox](https://github.com/patrick-kidger/equinox).

## Installation

(Many thanks to [Marin Bukov](https://github.com/mgbukov) for the help with QuSpin)

### Step 1 - Create a conda environment

#### Step 1.1 - Requirement file

Download [quspin-linux_reqs.txt](quspin-linux_reqs.txt)

#### Step 1.2 - Create environment

In the folder of `quspin-linux_reqs.txt`

`conda create -n quantax --file quspin-linux_reqs.txt`
(you are free to choose other environment names)

#### Step 1.3 - Activate environment

`conda activate quantax`

### Step 2 - Install QuSpin
Quantax relies on the dev_0.3.8 branch of QuSpin, which can't be easily installed
through pip or conda. Follow the instruction below for manual installation.

#### Step 2.1 - Clone the dev_0.3.8 branch of QuSpin

`cd ~` (or choose a folder where you want to clone the repository)

`git clone -b dev_0.3.8 https://github.com/QuSpin/QuSpin.git`

`cd QuSpin`

#### Step 2.2 - Install QuSpin

`python setup.py install build_ext -i --omp`

This will take some time and show many harmless warnings.

#### Step 2.3 - Possible problem

Try `import quspin` in a jupyter notebook. The manually installed QuSpin may raise circular import error. This problem can be solved by copying the quspin source code into `site-packages` of the conda environment and replacing the installed QuSpin files.

### Step 3 - Install Quantax

#### Step 3.1 - Install jax

Install JAX according to the [installation guide](https://jax.readthedocs.io/en/latest/installation.html)

#### Step 3.2 - Clone the repository

`cd ~` (or choose a folder where you want to clone the repository)

`git clone https://github.com/ChenAo-Phys/quantax.git`

`cd quantax`

#### Step 3.3 - Install Quantax

`pip install .`

### Supported platforms
- CPU
- Nvidia GPU
