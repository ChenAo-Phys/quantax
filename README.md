# quantax
Flexible neural quantum states based on [QuSpin](https://github.com/QuSpin/QuSpin/tree/dev_0.3.8), [JAX](https://github.com/google/jax), and [Equinox](https://github.com/patrick-kidger/equinox).
Check [here](https://chenao-phys.github.io/) for documentation.

## Installation

### Step 1 - Install JAX

This step is necessary only if you need to run codes on GPU.

`pip install -U "jax[cuda12]"`

Check [JAX installation](https://jax.readthedocs.io/en/latest/installation.html) for
other installation options.

### Step 2 - Install Quantax

```
git clone https://github.com/ChenAo-Phys/quantax.git
pip install ./quantax
```

## Supported platforms
- CPU
- NVIDIA GPU
