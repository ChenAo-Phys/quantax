<h1 align='center'>Quantax</h1>

**Flexible neural quantum states based on [JAX](https://github.com/google/jax)**

[📖 Documentation](https://chenao-phys.github.io/quantax)

## Not only NQS, but also

- Exact diagonalization (based on [QuSpin](https://github.com/QuSpin/QuSpin))
- Fermionic mean-field wavefunctions
- Flexible neural network design (based on [Equinox](https://github.com/patrick-kidger/equinox))

## Installation

Requires Python 3.10+, JAX 0.4.34+

First, ensure that a correct JAX version is installed.
For details, check [JAX Installation](https://docs.jax.dev/en/latest/installation.html).

For a direct installation of full functionality (recommended in most cases),
```
pip install quantax[full]
```

For a minimal installation,
```
pip install quantax
```


## Quick Start

```python
import quantax as qtx
import matplotlib.pyplot as plt

# Define a spin chain with 8 spins, stored as a global object in quantax
lattice = qtx.sites.Chain(L=8)

# Ising hamiltonian with transverse field h=1
H = qtx.operator.Ising(h=1)

# Exact diagonalization
E, wf = H.diagonalize()

# RBM wavefunction with 16 hidden units
model = qtx.model.RBM_Dense(features=16)

# Construct variational state
state = qtx.state.Variational(model)

# Sampler with local flip updates
sampler = qtx.sampler.LocalFlip(state, nsamples=64)

# Stochastic reconfiguration optimizer
optimizer = qtx.optimizer.SR(state, H)

energy_data = qtx.utils.DataTracer()
for i in range(100):
    samples = sampler.sweep()
    step = optimizer.get_step(samples)
    state.update(step * 1e-2)
    energy_data.append(optimizer.energy)

energy_data.plot(baseline=E)
plt.show()
```
