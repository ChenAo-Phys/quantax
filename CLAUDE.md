# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Quantax is a research-oriented JAX package for neural quantum states (NQS) and variational Monte Carlo (VMC) in quantum many-body physics. It also supports exact diagonalization (via QuSpin), fermionic mean-field wavefunctions, and tensor networks (via quimb and symmray). Requires Python 3.10–3.13 and JAX 0.6.1+.

Install for development: `pip install -e .[full]` from the repo root. The `[full]` extra pulls in QuSpin, matplotlib, jaxmg, and recent numpy/scipy.

## Tutorials and Examples

Tutorials live in [tutorials/](tutorials/) and more advanced reproductions live in [examples/](examples/). All are Jupyter notebooks — open them with Jupyter to run.

When running on a multi-GPU machine, check available devices first (e.g. `nvidia-smi`) and add the following at the **very top** of the notebook, before any `import jax` / `import quantax`:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '<gpu_id>'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'
```

This pins JAX to a single GPU and disables its default behavior of preallocating ~75% of device memory.

**If you intend to commit your edits back to the repo, delete these two lines before saving.** They are machine-specific and would otherwise be persisted into the notebook (and the rendered docs site).

## Documentation

- **Sources**: reStructuredText and notebooks under [docs/source/](docs/source/). The tutorial and example `.ipynb` files at [tutorials/](tutorials/) and [examples/](examples/) are pulled into the docs tree via MyST-NB and are rendered as-is (`nb_execution_mode = "off"` in [docs/source/conf.py](docs/source/conf.py)) — not re-executed at build time.
- **Build**: `cd docs && sphinx-build -b html ./source .` — output HTML is written directly into [docs/](docs/) (not `docs/build/`) so the same directory can serve as the GitHub Pages source.
- **Publish**: pushing to `main` automatically updates <https://chenao-phys.github.io/quantax/>. This works because GitHub Pages is configured (at <https://github.com/ChenAo-Phys/quantax/settings/pages>) to serve directly from the `/docs` folder on `main` — no GitHub Action is involved, so the built HTML must be committed alongside the sources.

## Test

Test files are in progress and will be added later. No test framework is currently configured.

## Format

- **Formatter**: [black](https://black.readthedocs.io/) — config in `[tool.black]` of [pyproject.toml](pyproject.toml) (only `target-version` is pinned; line length is the default 88). Run `black quantax/` before committing.
- **Type checker**: [pyright](https://microsoft.github.io/pyright/) — config in `[tool.pyright]` of [pyproject.toml](pyproject.toml) (`typeCheckingMode = "standard"`, `pythonVersion = "3.10"`, scoped to the `quantax` package). Run `pyright` from the repo root to check.

## Architecture

The package follows a layered pipeline. A typical VMC workflow composes objects from the modules below in this order (see [README.md](README.md) quick start):

```
sites (global) → operator → model → state → sampler → optimizer → loop
```

### Global state (important gotcha)

[quantax/global_defs.py](quantax/global_defs.py) holds three globals that other modules read:

- `Sites._SITES` — the single quantum system geometry/Hilbert space. **Only one `Sites` (or `Lattice`) instance is allowed per Python process.** Switching systems requires restarting the interpreter. Retrieved via `qtx.get_sites()` / `qtx.get_lattice()`.
- `DTYPE` — default float/complex dtype (default `float64`). `set_default_dtype` changes it and `get_default_dtype` reads it, but **models in [quantax/model/](quantax/model/) deliberately ignore this** and pick their own dtype (often float32) for efficiency.
- `KEY` — replicated JAX PRNG key. Use `get_subkeys(n)` to draw subkeys; this is not jittable because it mutates the global.

### Module map

- [quantax/global_defs.py](quantax/global_defs.py) — Modifiers and utility functions of global constants
- [quantax/sites/](quantax/sites/) — Geometry and particles of the quantum system
- [quantax/symmetry/](quantax/symmetry/) — Symmetry sector of the studied system
- [quantax/operator/](quantax/operator/) — Quantum operators on the Hilbert space
- [quantax/nn/](quantax/nn/) — Network components
- [quantax/model/](quantax/model/) — Variational wavefunctions
- [quantax/state/](quantax/state/) — Quantum states
- [quantax/sampler/](quantax/sampler/) — Samplers for generating configurations
- [quantax/optimizer/](quantax/optimizer/) — Optimizers for training wavefunctions
- [quantax/utils/](quantax/utils/) — Utility functions

## Conventions

- Spin configurations are represented as ±1 arrays internally; basis conversion to QuSpin's 0/1 convention happens in [quantax/utils/basis.py](quantax/utils/basis.py).
- Tutorials in [tutorials/](tutorials/) are arranged roughly in learning order (`quick_start` → `build_net` → `local_updates` → `fermion_mf` → `dynamics`); [examples/](examples/) holds more advanced cases (`RBM.ipynb`, `Backflow.ipynb`, `MinSR.ipynb`, `TDVP.ipynb`).
- Commit messages are short imperative phrases (e.g., "add norm_clip to SPRING and MARCH", "update QNGD"). No enforced format.
- JAX `float64` is enabled at `import quantax` ([quantax/global_defs.py:8](quantax/global_defs.py#L8)); to opt out, call `jax.config.update("jax_enable_x64", False)` *after* the import.
- On GPU, JAX defaults to TF32 matmul precision; for full float32, call `jax.config.update("jax_default_matmul_precision", "highest")`.
