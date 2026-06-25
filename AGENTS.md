# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Project

Quantax is a research-oriented JAX package for neural quantum states (NQS) and variational Monte Carlo (VMC) in quantum many-body physics. It also supports exact diagonalization (via QuSpin), fermionic mean-field wavefunctions, and tensor networks (via quimb and symmray). Requires Python 3.11–3.13 and JAX 0.8.0+.

Install for development: `pip install -e .[full]` from the repo root. The `[full]` extra pulls optional dependencies.

## Tutorials and Examples

Tutorials live in [tutorials/](tutorials/) and more advanced reproductions of NQS papers live in [examples/](examples/).

When running on a multi-GPU machine, check available devices first (e.g. `nvidia-smi`) and add the following at the **very top** of the notebook, before any `import jax` / `import quantax`:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '<gpu_id>'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'
```

This pins JAX to a single GPU and disables its default behavior of preallocating ~75% of device memory.

**If you intend to commit your edits back to the repo, delete these two lines before saving.** They are machine-specific and would otherwise be persisted into the notebook (and the rendered docs site).

### Sharp bits

Some caveats of Quantax are listed in [sharp_bits](tutorials/sharp_bits.ipynb). Currently, it includes

- Compatible devices
- Global system
- Randomness
- Data precision
- to JIT or not to JIT
- wavefunction overflow
- sharding
- HPC usage

Read through sharp bits carefully when the related code is being editted.

## Documentation

- **Sources**: reStructuredText and notebooks under [docs/source/](docs/source/). The tutorial and example `.ipynb` files at [tutorials/](tutorials/) and [examples/](examples/) are pulled into the docs tree via MyST-NB and are rendered as-is (`nb_execution_mode = "off"` in [docs/source/conf.py](docs/source/conf.py)) — not re-executed at build time.
- **Build**: the convenient way is `cd docs && make html` (equivalent to `sphinx-build -b html docs/source docs/build/html`). Output HTML is written to `docs/build/html`, which is git-ignored. Requires the docs toolchain: `pip install -e .[docs]`.
- **Publish**: pushing to `main` triggers [.github/workflows/docs.yml](.github/workflows/docs.yml), which builds the docs and deploys them to <https://chenao-phys.github.io/quantax/> via GitHub Pages. GitHub Pages must be configured (at <https://github.com/ChenAo-Phys/quantax/settings/pages>) with source **"GitHub Actions"** — the built HTML is no longer committed; only the sources under [docs/source/](docs/source/) are tracked.

## Test

- **Framework**: [pytest](https://docs.pytest.org/) (install separately: `pip install pytest`). Tests live in [tests/](tests/), mirroring the package layout (`tests/sites/`, `tests/operator/`, `tests/nn/`, `tests/model/`, `tests/state/`, `tests/sampler/`, `tests/symmetry/`, `tests/utils/`).
- **Run**: `pytest tests/ -q` from the repo root. Tests are designed to run on CPU; set `JAX_PLATFORMS=cpu`, and set `XLA_FLAGS="--xla_force_host_platform_device_count=4"` to emulate 4 devices for sharding-related tests.
- **Global state**: an autouse fixture in [tests/conftest.py](tests/conftest.py) resets quantax's global state (random seed, default dtype, `Sites._SITES`, and the memoized symmetry singletons) before and after each test, so each test can define its own lattice despite the one-`Sites`-per-process rule.
- **CI**: pushes and PRs to `main`/`dev` trigger [.github/workflows/tests.yml](.github/workflows/tests.yml), which runs the suite on Python 3.11–3.13 with the CPU settings above.

## Format

- **Formatter**: [black](https://black.readthedocs.io/) — config in `[tool.black]` of [pyproject.toml](pyproject.toml) (only `target-version` is pinned; line length is the default 88). Run `black quantax/` before committing.
- **Type checker**: [pyright](https://microsoft.github.io/pyright/) — config in `[tool.pyright]` of [pyproject.toml](pyproject.toml) (`typeCheckingMode = "standard"`, `pythonVersion = "3.12"`, scoped to the `quantax` package). Run `pyright` from the repo root to check.

## Architecture

The package follows a layered pipeline. A typical VMC workflow composes objects from the modules below in this order (see [quick start](tutorials/quick_start.ipynb)):

```
sites (global) → operator → model → state → sampler → optimizer → loop
```

### Global state (important gotcha)

[quantax/global_defs.py](quantax/global_defs.py) holds three globals that other modules read:

- `Sites._SITES` — the single quantum system geometry/Hilbert space. **Only one `Sites` (or `Lattice`) instance is allowed per Python process.** Switching systems requires restarting the interpreter. Retrieved via `qtx.get_sites()` / `qtx.get_lattice()`.
- `DTYPE` — default float/complex dtype (default `float32`). `set_default_dtype` changes it and `get_default_dtype` reads it, but **models in [quantax/model/](quantax/model/) deliberately ignore this** and pick their own dtype (often float32) for efficiency.
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
- JAX `float64` is *not* enabled by default; to enable it, call `jax.config.update("jax_enable_x64", True)`. To use it as default dtype in Quantax, call `quantax.set_default_dtype(jnp.float64)`.
- On GPU, JAX defaults to TF32 matmul precision; for full float32, call `jax.config.update("jax_default_matmul_precision", "highest")`.
- Keep the repository clean. Test files should be saved in other folders.

## Coding principles

### Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

**Brief comments that explain the purpose.**

- Consider the necessity of comments before start writing.
- If the code shows clear purpose, avoid comments. 
- Keep comments brief.
- If multi-line comments are necessary, consider putting them in docstring.

### Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

