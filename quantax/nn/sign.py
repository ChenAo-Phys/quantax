import jax
import jax.numpy as jnp
from ..sites import TriangularB
from ..utils import neel, stripe
from ..global_defs import get_lattice


def compute_sign(
    kernel: jax.Array, s: jax.Array, output: str, neg: bool = False
) -> jax.Array:
    phase = jnp.dot(kernel.flatten(), s.flatten())

    if output == "sign":
        out = jnp.sign(jnp.cos(phase))
    elif output == "phase":
        out = jnp.exp(1j * phase)
    elif output == "cos":
        out = jnp.cos(phase)
    else:
        raise ValueError(f"Unknown output type: {output}")

    if neg:
        out = -out
    return out


def marshall_sign(s: jax.Array) -> jax.Array:
    L = get_lattice().Nsites
    neg = (L // 4) % 2 == 1
    kernel = jnp.pi / 4 * neel().astype(jnp.float32)
    return compute_sign(kernel, s, "sign", neg)


def stripe_sign(s: jax.Array, alternate_dim: int = 1) -> jax.Array:
    L = get_lattice().Nsites
    neg = (L // 4) % 2 == 1
    kernel = jnp.pi / 4 * stripe(alternate_dim).astype(jnp.float32)
    return compute_sign(kernel, s, "sign", neg)


def neel120_phase(s: jax.Array) -> jax.Array:
    lattice = get_lattice()
    Lx, Ly = lattice.shape[1:]
    x = 2 * jnp.arange(Lx)
    if isinstance(lattice, TriangularB):
        y = jnp.zeros(Ly, dtype=x.dtype)
    else:
        y = jnp.arange(Ly)
    kernel = (x[:, None] + y[None, :]) % 3
    kernel = jnp.pi / 3 * kernel - jnp.pi / 6
    kernel = kernel.astype(jnp.float32)
    return compute_sign(kernel, s, "phase")
