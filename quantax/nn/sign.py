import jax
import jax.numpy as jnp
from ..sites import TriangularB
from ..utils import neel, stripe
from ..global_defs import get_lattice


def compute_sign(
    kernel: jax.Array, s: jax.Array, output: str, neg: bool = False
) -> jax.Array:
    """
    Compute the sign, phase, or cosine value based on the provided kernel and spin configuration.
    
    :param kernel:
        The kernel array used for computation.

    :param s:
        The spin configuration array.

    :param output:
        The type of output to compute: "sign", "phase", or "cos".

    :param neg:
        Whether to negate the output.

    :return:
        The phase value obtained by ``jnp.dot(kernel.flatten(), s.flatten())``,
        transformed according to the specified output type.
    """
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
    """Marshall sign rule for bipartite lattices."""
    L = get_lattice().Nsites
    neg = (L // 4) % 2 == 1
    kernel = jnp.pi / 4 * neel().astype(jnp.float32)
    return compute_sign(kernel, s, "sign", neg)


def stripe_sign(s: jax.Array, alternate_dim: int = 1) -> jax.Array:
    """Stripe sign rule for bipartite lattices."""
    L = get_lattice().Nsites
    neg = (L // 4) % 2 == 1
    kernel = jnp.pi / 4 * stripe(alternate_dim).astype(jnp.float32)
    return compute_sign(kernel, s, "sign", neg)


def neel120_phase(s: jax.Array) -> jax.Array:
    """120 degree Neel phase for triangular lattices."""
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
