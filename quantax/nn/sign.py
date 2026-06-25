import jax
import jax.numpy as jnp
from ..sites import TriangularB
from ..utils import neel, stripe
from ..global_defs import get_lattice


def compute_sign(
    kernel: jax.Array,
    s: jax.Array,
    output: str,
    neg: bool = False,
    bias: jax.Array | float = 0.0,
) -> jax.Array:
    """
    Compute a sign, phase, or cosine structure from a linear form of the spin
    configuration, ``phase = jnp.dot(kernel.flatten(), s.flatten()) + bias``.

    :param kernel:
        The kernel array used for computation.

    :param s:
        The spin configuration array.

    :param output:
        The type of output to compute:

        - ``"sign"``: real ``jnp.sign(jnp.cos(phase))`` (``±1``, or ``0`` at a node);
        - ``"phase"``: complex ``jnp.exp(1j * phase)``;
        - ``"cos"``: real ``jnp.cos(phase)``.

    :param neg:
        Whether to negate the output.

    :param bias:
        A constant added to the exponent before the transform, used to encode a
        configuration-independent offset (e.g. the constant in a Marshall sign).

    :return:
        ``phase`` transformed according to the specified output type.
    """
    phase = jnp.dot(kernel.flatten(), s.flatten()) + bias

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


def _sublattice_sign(reference: jax.Array, s: jax.Array) -> jax.Array:
    r"""
    Sign rule :math:`(-1)^{n_\downarrow}` where :math:`n_\downarrow` counts the down
    spins on the sublattice marked by ``+1`` in ``reference`` (a ``±1`` pattern).

    The exponent ``phase = (pi/2) * (N_A - S_A) = pi * n_down`` is always an integer
    multiple of :math:`\pi`, so ``cos(phase)`` is exactly ``±1`` for any spin
    configuration and any sublattice size (no nodes).
    """
    sublattice_A = (reference.astype(jnp.float32) + 1) / 2  # 1 on A, 0 on B
    kernel = -jnp.pi / 2 * sublattice_A
    bias = jnp.pi / 2 * jnp.sum(sublattice_A)
    return compute_sign(kernel, s, "sign", bias=bias)


def marshall_sign(s: jax.Array) -> jax.Array:
    """Marshall sign rule for bipartite lattices."""
    return _sublattice_sign(neel(), s)


def stripe_sign(s: jax.Array, alternate_dim: int = 1) -> jax.Array:
    """Stripe sign rule for bipartite lattices."""
    return _sublattice_sign(stripe(alternate_dim), s)


def neel120_phase(s: jax.Array) -> jax.Array:
    """120 degree Neel phase for triangular lattices."""
    lattice = get_lattice()
    if lattice.shape[0] > 1 or len(lattice.shape) != 3:
        raise ValueError(
            "`neel120_phase` is only defined for 2D lattices with a single site "
            "per unit cell."
        )
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
