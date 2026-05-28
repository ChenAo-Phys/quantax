from typing import Callable
import os
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from jax.sharding import NamedSharding, AxisType
from jax.lax import with_sharding_constraint
from jax.scipy.linalg import solve, eigh
from jax.scipy.sparse.linalg import cg
from ..nn import Sequential
from ..state import Variational
from ..utils import (
    array_extend,
    tree_fully_flatten,
    get_distributed_sharding,
    make_mesh,
)


def _to_dtype(arr: jax.Array, dtype: DTypeLike | None) -> jax.Array:
    if dtype is None:
        if jnp.iscomplexobj(arr):
            dtype = jnp.complex128
        else:
            dtype = jnp.float64

    return arr.astype(dtype)


def _get_rtol(dtype: DTypeLike) -> float:
    real_dtype = jnp.finfo(dtype).dtype
    if real_dtype == jnp.float64:
        rtol = 1e-12
    elif real_dtype == jnp.float32:
        rtol = 1e-6
    elif real_dtype == jnp.float16:
        rtol = 1e-3
    else:
        raise ValueError(f"Invalid dtype {dtype} for determining eigenvalue tolerance.")
    return rtol


class lstsq_shift_cg:
    def __init__(
        self,
        diag_shift: float = 0.01,
        rtol: float = 1e-5,
        atol: float = 0.0,
        maxiter: int | None = None,
    ):
        @jax.jit
        def S_apply(A, x):
            S_apply_x = jnp.einsum("sk,sl,l->k", A, A, x)
            S_apply_x += diag_shift * jnp.einsum("sk,sk,k->k", A, A, x)
            return S_apply_x

        self.S_apply = S_apply
        self.rtol = rtol
        self.atol = atol
        self.maxiter = maxiter

    def __call__(self, A: jax.Array, b: jax.Array) -> jax.Array:
        F = jnp.einsum("sk,s->k", A.conj(), b)
        Apply = lambda x: self.S_apply(A, x)
        x = cg(Apply, F, tol=self.rtol, atol=self.atol, maxiter=self.maxiter)
        return x[0]


def _diag_shift(A: jax.Array, rshift: float | None, ashift: float) -> jax.Array:
    n = A.shape[0]
    trace = jnp.linalg.trace(A).real
    if rshift is None:
        rshift = _get_rtol(trace.dtype)
    shift = rshift * trace / jnp.sqrt(A.shape[0]) + ashift
    A += shift * jnp.identity(n, A.dtype)
    return A


def minnorm_shift_eig(
    rshift: float | None = None,
    ashift: float = 1e-6,
    dtype: DTypeLike | None = None,
    *,
    jaxmg_ndevices: int = 1,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    if jaxmg_ndevices > 1:
        os.environ["JAXMG_NUMBER_OF_DEVICES"] = str(jaxmg_ndevices)

    @jax.jit
    def solution(A: jax.Array, b: jax.Array) -> jax.Array:
        input_dtype = A.dtype
        n, m = A.shape
        Adag = A.conj().T
        ndevices = jax.device_count()
        Adag = array_extend(Adag, ndevices)
        Adag = with_sharding_constraint(Adag, get_distributed_sharding())

        with jax.enable_x64():
            Adag = _to_dtype(Adag, dtype)
            b = _to_dtype(b, dtype)

            T = Adag.conj().T @ Adag
            T = _diag_shift(T, rshift, ashift)

            if jaxmg_ndevices > 1:
                from jaxmg import potrs

                shape = (jax.device_count() // jaxmg_ndevices, jaxmg_ndevices)
                mesh = jax.make_mesh(
                    shape, ("node", "device"), (AxisType.Auto, AxisType.Auto)
                )
                T = jax.device_put(T, NamedSharding(mesh, jax.P("device", None)))
                b = jax.device_put(b[:, None], NamedSharding(mesh, jax.P(None, None)))
                T_A = n // jaxmg_ndevices
                T_inv_b = potrs(T, b, T_A, mesh, in_specs=jax.P("device", None))
                T_inv_b = T_inv_b[:, 0]  # type: ignore
            else:
                T_inv_b = solve(
                    T, b, assume_a="pos"
                )  # cholesky solver is used internally

            x = (Adag @ T_inv_b).astype(input_dtype)

        return x[:m]

    return solution


def process_minnorm_shift_eig(
    rshift: float | None = None,
    ashift: float = 1e-6,
    dtype: DTypeLike | None = None,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    r"""
    Obtain a distributed MinSR solver for the linear equation :math:`Ax=b` using
    diagonal shift, see `~quantax.optimizer.minnorm_shift_eig`.

    The inputs ``A`` and ``b`` are assumed to be sharded across all devices along their
    first axis. Instead of solving one global system, each JAX process builds its own
    :math:`T = A A^†` from the data held by its local devices and solves
    :math:`x = A^† (A A^†)^{-1} b` independently. The per-process solutions are then
    averaged into the returned ``x``. This avoids the expensive inter-process
    communication of assembling a single global :math:`T`, at the cost of approximating
    the global solution by the average of per-process solutions.

    Within each process the heavy linear algebra is still sharded across the local
    devices along the parameter axis, exactly like `~quantax.optimizer.minnorm_shift_eig`.
    With a single process the two solvers are therefore equivalent, both in result and in
    cost.

    The diagonal shift modifies the per-process :math:`T = A A^†` to
    :math:`T' = T + \epsilon I` for stable inversion, with
    :math:`\epsilon = \mathrm{Tr}(T) \times \mathrm{rshift} + \mathrm{ashift}`,
    where rshift and ashift are adjustable arguments.

    :param rshift:
        The relative diagonal shift. Default to be :math:`10^{-12}` for double precision
        and :math:`10^{-6}` for single precision.

    :param ashift:
        The absolute diagonal shift, default to 1e-6.

    :param dtype:
        The dtype used internally in the solver. By default, real-valued inputs use float64
        and complex-valued inputs use complex128.

    :return:
        A solver function with two arguments A and b and one output x as the solution of
        :math:`A x = b`.
    """

    @jax.jit
    def solution(A: jax.Array, b: jax.Array) -> jax.Array:
        input_dtype = A.dtype
        n, m = A.shape
        mesh = make_mesh()
        nprocess = mesh.shape["process"]
        ndevices = mesh.shape["device"]

        # Group the samples by process and shard the parameter axis over the local
        # devices, so each process forms and factorizes its own T independently while
        # the heavy matmul is split across that process's devices.
        A = array_extend(A, ndevices, axis=1)
        A = A.reshape(nprocess, n // nprocess, A.shape[1])
        b = b.reshape(nprocess, n // nprocess)
        sharding = NamedSharding(mesh, jax.P("process", None, "device"))
        A = with_sharding_constraint(A, sharding)

        with jax.enable_x64():
            A = _to_dtype(A, dtype)
            b = _to_dtype(b, dtype)

            T = jnp.einsum("pik,pjk->pij", A, A.conj())  # per-process T = A A^†
            T = jax.vmap(lambda Tp: _diag_shift(Tp, rshift, ashift))(T)
            # cholesky solver is used internally
            y = jax.vmap(lambda Tp, bp: solve(Tp, bp, assume_a="pos"))(T, b)
            x = jnp.einsum("pik,pi->pk", A.conj(), y)  # per-process x = A^† y
            x = jnp.mean(x, axis=0).astype(input_dtype)  # average over processes

        return x[:m]

    return solution


def lstsq_shift_eig(
    rshift: float | None = None,
    ashift: float = 1e-6,
    dtype: DTypeLike | None = None,
    *,
    jaxmg_ndevices: int = 1,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    if jaxmg_ndevices > 1:
        os.environ["JAXMG_NUMBER_OF_DEVICES"] = str(jaxmg_ndevices)

    @jax.jit
    def solution(A: jax.Array, b: jax.Array) -> jax.Array:
        input_dtype = A.dtype

        with jax.enable_x64():
            A = _to_dtype(A, dtype)
            b = _to_dtype(b, dtype)
            S = A.conj().T @ A
            F = A.conj().T @ b
            S = _diag_shift(S, rshift, ashift)

            if jaxmg_ndevices > 1:
                from jaxmg import potrs

                shape = (jax.device_count() // jaxmg_ndevices, jaxmg_ndevices)
                mesh = jax.make_mesh(
                    shape, ("node", "device"), (AxisType.Auto, AxisType.Auto)
                )
                S = jax.device_put(S, NamedSharding(mesh, jax.P("device", None)))
                F = jax.device_put(F[:, None], NamedSharding(mesh, jax.P(None, None)))
                T_A = S.shape[0] // jaxmg_ndevices
                x = potrs(S, F, T_A, mesh, in_specs=jax.P("device", None))
                x = x[:, 0]  # type: ignore
            else:
                x = solve(S, F, assume_a="pos")  # cholesky solver is used internally
            x = x.astype(input_dtype)

        return x

    return solution


def auto_shift_eig(
    rshift: float | None = None,
    ashift: float = 1e-6,
    dtype: DTypeLike | None = None,
    *,
    jaxmg_ndevices: int = 1,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    r"""
    Obtain the least-square minimum-norm solver for the linear equation
    :math:`Ax=b` using diagonal shift. It automatically chooses between
    :math:`x = (A^† A)^{-1} A^† b` and :math:`x = A^† (A A^†)^{-1} b`, which respectively
    correspond to SR and MinSR.

    Given :math:`M = A^† A` or :math:`M = A A^†`, the diagonal shift modifies it to
    :math:`M' = M + \epsilon I` for stable inversion.
    :math:`\epsilon = \mathrm{Tr}(M) \times \mathrm{rshift} + \mathrm{ashift},
    where rshift and ashift are adjustable arguments.

    :param rtol:
        The relative tolerance for pseudo-inverse. Default to be :math:`10^{-12}` for
        double precision and :math:`10^{-6}` for single precision.

    :param atol:
        The absolute tolerance for pseudo-inverse, default to 1e-6.

    :param dtype:
        The dtype used internally in the solver. By default, real-valued inputs use float64
        and complex-valued inputs use complex128.

    :param jaxmg_ndevices:
        The number of devices to use with `jaxmg <https://github.com/flatironinstitute/jaxmg>`_
        for distributed linear algebra. By default it is set to 1, which means not using
        `jaxmg`. Setting it to the number of devices per node will enable `jaxmg`.
        This option is often used for large-scale problems where the matrix is too large
        to fit in memory on a single device. It requires `jaxmg` to be installed and
        properly configured.

    :return:
        A solver function with two arguments A and b and one output x as the solution of
        :math:`A x = b`.
    """
    minnorm_solver = minnorm_shift_eig(
        rshift, ashift, dtype, jaxmg_ndevices=jaxmg_ndevices
    )
    lstsq_solver = lstsq_shift_eig(rshift, ashift, dtype, jaxmg_ndevices=jaxmg_ndevices)

    @jax.jit
    def solve(A: jax.Array, b: jax.Array) -> jax.Array:
        if A.shape[0] < A.shape[1]:
            return minnorm_solver(A, b)
        else:
            return lstsq_solver(A, b)

    return solve


@jax.jit
def _get_eigs_inv(vals: jax.Array, rtol: float | None, atol: float) -> jax.Array:
    vals_abs = jnp.abs(vals)
    if rtol is None:
        rtol = _get_rtol(vals_abs.dtype)
    inv_factor = 1 + ((rtol * jnp.max(vals_abs) + atol) / vals_abs) ** 6
    eigs_inv = 1 / (vals * inv_factor)
    return jnp.where(vals_abs > 0.0, eigs_inv, 0.0)


def pinvh_solve(
    rtol: float | None = None, atol: float = 0.0
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    @jax.jit
    def solve(H: jax.Array, b: jax.Array) -> jax.Array:
        eig_vals, U = eigh(H)
        eig_inv = _get_eigs_inv(eig_vals, rtol, atol)
        return jnp.einsum("rs,s,ts,t->r", U, eig_inv, U.conj(), b)

    return solve


@jax.jit
def _sum_without_noise(inputs: jax.Array, tol_snr: float) -> jax.Array:
    """
    Noise truncation, see https://arxiv.org/pdf/2108.03409.pdf
    """
    x = jnp.sum(inputs, axis=0)
    x_mean = x / inputs.shape[0]
    x_var = jnp.abs(inputs - x_mean[None, :]) ** 2
    x_var = jnp.sqrt(jnp.mean(x_var, axis=0) / inputs.shape[0])
    snr = jnp.abs(x_mean) / x_var
    x = jnp.where(tol_snr > 1e-6, x / (1 + (tol_snr / snr) ** 6), x)
    return x


def minnorm_pinv_eig(
    rtol: float | None = None,
    atol: float = 0.0,
    tol_snr: float = 0.0,
    dtype: DTypeLike | None = None,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    @jax.jit
    def solve(A: jax.Array, b: jax.Array) -> jax.Array:
        input_dtype = A.dtype
        n, m = A.shape
        Adag = A.conj().T
        ndevices = jax.device_count()
        Adag = array_extend(Adag, ndevices)
        Adag = with_sharding_constraint(Adag, get_distributed_sharding())

        with jax.enable_x64():
            Adag = _to_dtype(Adag, dtype)
            b = _to_dtype(b, dtype)

            T = Adag.conj().T @ Adag
            # T_inv_b = pinv_solve(T, b, tol, atol, tol_snr)
            # x = jnp.einsum("rk,r->k", A.conj(), T_inv_b)
            eig_vals, U = eigh(T)
            eig_inv = _get_eigs_inv(eig_vals, rtol, atol)
            rho_ts = jnp.einsum("ts,t->ts", U.conj(), b)
            rho = _sum_without_noise(rho_ts, tol_snr)
            x = jnp.einsum("kr,rs,s,s->k", Adag, U, eig_inv, rho)
            x = x.astype(input_dtype)

        return x[:m]

    return solve


def lstsq_pinv_eig(
    rtol: float | None = None,
    atol: float = 0.0,
    tol_snr: float = 0.0,
    dtype: DTypeLike | None = None,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    @jax.jit
    def solve(A: jax.Array, b: jax.Array) -> jax.Array:
        input_dtype = A.dtype

        with jax.enable_x64():
            A = _to_dtype(A, dtype)
            b = _to_dtype(b, dtype)
            S = A.conj().T @ A
            eig_vals, V = eigh(S)
            eig_inv = _get_eigs_inv(eig_vals, rtol, atol)
            rho_sk = jnp.einsum("lk,sl,s->sk", V.conj(), A.conj(), b)
            rho = _sum_without_noise(rho_sk, tol_snr)
            x = jnp.einsum("kl,l,l->k", V, eig_inv, rho)
            x = x.astype(input_dtype)

        return x

    return solve


def auto_pinv_eig(
    rtol: float | None = None,
    atol: float = 0.0,
    tol_snr: float = 0.0,
    dtype: DTypeLike | None = None,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """
    Obtain the least-square minimum-norm solver for the linear equation
    :math:`Ax=b` using pseudo-inverse. It automatically chooses between
    :math:`x = (A^† A)^{-1} A^† b` and :math:`x = A^† (A A^†)^{-1} b`, which respectively
    correspond to SR and MinSR.

    :param rtol:
        The relative tolerance for pseudo-inverse. Default to be :math:`10^{-12}` for
        double precision and :math:`10^{-6}` for single precision.

    :param atol:
        The absolute tolerance for pseudo-inverse, default to 0.

    :param tol_snr:
        The tolerence of signal-to-noise ratio (SNR), default to 0 which means no regularization
        based on SNR. For details see `Phys. Rev. Lett. 125, 100503 <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.100503>`_.

    :param dtype:
        The dtype used internally in the solver. By default, real-valued inputs use float64
        and complex-valued inputs use complex128.

    :return:
        A solver function with two arguments A and b and one output x as the solution of
        :math:`A x = b`.
    """
    minnorm_solver = minnorm_pinv_eig(rtol, atol, tol_snr, dtype)
    lstsq_solver = lstsq_pinv_eig(rtol, atol, tol_snr, dtype)

    @jax.jit
    def solve(A: jax.Array, b: jax.Array) -> jax.Array:
        if A.shape[0] < A.shape[1]:
            return minnorm_solver(A, b)
        else:
            return lstsq_solver(A, b)

    return solve


def block_pinv_eig(
    state: Variational,
    rtol: float | None = None,
    atol: float = 0.0,
    tol_snr: float = 0.0,
    dtype: DTypeLike | None = None,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """
    Obtain the layerwise least-square minimum-norm solver for the linear equation
    :math:`Ax=b` using pseudo-inverse. See `LayerSR <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.108.054410>`_
    or `block QGT <https://arxiv.org/abs/2510.08430>`_ for details.

    It automatically chooses between
    :math:`x = (A^† A)^{-1} A^† b` and :math:`x = A^† (A A^†)^{-1} b`, which respectively
    correspond to SR and MinSR.

    :param state:
        The variational state, used to identify the parameter blocks.

    :param rtol:
        The relative tolerance for pseudo-inverse. Default to be :math:`10^{-12}` for
        double precision and :math:`10^{-6}` for single precision.

    :param atol:
        The absolute tolerance for pseudo-inverse, default to 0.

    :param tol_snr:
        The tolerence of signal-to-noise ratio (SNR), default to 0 which means no regularization
        based on SNR. For details see `Phys. Rev. Lett. 125, 100503 <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.100503>`_.

    :param dtype:
        The dtype used internally in the solver. By default, real-valued inputs use float64
        and complex-valued inputs use complex128.

    :return:
        A solver function with two arguments A and b and one output x as the solution of
        :math:`A x = b`.
    """
    if not isinstance(state.model, Sequential):
        raise ValueError("`block_pinv_eig` solver only works for `Sequential` models.")

    Np_layer = []
    idx = 0
    params, static = state.partition()
    for p in params.layers:
        p = tree_fully_flatten(p)
        if p.size > 0:
            idx += p.size
            Np_layer.append(idx)

    nlayers = len(Np_layer)
    Np_layer = Np_layer[:-1]
    solver0 = auto_pinv_eig(rtol, atol, tol_snr, dtype)

    @jax.jit
    def solve(Obar: jax.Array, Ebar: jax.Array) -> jax.Array:
        Obar_list = jnp.split(Obar, Np_layer, axis=1)
        Ebar /= nlayers
        return jnp.concatenate([solver0(Oi, Ebar) for Oi in Obar_list], axis=0)

    return solve


def minsr_pinv_eig(
    rtol: float | None = None, atol: float = 0.0, tol_snr: float = 0.0
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """
    Obtain the pseudo-inverse solver for the inverse problem in MinSR
    :math:`Tx=b`, where :math:`T` is a Hermitian matrix.

    :param rtol:
        The relative tolerance for pseudo-inverse. Default to be :math:`10^{-12}` for
        double precision and :math:`10^{-6}` for single precision.

    :param atol:
        The absolute tolerance for pseudo-inverse, default to 0.

    :param tol_snr:
        The tolerence of signal-to-noise ratio (SNR), default to 0 which means no regularization
        based on SNR. For details see `Phys. Rev. Lett. 125, 100503 <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.100503>`_.

    :return:
        A solver function with two arguments T and b and one output x as the solution of
        :math:`T x = b`.
    """

    @jax.jit
    def solve(T: jax.Array, b: jax.Array) -> jax.Array:
        eig_vals, U = eigh(T)
        eig_inv = _get_eigs_inv(eig_vals, rtol, atol)
        rho_ts = jnp.einsum("ts,t->ts", U.conj(), b)
        rho = _sum_without_noise(rho_ts, tol_snr)
        x = jnp.einsum("rs,s,s->r", U, eig_inv, rho)
        return x

    return solve


def sgd_solver() -> Callable[[jax.Array, jax.Array], jax.Array]:
    @jax.jit
    def solve(A: jax.Array, b: jax.Array) -> jax.Array:
        return jnp.einsum("sk,s->k", A.conj(), b) / b.shape[0]

    return solve
