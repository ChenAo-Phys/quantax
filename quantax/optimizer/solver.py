from typing import Callable
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from jax.lax import with_sharding_constraint
from jax.scipy.linalg import solve, eigh
from jax.scipy.sparse.linalg import cg as jax_cg
from ..nn import Sequential
from ..state import Variational
from ..utils import (
    array_extend,
    tree_fully_flatten,
    get_distributed_sharding,
    get_replicated_sharding,
    precision_context,
)


def _to_dtype(arr: jax.Array, dtype: DTypeLike | None) -> jax.Array:
    if dtype is None:
        return arr
    if jnp.iscomplexobj(arr):
        dtype = jnp.promote_types(dtype, jnp.complex64)  # keep the imaginary part

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


def _shift(A: jax.Array, rshift: float | None, ashift: float) -> jax.Array:
    # Tr(A^† A) * rshift / sqrt(n) + ashift, with Tr(A^† A) = Tr(A A^†) = sum |A|^2
    # obtained without forming A^† A or A A^†, and n the smaller dimension of A
    n = min(A.shape)
    trace = jnp.sum(jnp.abs(A) ** 2)
    if rshift is None:
        rshift = _get_rtol(trace.dtype)
    return rshift * trace / jnp.sqrt(n) + ashift


# =================================================================================
#                                Iterative solvers
# =================================================================================


def lstsq_cg(
    rshift: float = 1e-4,
    ashift: float = 0.0,
    rtol: float = 1e-2,
    atol: float = 0.0,
    maxiter: int = 500,
    dtype: DTypeLike | None = None,
) -> Callable[..., jax.Array]:
    r"""
    Obtain the least-square solver for the linear equation :math:`Ax=b` using
    diagonal shift, corresponding to SR. The solution
    :math:`x = (A^† A + \epsilon I)^{-1} A^† b` is obtained by the conjugate gradient
    method `jax.scipy.sparse.linalg.cg
    <https://docs.jax.dev/en/latest/_autosummary/jax.scipy.sparse.linalg.cg.html>`_,
    applied to the shifted normal equation :math:`(A^† A + \epsilon I) x = A^† b`
    matrix-free, without forming :math:`A^† A`. This is suitable for overdetermined
    problems where the number of samples exceeds the number of parameters.

    The solution is further regularized by the finite number of CG iterations: the
    iteration first resolves the dominant eigen-directions of :math:`A^† A`, and the
    small shift keeps it from amplifying the small, noise-dominated ones.

    :param rshift:
        The relative diagonal shift, entering the shift as
        :math:`\epsilon = \mathrm{Tr}(A^† A) \times \mathrm{rshift} / \sqrt{d} + \mathrm{ashift}`,
        with :math:`d` the smaller one of the numbers of rows and columns of :math:`A`,
        i.e. the number of parameters for an overdetermined problem. Default to 1e-4.

    :param ashift:
        The absolute diagonal shift, default to 0.

    :param rtol:
        The relative tolerance for terminating the CG iteration.

    :param atol:
        The absolute tolerance for terminating the CG iteration.

    :param maxiter:
        The maximum number of CG iterations.

    :param dtype:
        The precision used internally in the CG iteration. By default (``None``) the
        dtype of the inputs is kept. Complex inputs use the complex counterpart, e.g.
        complex64 for float32. Running CG on the normal-equation operator :math:`A^† A`
        squares the condition number, so pass ``dtype=jnp.float64`` if the
        single-precision iteration loses conjugacy and diverges for an ill-conditioned
        :math:`A^† A`.

    :return:
        A solver function with two arguments A and b and one output x as the solution of
        :math:`A x = b`.
    """

    @jax.jit
    def solution(A: jax.Array, b: jax.Array, **kwargs) -> jax.Array:
        input_dtype = A.dtype

        # x64 follows the input dtype by default, as in `lsmr`
        with precision_context(input_dtype if dtype is None else dtype):
            A = _to_dtype(A, dtype)
            b = _to_dtype(b, dtype)

            F = jnp.einsum("sk,s->k", A.conj(), b, precision="highest")
            shift = _shift(A, rshift, ashift)

            def S_apply(x):
                S_x = jnp.einsum("sk,sl,l->k", A.conj(), A, x, precision="highest")
                return S_x + shift * x

            x, _ = jax_cg(S_apply, F, tol=rtol, atol=atol, maxiter=maxiter)
        return x.astype(input_dtype)

    return solution


def minnorm_cg(
    rshift: float = 1e-4,
    ashift: float = 0.0,
    rtol: float = 1e-2,
    atol: float = 0.0,
    maxiter: int = 500,
    dtype: DTypeLike | None = None,
) -> Callable[..., jax.Array]:
    r"""
    Obtain the minimum-norm solver for the linear equation :math:`Ax=b` using
    diagonal shift, corresponding to MinSR. The solution
    :math:`x = A^† (A A^† + \epsilon I)^{-1} b` is obtained by solving the auxiliary
    system :math:`(A A^† + \epsilon I) y = b` with the conjugate gradient method
    `jax.scipy.sparse.linalg.cg
    <https://docs.jax.dev/en/latest/_autosummary/jax.scipy.sparse.linalg.cg.html>`_
    matrix-free, without forming :math:`A A^†`, followed by :math:`x = A^† y`. This is
    suitable for underdetermined problems where the number of parameters exceeds the
    number of samples.

    The shift is necessary for the rank-deficient :math:`A A^†` of a VMC Jacobian:
    without it, the iteration amplifies the small, noise-dominated eigen-directions
    from the first steps, and the residual tolerance is never reached.

    :param rshift:
        The relative diagonal shift, entering the shift as
        :math:`\epsilon = \mathrm{Tr}(A A^†) \times \mathrm{rshift} / \sqrt{d} + \mathrm{ashift}`,
        with :math:`d` the smaller one of the numbers of rows and columns of :math:`A`,
        i.e. the number of samples for an underdetermined problem, or twice that for
        non-holomorphic states. Default to 1e-4.

    :param ashift:
        The absolute diagonal shift, default to 0.

    :param rtol:
        The relative tolerance for terminating the CG iteration.

    :param atol:
        The absolute tolerance for terminating the CG iteration.

    :param maxiter:
        The maximum number of CG iterations.

    :param dtype:
        The precision used internally in the CG iteration. By default (``None``) the
        dtype of the inputs is kept. Complex inputs use the complex counterpart, e.g.
        complex64 for float32. Running CG on the normal-equation operator :math:`A A^†`
        squares the condition number, so pass ``dtype=jnp.float64`` if the
        single-precision iteration loses conjugacy and diverges for an ill-conditioned
        :math:`A A^†`.

    :return:
        A solver function with two arguments A and b and one output x as the solution of
        :math:`A x = b`.
    """

    @jax.jit
    def solution(A: jax.Array, b: jax.Array, **kwargs) -> jax.Array:
        input_dtype = A.dtype

        # x64 follows the input dtype by default, as in `lsmr`
        with precision_context(input_dtype if dtype is None else dtype):
            A = _to_dtype(A, dtype)
            b = _to_dtype(b, dtype)

            shift = _shift(A, rshift, ashift)

            def T_apply(y):
                T_y = jnp.einsum("sk,tk,t->s", A, A.conj(), y, precision="highest")
                return T_y + shift * y

            y, _ = jax_cg(T_apply, b, tol=rtol, atol=atol, maxiter=maxiter)
            x = jnp.einsum("sk,s->k", A.conj(), y, precision="highest")
        return x.astype(input_dtype)

    return solution


def lsmr(
    rtol: float = 0.1,
    atol: float = 0.0,
    maxiter: int = 500,
    dtype: DTypeLike | None = None,
) -> Callable[..., jax.Array]:
    r"""
    Obtain the least-square solver for the linear equation :math:`Ax=b` solved by
    the LSMR method
    `lineax.LSMR <https://docs.kidger.site/lineax/api/solvers/#lineax.LSMR>`_.

    LSMR is applied to the pure least-square problem :math:`\min_x \|Ax - b\|^2`
    without a diagonal shift. The solution is regularized by early stopping
    instead: the Krylov space is built from :math:`b`, so the iteration only
    explores the dominant singular directions that :math:`b` excites and never
    amplifies the small, noise-dominated directions before the residual tolerance
    is reached. Working on :math:`A` directly rather than :math:`A^† A` keeps the
    effective condition number squared smaller, which is numerically more stable
    than `~quantax.optimizer.lstsq_cg`, especially in single precision.
    This solver gives the same solution regardless of whether the problem is
    over- or under-determined.

    :param rtol:
        The relative tolerance for terminating the LSMR iteration, default to 0.1.
        It is divided by the number of samples :math:`n`, i.e. the number of rows of
        :math:`A`, before being passed to ``lineax.LSMR``, so the effective tolerance
        :math:`\mathrm{rtol}/n` tightens as the sample size grows. As the solution is
        regularized by early stopping, ``rtol`` also controls the strength of the
        regularization.

        .. warning::
            In single precision the internal residual estimate of LSMR is
            optimistic: the true relative residual :math:`\|Ax-b\| / \|b\|` is
            typically about two orders of magnitude larger than the effective
            tolerance :math:`\mathrm{rtol}/n` at termination. Choose ``rtol``
            accordingly, or verify the residual with one extra matrix-vector product
            after solving.

    :param atol:
        The absolute tolerance for terminating the LSMR iteration, default to 0.
        Like ``rtol``, it is divided by the number of samples :math:`n`.

    :param maxiter:
        The maximum number of LSMR iterations, default to 500.

    :param dtype:
        The precision used internally in the LSMR iteration. By default (``None``) the
        dtype of the inputs is kept. Complex inputs use the complex counterpart, e.g.
        complex64 for float32. Pass ``dtype=jnp.float64`` for a more accurate solution
        at a higher cost.

    :return:
        A solver function with two arguments A and b and one output x as the solution of
        :math:`A x = b`.
    """
    import lineax as lx

    @jax.jit
    def solution(
        A: jax.Array,
        b: jax.Array,
        *,
        diag_preconditioner: jax.Array | None = None,
        **kwargs,
    ) -> jax.Array:
        input_dtype = A.dtype
        options = {}
        if diag_preconditioner is not None:
            diag_preconditioner = with_sharding_constraint(
                diag_preconditioner, get_replicated_sharding()
            )
        n, m = A.shape

        # lineax promotes python scalars to float64 under x64 and fails for
        # single-precision inputs, so x64 follows the input dtype by default
        with precision_context(input_dtype if dtype is None else dtype):
            A = _to_dtype(A, dtype)
            b = _to_dtype(b, dtype)
            if diag_preconditioner is not None:
                diag_preconditioner = _to_dtype(diag_preconditioner, dtype)

            def M_apply(x):
                if diag_preconditioner is not None:
                    x /= diag_preconditioner
                Ax = jnp.einsum("sk,k->s", A, x, precision="highest")
                return Ax

            operator = lx.FunctionLinearOperator(
                M_apply, jax.ShapeDtypeStruct((m,), A.dtype)
            )
            solver = lx.LSMR(rtol=rtol / n, atol=atol / n, max_steps=maxiter)
            # Call LSMR directly instead of lx.linear_solve, whose primitive is
            # re-traced at lowering outside the x64 context.
            state = solver.init(operator, options)
            x, _, _ = solver.compute(state, b, options)
            if diag_preconditioner is not None:
                x /= diag_preconditioner
        return x.astype(input_dtype)

    return solution


# =================================================================================
#                                Non-iterative solvers
# =================================================================================


def minnorm_shift_eig(
    rshift: float | None = None,
    ashift: float = 1e-6,
    dtype: DTypeLike | None = jnp.float64,
) -> Callable[..., jax.Array]:
    r"""
    The MinSR branch of `~quantax.optimizer.auto_shift_eig`, computing
    :math:`x = A^† (A A^† + \epsilon I)^{-1} b` by directly forming the shifted
    matrix and solving it with a Cholesky solver. Suitable for underdetermined
    problems where the number of parameters exceeds the number of samples.

    :param rshift:
        The relative diagonal shift, entering the shift as
        :math:`\epsilon = \mathrm{Tr}(A A^†) \times \mathrm{rshift} / \sqrt{d} + \mathrm{ashift}`,
        with :math:`d` the smaller one of the numbers of rows and columns of :math:`A`,
        i.e. the number of samples for an underdetermined problem. Default to
        :math:`10^{-12}` for double precision and :math:`10^{-6}` for single precision.

    :param ashift:
        The absolute diagonal shift, default to 1e-6.

    :param dtype:
        The dtype used internally in the solver, default to float64. Complex inputs use
        the complex counterpart, e.g. complex128 for float64. ``None`` keeps the dtype
        of the inputs.

    :return:
        A solver function with two arguments A and b and one output x as the solution of
        :math:`A x = b`.
    """

    @jax.jit
    def solution(A: jax.Array, b: jax.Array, **kwargs) -> jax.Array:
        input_dtype = A.dtype
        n, m = A.shape
        Adag = A.conj().T
        Adag = array_extend(Adag, jax.device_count())  # shard parameters evenly
        Adag = with_sharding_constraint(Adag, get_distributed_sharding())

        with precision_context(dtype):
            Adag = _to_dtype(Adag, dtype)
            b = _to_dtype(b, dtype)

            T = jnp.matmul(Adag.conj().T, Adag, precision="highest")
            T += _shift(Adag, rshift, ashift) * jnp.identity(n, T.dtype)
            T_inv_b = solve(T, b, assume_a="pos")  # cholesky solver is used internally

            x = jnp.matmul(Adag, T_inv_b, precision="highest")

        # cast back outside the precision context, which may disable x64
        return x[:m].astype(input_dtype)

    return solution


def lstsq_shift_eig(
    rshift: float | None = None,
    ashift: float = 1e-6,
    dtype: DTypeLike | None = jnp.float64,
) -> Callable[..., jax.Array]:
    r"""
    The SR branch of `~quantax.optimizer.auto_shift_eig`, computing
    :math:`x = (A^† A + \epsilon I)^{-1} A^† b` by directly forming the shifted
    matrix and solving it with a Cholesky solver. Suitable for overdetermined
    problems where the number of samples exceeds the number of parameters.

    :param rshift:
        The relative diagonal shift, entering the shift as
        :math:`\epsilon = \mathrm{Tr}(A^† A) \times \mathrm{rshift} / \sqrt{d} + \mathrm{ashift}`,
        with :math:`d` the smaller one of the numbers of rows and columns of :math:`A`,
        i.e. the number of parameters for an overdetermined problem. Default to
        :math:`10^{-12}` for double precision and :math:`10^{-6}` for single precision.

    :param ashift:
        The absolute diagonal shift, default to 1e-6.

    :param dtype:
        The dtype used internally in the solver, default to float64. Complex inputs use
        the complex counterpart, e.g. complex128 for float64. ``None`` keeps the dtype
        of the inputs.

    :return:
        A solver function with two arguments A and b and one output x as the solution of
        :math:`A x = b`.
    """

    @jax.jit
    def solution(A: jax.Array, b: jax.Array, **kwargs) -> jax.Array:
        input_dtype = A.dtype

        with precision_context(dtype):
            A = _to_dtype(A, dtype)
            b = _to_dtype(b, dtype)
            S = jnp.matmul(A.conj().T, A, precision="highest")
            F = jnp.matmul(A.conj().T, b, precision="highest")
            S += _shift(A, rshift, ashift) * jnp.identity(A.shape[1], S.dtype)
            x = solve(S, F, assume_a="pos")  # cholesky solver is used internally

        # cast back outside the precision context, which may disable x64
        return x.astype(input_dtype)

    return solution


def auto_shift_eig(
    rshift: float | None = None,
    ashift: float = 1e-6,
    dtype: DTypeLike | None = jnp.float64,
) -> Callable[..., jax.Array]:
    r"""
    Obtain the least-square minimum-norm solver for the linear equation
    :math:`Ax=b` using diagonal shift. It automatically chooses between
    :math:`x = (A^† A)^{-1} A^† b` and :math:`x = A^† (A A^†)^{-1} b`, which respectively
    correspond to SR and MinSR.

    Given :math:`M = A^† A` or :math:`M = A A^†` of dimension :math:`n`, the
    diagonal shift modifies it to :math:`M' = M + \epsilon I` for stable inversion,
    with :math:`\epsilon = \mathrm{Tr}(M) \times \mathrm{rshift} / \sqrt{n} + \mathrm{ashift}`.

    :param rshift:
        The relative diagonal shift. Default to be :math:`10^{-12}` for
        double precision and :math:`10^{-6}` for single precision.

    :param ashift:
        The absolute diagonal shift, default to 1e-6.

    :param dtype:
        The dtype used internally in the solver, default to float64. Complex inputs use
        the complex counterpart, e.g. complex128 for float64. ``None`` keeps the dtype
        of the inputs.

    :return:
        A solver function with two arguments A and b and one output x as the solution of
        :math:`A x = b`.
    """
    minnorm_solver = minnorm_shift_eig(rshift, ashift, dtype)
    lstsq_solver = lstsq_shift_eig(rshift, ashift, dtype)

    @jax.jit
    def solution(A: jax.Array, b: jax.Array, **kwargs) -> jax.Array:
        if A.shape[0] < A.shape[1]:
            return minnorm_solver(A, b, **kwargs)
        else:
            return lstsq_solver(A, b, **kwargs)

    return solution


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
) -> Callable[..., jax.Array]:
    r"""
    Solver for a Hermitian linear equation :math:`H x = b` via the pseudo-inverse
    of :math:`H` obtained from its eigendecomposition. Unlike the other solvers,
    the input is the Hermitian matrix :math:`H` itself rather than the
    rectangular :math:`A`, so this is used to solve the :math:`S \dot\theta = F`
    equation in `~quantax.optimizer.TimeEvol`.

    :param rtol:
        The relative tolerance for the pseudo-inverse. Eigenvalues smaller than
        ``rtol * max|eigenvalue| + atol`` are smoothly truncated. Default to
        :math:`10^{-12}` for double precision and :math:`10^{-6}` for single
        precision.

    :param atol:
        The absolute tolerance for the pseudo-inverse, default to 0.

    :return:
        A solver function with two arguments H and b and one output x as the
        solution of :math:`H x = b`.
    """

    @jax.jit
    def solution(H: jax.Array, b: jax.Array, **kwargs) -> jax.Array:
        eig_vals, U = eigh(H)
        eig_inv = _get_eigs_inv(eig_vals, rtol, atol)
        return jnp.einsum("rs,s,ts,t->r", U, eig_inv, U.conj(), b)

    return solution


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
    dtype: DTypeLike | None = jnp.float64,
) -> Callable[..., jax.Array]:
    r"""
    The MinSR branch of `~quantax.optimizer.auto_pinv_eig`, computing
    :math:`x = A^† (A A^†)^{-1} b` via the pseudo-inverse of :math:`A A^†` from its
    eigendecomposition. Suitable for underdetermined problems where the number of
    parameters exceeds the number of samples.

    :param rtol:
        The relative tolerance for pseudo-inverse. Default to be :math:`10^{-12}` for
        double precision and :math:`10^{-6}` for single precision.

    :param atol:
        The absolute tolerance for pseudo-inverse, default to 0.

    :param tol_snr:
        The tolerance of signal-to-noise ratio (SNR), default to 0 which means no regularization
        based on SNR. For details see `Phys. Rev. Lett. 125, 100503 <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.100503>`_.

    :param dtype:
        The dtype used internally in the solver, default to float64. Complex inputs use
        the complex counterpart, e.g. complex128 for float64. ``None`` keeps the dtype
        of the inputs.

    :return:
        A solver function with two arguments A and b and one output x as the solution of
        :math:`A x = b`.
    """

    @jax.jit
    def solution(A: jax.Array, b: jax.Array, **kwargs) -> jax.Array:
        input_dtype = A.dtype
        n, m = A.shape
        Adag = A.conj().T
        ndevices = jax.device_count()
        Adag = array_extend(Adag, ndevices)
        Adag = with_sharding_constraint(Adag, get_distributed_sharding())

        with precision_context(dtype):
            Adag = _to_dtype(Adag, dtype)
            b = _to_dtype(b, dtype)

            T = jnp.matmul(Adag.conj().T, Adag, precision="highest")
            # T_inv_b = pinv_solve(T, b, tol, atol, tol_snr)
            # x = jnp.einsum("rk,r->k", A.conj(), T_inv_b)
            eig_vals, U = eigh(T)
            eig_inv = _get_eigs_inv(eig_vals, rtol, atol)
            rho_ts = jnp.einsum("ts,t->ts", U.conj(), b, precision="highest")
            rho = _sum_without_noise(rho_ts, tol_snr)
            x = jnp.einsum("kr,rs,s,s->k", Adag, U, eig_inv, rho, precision="highest")

        # cast back outside the precision context, which may disable x64
        return x[:m].astype(input_dtype)

    return solution


def lstsq_pinv_eig(
    rtol: float | None = None,
    atol: float = 0.0,
    tol_snr: float = 0.0,
    dtype: DTypeLike | None = jnp.float64,
) -> Callable[..., jax.Array]:
    r"""
    The SR branch of `~quantax.optimizer.auto_pinv_eig`, computing
    :math:`x = (A^† A)^{-1} A^† b` via the pseudo-inverse of :math:`A^† A` from its
    eigendecomposition. Suitable for overdetermined problems where the number of
    samples exceeds the number of parameters.

    :param rtol:
        The relative tolerance for pseudo-inverse. Default to be :math:`10^{-12}` for
        double precision and :math:`10^{-6}` for single precision.

    :param atol:
        The absolute tolerance for pseudo-inverse, default to 0.

    :param tol_snr:
        The tolerance of signal-to-noise ratio (SNR), default to 0 which means no regularization
        based on SNR. For details see `Phys. Rev. Lett. 125, 100503 <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.100503>`_.

    :param dtype:
        The dtype used internally in the solver, default to float64. Complex inputs use
        the complex counterpart, e.g. complex128 for float64. ``None`` keeps the dtype
        of the inputs.

    :return:
        A solver function with two arguments A and b and one output x as the solution of
        :math:`A x = b`.
    """

    @jax.jit
    def solution(A: jax.Array, b: jax.Array, **kwargs) -> jax.Array:
        input_dtype = A.dtype

        with precision_context(dtype):
            A = _to_dtype(A, dtype)
            b = _to_dtype(b, dtype)
            S = jnp.matmul(A.conj().T, A, precision="highest")
            eig_vals, V = eigh(S)
            eig_inv = _get_eigs_inv(eig_vals, rtol, atol)
            rho_sk = jnp.einsum(
                "lk,sl,s->sk", V.conj(), A.conj(), b, precision="highest"
            )
            rho = _sum_without_noise(rho_sk, tol_snr)
            x = jnp.einsum("kl,l,l->k", V, eig_inv, rho, precision="highest")

        # cast back outside the precision context, which may disable x64
        return x.astype(input_dtype)

    return solution


def auto_pinv_eig(
    rtol: float | None = None,
    atol: float = 0.0,
    tol_snr: float = 0.0,
    dtype: DTypeLike | None = jnp.float64,
) -> Callable[..., jax.Array]:
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
        The tolerance of signal-to-noise ratio (SNR), default to 0 which means no regularization
        based on SNR. For details see `Phys. Rev. Lett. 125, 100503 <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.100503>`_.

    :param dtype:
        The dtype used internally in the solver, default to float64. Complex inputs use
        the complex counterpart, e.g. complex128 for float64. ``None`` keeps the dtype
        of the inputs.

    :return:
        A solver function with two arguments A and b and one output x as the solution of
        :math:`A x = b`.
    """
    minnorm_solver = minnorm_pinv_eig(rtol, atol, tol_snr, dtype)
    lstsq_solver = lstsq_pinv_eig(rtol, atol, tol_snr, dtype)

    @jax.jit
    def solution(A: jax.Array, b: jax.Array, **kwargs) -> jax.Array:
        if A.shape[0] < A.shape[1]:
            return minnorm_solver(A, b, **kwargs)
        else:
            return lstsq_solver(A, b, **kwargs)

    return solution


def block_pinv_eig(
    state: Variational,
    rtol: float | None = None,
    atol: float = 0.0,
    tol_snr: float = 0.0,
    dtype: DTypeLike | None = jnp.float64,
) -> Callable[..., jax.Array]:
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
        The tolerance of signal-to-noise ratio (SNR), default to 0 which means no regularization
        based on SNR. For details see `Phys. Rev. Lett. 125, 100503 <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.100503>`_.

    :param dtype:
        The dtype used internally in the solver, default to float64. Complex inputs use
        the complex counterpart, e.g. complex128 for float64. ``None`` keeps the dtype
        of the inputs.

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
    def solution(Obar: jax.Array, Ebar: jax.Array, **kwargs) -> jax.Array:
        Obar_list = jnp.split(Obar, Np_layer, axis=1)
        Ebar /= nlayers
        return jnp.concatenate([solver0(Oi, Ebar) for Oi in Obar_list], axis=0)

    return solution


def sgd_solver() -> Callable[..., jax.Array]:
    r"""
    The plain stochastic gradient descent solver, returning
    :math:`x = A^† b / N_s` without any preconditioning. Used in a
    `~quantax.optimizer.QNGD` optimizer, it recovers ordinary SGD on the energy
    instead of the natural gradient.

    :return:
        A solver function with two arguments A and b and one output x.
    """

    @jax.jit
    def solution(A: jax.Array, b: jax.Array, **kwargs) -> jax.Array:
        return jnp.einsum("sk,s->k", A.conj(), b) / b.shape[0]

    return solution
