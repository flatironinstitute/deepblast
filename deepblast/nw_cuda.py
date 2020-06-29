import numba
from numba import cuda
from math import log, exp

max_cols = 2048
float_type = numba.float32


@cuda.jit(device=True)
def _soft_max(X, A):
    M = max(X[0], X[1], X[2])

    S = 0.0
    for i in range(3):
        A[i] = exp(X[i] - M)
        S += A[i]

    for i in range(3):
        A[i] /= S

    M += log(S)

    return M


@cuda.jit(device=True)
def _soft_max_hessian_product(P, Z, res):
    prod = numba.cuda.local.array(3, float_type)

    total = 0.0
    for i in range(3):
        prod[i] = P[i] * Z[i]
        total += prod[i]

    for i in range(3):
        res[i] = prod[i] - P[i] * total


@cuda.jit(device=True)
def _forward_pass(theta, A, Q):
    maxargs = numba.cuda.local.array(3, float_type)
    V = numba.cuda.local.array((2, max_cols), float_type)
    N, M = theta.shape

    last, curr = 0, 1
    for j in range(M + 1):
        V[last, j] = 0.0
    V[curr][0] = 0.0

    Q[-1, -1] = 1.0

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            maxargs[0] = A + V[last, j]  # x
            maxargs[1] = V[last, j - 1]  # m
            maxargs[2] = A + V[curr, j - 1]  # y

            v = _soft_max(maxargs, Q[i, j])
            V[curr, j] = theta[i - 1, j - 1] + v

        last = curr
        curr = 1 - curr

    return V[last, M]


@cuda.jit
def _forward_pass_batch(theta, A, Q, Vt):
    batchid = cuda.grid(1)
    if batchid < theta.shape[0]:
        Vt[batchid] = _forward_pass(theta[batchid], A[0], Q[batchid])


@cuda.jit(device=True)
def _backward_pass(Et, Q, E):
    m, x, y = 1, 0, 2
    n_1, m_1, _ = Q.shape
    N, M = n_1 - 2, m_1 - 2

    E[N + 1, M + 1] = Et
    Q[N + 1, M + 1] = 1
    for ir in range(1, N + 1):
        i = N + 1 - ir
        for jr in range(1, M + 1):
            j = M + 1 - jr
            E[i, j] = Q[i + 1, j, x] * E[i + 1, j] + \
                Q[i + 1, j + 1, m] * E[i + 1, j + 1] + \
                Q[i, j + 1, y] * E[i, j + 1]


@cuda.jit
def _backward_pass_batch(Et, Q, E):
    batchid = cuda.grid(1)
    if batchid < Q.shape[0]:
        _backward_pass(Et[0], Q[batchid], E[batchid])


@cuda.jit(device=True)
def _adjoint_forward_pass(Q, Ztheta, ZA, Qd):
    Vd = numba.cuda.local.array((2, max_cols), float_type)
    maxargs = numba.cuda.local.array(3, float_type)
    N, M = Ztheta.shape
    N, M = N - 2, M - 2

    last, curr = 0, 1
    for j in range(M + 1):
        Vd[last, j] = 0.0
    Vd[curr][0] = 0.0

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            maxargs[0] = ZA + Vd[last, j]  # x
            maxargs[1] = ZA + Vd[last, j - 1]  # m
            maxargs[2] = ZA + Vd[curr, j - 1]  # y
            Vd[curr, j] = Ztheta[i, j] + \
                Q[i, j, 0] * maxargs[0] + \
                Q[i, j, 1] * maxargs[1] + \
                Q[i, j, 2] * maxargs[2]
            _soft_max_hessian_product(Q[i, j], maxargs, Qd[i, j])

        last = curr
        curr = 1 - curr

    return Vd[last, M]


@cuda.jit
def _adjoint_forward_pass_batch(Q, Ztheta, ZA, Vd, Qd):
    batchid = cuda.grid(1)
    if batchid < Q.shape[0]:
        Vd[batchid] = _adjoint_forward_pass(Q[batchid], Ztheta[batchid],
                                            ZA[0], Qd[batchid])


@cuda.jit(device=True)
def _adjoint_backward_pass(E, Q, Qd, Ed):
    m, x, y = 1, 0, 2
    n_1, m_1, _ = Q.shape
    N, M = n_1 - 2, m_1 - 2
    # Ed = np.zeros((N + 2, M + 2))
    for ir in range(1, N + 1):
        i = N + 1 - ir
        for jr in range(1, M + 1):
            j = M + 1 - jr
            Ed[i, j] = Qd[i + 1, j, x] * E[i + 1, j] + \
                Q[i + 1, j, x] * Ed[i + 1, j] + \
                Qd[i + 1, j + 1, m] * E[i + 1, j + 1] + \
                Q[i + 1, j + 1, m] * Ed[i + 1, j + 1] + \
                Qd[i, j + 1, y] * E[i, j + 1] + \
                Q[i, j + 1, y] * Ed[i, j + 1]


@cuda.jit
def _adjoint_backward_pass_batch(E, Q, Qd, Ed):
    batchid = cuda.grid(1)
    if batchid < Q.shape[0]:
        _adjoint_backward_pass(E[batchid], Q[batchid], Qd[batchid],
                               Ed[batchid])
