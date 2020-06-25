import numba
from numba import cuda
from math import log, exp


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
def _forward_pass(theta, A, Q):
    maxargs = numba.cuda.local.array(3, numba.float64)
    V = numba.cuda.local.array((2, 2048), numba.float64)
    N, M = theta.shape

    last, curr = 0, 1
    for j in range(M + 1):
        V[last, j] = 0.0
    V[curr][0] = 0.0

    Q[-1, -1] = 1.0

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            maxargs[0] = A + V[last, j]  # x
            maxargs[1] = V[last, j-1]    # m
            maxargs[2] = A + V[curr, j-1]  # y

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
    if batchid < E.shape[0]:
        _backward_pass(Et[0], Q[batchid], E[batchid])
