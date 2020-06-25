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

    last = 0
    curr = 1
    for j in range(theta.shape[0] + 1):
        V[last, j] = 0.0
    V[1][0] = 0.0

    Q[-1, -1] = 1.0

    for i in range(1, theta.shape[0] + 1):
        for j in range(1, theta.shape[1] + 1):
            maxargs[0] = A + V[last, j]  # x
            maxargs[1] = V[last, j-1]    # m
            maxargs[2] = A + V[curr, j-1]  # y

            v = _soft_max(maxargs, Q[i, j])
            V[curr, j] = theta[i - 1, j - 1] + v

        last = curr
        curr = 1 - curr


@cuda.jit
def _forward_pass_batch(theta, A, Q):
    batchid = cuda.grid(1)
    if batchid < theta.shape[0]:
        _forward_pass(theta[batchid], A[0], Q[batchid])
