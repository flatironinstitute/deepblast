import torch
import numba
from numba import cuda
from math import exp, log, isnan
import torch.nn as nn
# FIXME: Refactor cuda utility and variables
torch.autograd.set_detect_anomaly(True)

max_cols = 2048
float_type = numba.float32
max_S = 8
tpb = 32  # threads per block


@cuda.jit(device=True)
def _soft_max_device(X, S, A):
    M = X[0]
    for i in range(1, S):
        M = max(M, X[i])
    sumA = 0
    for i in range(S):
        sumA += exp(X[i] - M)
    M += log(sumA)
    for i in range(S):
        A[i] = exp(X[i] - M)
    return M


@cuda.jit(device=True)
def _forward_pass_device(theta, A, pos, Q):
    maxargs = cuda.local.array(max_S, float_type)
    V = cuda.local.array((2, max_cols, max_S), float_type)
    N, M, S = theta.shape

    neg_inf = -1e8
    last, curr = 0, 1
    for j in range(M + 1):
        for k in range(S):
            V[last, j, k] = neg_inf
            V[curr, j, k] = neg_inf

    for k in range(S):
        V[last, 0][k] = 0.0

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            for k in range(S):
                di = pos[k][0]
                dj = pos[k][1]

                for l in range(S):
                    maxargs[l] = V[curr + di, j + dj, l] + A[i - 1, j - 1, k, l]

                V[curr, j, k] = _soft_max_device(maxargs, S, Q[i, j, k])

            for k in range(0, S):
                V[curr, j, k] += theta[i - 1, j - 1, k]

        last = curr
        curr = 1 - curr

    Vt = _soft_max_device(V[last, M], S, Q[N + 1, M + 1, 0])
    return Vt


@cuda.jit
def _forward_pass_kernel(theta, A, pos, Q, Vt):
    batchid = cuda.grid(1)
    if batchid < theta.shape[0]:
        Vt[batchid] = _forward_pass_device(theta[batchid], A[batchid], pos[batchid],
                                           Q[batchid])


@cuda.jit(device=True)
def _backward_pass_device(Et, Q, pos, E):
    Etmp = cuda.local.array(max_S, float_type)
    n_1, m_1, S = Q.shape[0], Q.shape[1], Q.shape[2]
    N, M = n_1 - 2, m_1 - 2
    E[N + 1, M + 1, 0] = Et
    for ir in range(1, N + 1):
        i = N + 1 - ir
        for jr in range(1, M + 1):
            j = M + 1 - jr

            for k in range(S):
                di, dj = pos[k][0], pos[k][1]
                for l in range(S):
                    Etmp[l] = Q[i - di, j - dj, k, l] * E[i - di, j - dj, k]
                for l in range(S):
                    E[i, j, l] += Etmp[l]


@cuda.jit
def _backward_pass_kernel(Et, Q, pos, E):
    batchid = cuda.grid(1)
    if batchid < Q.shape[0]:
        _backward_pass_device(Et[batchid], Q[batchid], pos[batchid], E[batchid])


class ForwardFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, theta, A, pos):
        # Return both the alignment matrix
        B, N, M, S = theta.shape

        Q = torch.zeros((B, N + 2, M + 2, S, S),
                        dtype=theta.dtype,
                        device=theta.device)
        Vt = torch.zeros((B), dtype=theta.dtype, device=theta.device)
        bpg = (B + (tpb - 1)) // tpb  # blocks per grid
        #print(type(theta), type(A), type(pos), type(Q), type(Vt))
        _forward_pass_kernel[tpb, bpg](theta.detach(), A.detach(), pos, Q, Vt)
        ctx.save_for_backward(theta, A, Q)
        ctx.others = pos

        return Vt

    @staticmethod
    def backward(ctx, Et):
        theta, A, Q = ctx.saved_tensors
        B, N, M, S = theta.shape
        pos = ctx.others
        bpg = (B + (tpb - 1)) // tpb  # blocks per grid
        E = torch.zeros((B, N + 2, M + 2, S),
                        dtype=theta.dtype,
                        device=theta.device)
        _backward_pass_kernel[tpb, bpg](Et, Q, pos, E)

        return E[:, 1:-1, 1:-1], A, None, None, None


def baumwelch(theta, A, pos):
    """ Implements the forward-backward algorithm.

    This is used to estimate the posterior distribution of
    states given the observed sequence.

    Parameters
    ----------
    theta : torch.Tensor
       Emission log probabilities of dimension B X N x M x S
    A : torch.Tensor
       Transition log probabilities of dimension B X N x M x S x S

    Returns
    -------
    posterior : torch.Tensor
       Posterior distribution across all states.  This is a
       B X N x M x S tensor of log probabilities.
    """
    fwd = ForwardFunction.apply(
        theta, A, pos)
    bwd = ForwardFunction.apply(
        theta[:, ::-1, ::-1], A[:, ::-1, ::-1].permute(0, 1, 2, 4, 3), pos)
    posterior = fwd + bwd
    return posterior


class ForwardDecoder(nn.Module):
    def __init__(self, pos):
        super().__init__()
        self.pos = pos

    def forward(self, theta, A):
        return ForwardFunction.apply(theta, A, self.pos)


class ViterbiDecoder(nn.Module):
    def __init__(self, pos):
        super().__init__()
        self.pos = pos

    def forward(self, theta, A):
        # offloading to CPU for now
        # Forward and Backward algorithm (aka Baum-Welch)
        return baumwelch(theta, A, self.pos)

    def traceback(self, grad):
        """ Computes traceback

        Parameters
        ----------
        grad : torch.Tensor
            Gradients of the alignment matrix.

        Returns
        -------
        states : list of tuple
            Indices representing matches.
        """
        x, y = 1, 2
        N, M = grad.shape
        states = torch.zeros(max(N, M))
        i, j = N - 1, M - 1
        states = [(i, j, m)]
        max_ = -1e8
        while True:
            if i > 0 and j > 0:
                t = torch.Tensor([grad[i + p[0], j + p[1]] for p in pos])
                s = torch.argmax(t)
                di, dj = self.pos[s]
                i, j = i + di, j + dj
                states.append(i, j, s)

        # take care of any outstanding gaps
        while i > 0:
            i = i - 1
            s = x
            states.append((i, j, s))

        while j > 0:
            j = j - 1
            s = y
            states.append((i, j, s))

        return states[::-1]

    def decode(self, theta, A):
        """ Shortcut for doing inference. """
        # data, batch_sizes = theta
        with torch.enable_grad():
            # data.requires_grad_()
            nll = self.forward(theta, A)
            v = torch.sum(nll)
            v_grad, = torch.autograd.grad(
                v, (theta, A),
               create_graph=True)
        return v_grad
