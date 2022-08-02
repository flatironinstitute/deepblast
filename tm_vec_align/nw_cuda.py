import numba
import torch
import torch.nn as nn
from numba import cuda
from math import log, exp

torch.autograd.set_detect_anomaly(True)

max_cols = 2048
float_type = numba.float32
tpb = 32  # threads per block


@cuda.jit(device=True)
def _soft_max_device(X, A):
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
def _forward_pass_device(theta, A, Q):
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
            maxargs[0] = A[last, j - 1] + V[last, j]  # x
            maxargs[1] = V[last, j - 1]  # m
            maxargs[2] = A[last, j - 1] + V[curr, j - 1]  # y

            v = _soft_max_device(maxargs, Q[i, j])
            V[curr, j] = theta[i - 1, j - 1] + v

        last = curr
        curr = 1 - curr

    return V[last, M]


@cuda.jit
def _forward_pass_kernel(theta, A, Q, Vt):
    batchid = cuda.grid(1)
    if batchid < theta.shape[0]:
        Vt[batchid] = _forward_pass_device(theta[batchid], A[batchid],
                                           Q[batchid])


@cuda.jit(device=True)
def _backward_pass_device(Et, Q, E):
    m, x, y = 1, 0, 2
    n_1, m_1, _ = Q.shape
    N, M = n_1 - 2, m_1 - 2

    E[N + 1, M + 1] = Et
    for ir in range(1, N + 1):
        i = N + 1 - ir
        for jr in range(1, M + 1):
            j = M + 1 - jr
            E[i, j] = Q[i + 1, j, x] * E[i + 1, j] + \
                Q[i + 1, j + 1, m] * E[i + 1, j + 1] + \
                Q[i, j + 1, y] * E[i, j + 1]


@cuda.jit
def _backward_pass_kernel(Et, Q, E):
    batchid = cuda.grid(1)
    if batchid < Q.shape[0]:
        _backward_pass_device(Et[batchid], Q[batchid], E[batchid])


@cuda.jit(device=True)
def _adjoint_forward_pass_device(Q, Ztheta, ZA, Qd):
    Vd = numba.cuda.local.array((2, max_cols), float_type)
    maxargs = numba.cuda.local.array(3, float_type)
    N, M = Ztheta.shape
    N, M = N - 2, M - 2

    last, curr = 0, 1
    for j in range(M + 1):
        Vd[last, j] = 0.0
        Vd[curr, j] = 0.0

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            maxargs[0] = ZA[last, j - 1] + Vd[last, j]  # x
            maxargs[1] = Vd[last, j - 1]  # m
            maxargs[2] = ZA[last, j - 1] + Vd[curr, j - 1]  # y
            Vd[curr, j] = Ztheta[i, j] + \
                Q[i, j, 0] * maxargs[0] + \
                Q[i, j, 1] * maxargs[1] + \
                Q[i, j, 2] * maxargs[2]
            _soft_max_hessian_product(Q[i, j], maxargs, Qd[i, j])

        last = curr
        curr = 1 - curr

    return Vd[last, M]


@cuda.jit
def _adjoint_forward_pass_kernel(Q, Ztheta, ZA, Vd, Qd):
    batchid = cuda.grid(1)
    if batchid < Q.shape[0]:
        Vd[batchid] = _adjoint_forward_pass_device(Q[batchid], Ztheta[batchid],
                                                   ZA[batchid], Qd[batchid])


@cuda.jit(device=True)
def _adjoint_backward_pass_device(E, Q, Qd, Ed):
    m, x, y = 1, 0, 2
    n_1, m_1, _ = Q.shape
    N, M = n_1 - 2, m_1 - 2

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
def _adjoint_backward_pass_kernel(E, Q, Qd, Ed):
    batchid = cuda.grid(1)
    if batchid < Q.shape[0]:
        _adjoint_backward_pass_device(E[batchid], Q[batchid], Qd[batchid],
                                      Ed[batchid])


class NeedlemanWunschFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, A, operator):
        if operator != 'softmax':
            raise NotImplementedError(
                "CUDA variant only supports 'softmax' operator")
        if theta.dtype != torch.float32:
            raise TypeError("CUDA variant only supports torch.float32 type")

        # Return both the alignment matrix
        B, N, M = theta.shape

        Q = torch.zeros((B, N + 2, M + 2, 3),
                        dtype=theta.dtype,
                        device=theta.device)
        Vt = torch.zeros((B), dtype=theta.dtype, device=theta.device)
        bpg = (B + (tpb - 1)) // tpb  # blocks per grid

        _forward_pass_kernel[tpb, bpg](theta.detach(), A.detach(), Q, Vt)

        ctx.save_for_backward(theta, A, Q)
        ctx.others = operator
        return Vt

    @staticmethod
    def backward(ctx, Et):
        """
        Parameters
        ----------
        ctx : ?
           Some autograd context object
        Et : torch.Tensor
           Last alignment trace (scalar value)
        """
        theta, A, Q = ctx.saved_tensors
        operator = ctx.others

        E, A = NeedlemanWunschFunctionBackward.apply(theta, A, Et, Q, operator)
        return E[:, 1:-1, 1:-1], A, None, None, None


class NeedlemanWunschFunctionBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, A, Et, Q, operator):
        if operator != 'softmax':
            raise NotImplementedError(
                "CUDA variant only supports 'softmax' operator")

        B, N, M = theta.shape

        E = torch.zeros((B, N + 2, M + 2),
                        dtype=theta.dtype,
                        device=theta.device)
        bpg = (B + (tpb - 1)) // tpb  # blocks per grid

        _backward_pass_kernel[tpb, bpg](Et.detach(), Q, E)

        ctx.save_for_backward(Q, E)
        ctx.others = operator
        return E, A

    @staticmethod
    def backward(ctx, Ztheta, ZA):
        """
        Parameters
        ----------
        ctx : ?
           Some autograd context object
        Ztheta : torch.Tensor
            Derivative of theta of dimension N x M
        ZA : torch.Tensor
            Derivative of transition matrix of dimension 3 x 3
        """
        # operator = ctx.others
        Q, E = ctx.saved_tensors

        B, ZN, ZM = Ztheta.shape
        bpg = (B + (tpb - 1)) // tpb  # blocks per grid

        Qd = torch.zeros((B, ZN, ZM, 3),
                         dtype=Ztheta.dtype,
                         device=Ztheta.device)
        Vtd = torch.zeros(B, dtype=Ztheta.dtype, device=Ztheta.device)
        Ed = torch.zeros((B, ZN, ZM), dtype=Ztheta.dtype, device=Ztheta.device)

        _adjoint_forward_pass_kernel[tpb, bpg](Q, Ztheta, ZA, Vtd, Qd)
        _adjoint_backward_pass_kernel[tpb, bpg](E.detach(), Q, Qd, Ed)

        Ed = Ed[:, 1:-1, 1:-1]
        return Ed, None, Vtd, None, None, None


class NeedlemanWunschDecoder(nn.Module):
    def __init__(self, operator):
        super().__init__()
        self.operator = operator

    def forward(self, theta, A):
        return NeedlemanWunschFunction.apply(theta, A, self.operator)

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
        m, x, y = 1, 0, 2
        N, M = grad.shape
        states = torch.zeros(max(N, M))
        i, j = N - 1, M - 1
        idx = torch.Tensor([[i - 1, j], [i - 1, j - 1], [i, j - 1]]).long()
        states = [(i, j, m)]
        max_ = -1e10
        while True:
            left = max_ if i <= 0 else grad[i - 1, j]
            diag = max_ if (i <= 0 and j <= 0) else grad[i - 1, j - 1]
            upper = max_ if j <= 0 else grad[i, j - 1]
            if diag == max_ or upper == max_ or left == max_:
                break
            ij = torch.argmax(torch.Tensor([left, diag, upper]))
            xmy = torch.Tensor([x, m, y])
            i, j = int(idx[ij][0]), int(idx[ij][1])
            idx = torch.Tensor([[i - 1, j], [i - 1, j - 1], [i, j - 1]]).long()
            s = int(xmy[ij])
            states.append((i, j, s))

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
        with torch.enable_grad():
            nll = self.forward(theta, A)
            v = torch.sum(nll)
            v_grad, _ = torch.autograd.grad(v, (theta, A), create_graph=True)
        return v_grad
