import torch
import torch.nn as nn
from deepblast.ops import operators
import numba
import numpy as np

use_numba = True


@numba.njit
def _soft_max_numba(X):
    M = X[0]
    for i in range(1, 3):
        M = X[i] if X[i] > M else M

    A = np.empty_like(X)
    S = 0.0
    for i in range(3):
        A[i] = np.exp(X[i] - M)
        S += A[i]

    for i in range(3):
        A[i] /= S

    M += np.log(S)

    return M, A


@numba.njit
def _soft_max_hessian_product_numba(P, Z):
    prod = P * Z

    prod = np.empty_like(P)
    for i in range(3):
        prod[i] = P[i] * Z[i]

    res = np.empty_like(P)
    total = np.sum(prod)
    for i in range(3):
        res[i] = prod[i] - P[i] * total

    return res


@numba.njit
def _forward_pass_numba(theta, A):
    N, M = theta.shape
    V = np.zeros((N + 1, M + 1))     # N x M
    Q = np.zeros((N + 2, M + 2, 3))  # N x M x S
    Q[N + 1, M + 1] = 1
    m, x, y = 1, 0, 2
    maxargs = np.empty(3)
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            maxargs[x] = A[i - 1, j - 1] + V[i - 1, j]  # x
            maxargs[m] = V[i - 1, j - 1]  # m
            maxargs[y] = A[i - j, 1 - 1] + V[i, j - 1]  # y
            v, Q[i, j] = _soft_max_numba(maxargs)
            V[i, j] = theta[i - 1, j - 1] + v
    Vt = V[N, M]
    return Vt, Q


def _forward_pass(theta, A, operator='softmax'):
    """  Forward pass to calculate DP alignment matrix

    Parameters
    ----------
    theta : torch.Tensor
        Input potentials of dimension N x M.
        This represents the pairwise residue match scores.
    A : torch.Tensor
        Gap penality (scalar valued)
    operator : str
        The smoothed maximum operator.

    Returns
    -------
    Vt : torch.Tensor
        Terminal alignment score (just 1 dimension)
    Q : torch.Tensor
        Derivatives of max theta + v of dimension N x M x S.
    """
    if not use_numba or operator != 'softmax':
        operator = operators[operator]
        new = theta.new
        N, M = theta.size()
        V = new(N + 1, M + 1).zero_()     # N x M
        Q = new(N + 2, M + 2, 3).zero_()  # N x M x S
        Q[N + 1, M + 1] = 1
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                tmp = torch.Tensor([
                    A[i - 1, j - 1] + V[i - 1, j],
                    V[i - 1, j - 1],
                    A[i - 1, j - 1] + V[i, j - 1]
                ])
                v, Q[i, j] = operator.max(tmp)
                V[i, j] = theta[i - 1, j - 1] + v

        Vt = V[N, M]
    else:
        Vt, Q = _forward_pass_numba(
            theta.detach().cpu().numpy(),
            A.detach().cpu().numpy())
        Vt = torch.tensor(Vt, dtype=theta.dtype)
        Q = torch.from_numpy(Q)

    return Vt, Q


@numba.njit
def _backward_pass_numba(Et, Q):
    m, x, y = 1, 0, 2
    n_1, m_1, _ = Q.shape
    N, M = n_1 - 2, m_1 - 2
    E = np.zeros((N + 2, M + 2))
    E[N + 1, M + 1] = Et
    Q[N + 1, M + 1] = 1
    for ir in range(1, N + 1):
        i = N + 1 - ir
        for jr in range(1, M + 1):
            j = M + 1 - jr
            E[i, j] = Q[i + 1, j, x] * E[i + 1, j] + \
                Q[i + 1, j + 1, m] * E[i + 1, j + 1] + \
                Q[i, j + 1, y] * E[i, j + 1]
    return E


def _backward_pass(Et, Q):
    """ Backward pass to calculate grad DP

    Parameters
    ----------
    Et : torch.Tensor
        Terminal alignment edge (scalar valued).
    Q : torch.Tensor
        Derivatives of max (theta + v) of dimension N x M x S.

    Returns
    -------
    E : torch.Tensor
        Traceback matrix of dimension N x M x S
    """
    if not use_numba:
        m, x, y = 1, 0, 2
        n_1, m_1, _ = Q.shape
        new = Q.new
        N, M = n_1 - 2, m_1 - 2
        E = new(N + 2, M + 2).zero_()
        E[N + 1, M + 1] = 1 * Et
        Q[N + 1, M + 1] = 1
        for i in reversed(range(1, N + 1)):
            for j in reversed(range(1, M + 1)):
                E[i, j] = Q[i + 1, j, x] * E[i + 1, j] + \
                    Q[i + 1, j + 1, m] * E[i + 1, j + 1] + \
                    Q[i, j + 1, y] * E[i, j + 1]
    else:
        import collections
        if isinstance(Et, collections.abc.Sequence):
            Et_float = float(Et[0])
        else:
            Et_float = float(Et)
        E = torch.from_numpy(_backward_pass_numba(
            Et_float, Q.detach().cpu().numpy()))

    return E


@numba.njit
def _adjoint_forward_pass_numba(Q, Ztheta, ZA):
    N, M = Ztheta.shape
    N, M = N - 2, M - 2
    Vd = np.zeros((N + 1, M + 1))      # N x M
    Qd = np.zeros((N + 2, M + 2, 3))   # N x M x S
    m, x, y = 1, 0, 2
    maxargs = np.empty(3)
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            # Note: the indexing of ZA doesn't match Ztheta
            # See forward_pass method.
            maxargs[x] = ZA[i - 1, j - 1] + Vd[i - 1, j]
            maxargs[m] = Vd[i - 1, j - 1]
            maxargs[y] = ZA[i - 1, j - 1] + Vd[i, j - 1]
            Vd[i, j] = Ztheta[i, j] + \
                Q[i, j, x] * maxargs[0] + \
                Q[i, j, m] * maxargs[1] + \
                Q[i, j, y] * maxargs[2]
            Qd[i, j] = _soft_max_hessian_product_numba(
                Q[i, j], maxargs)
    return Vd[N, M], Qd


def _adjoint_forward_pass(Q, Ztheta, ZA, operator='softmax'):
    """ Calculate directional derivatives and Hessians.

    Parameters
    ----------
    Q : torch.Tensor
        Derivatives of max theta + v of dimension N x M x S
    Ztheta : torch.Tensor
        Derivative of theta of dimension N x M
    ZA : torch.Tensor
        Derivative of gap score.
    operator : str
        The smoothed maximum operator.

    Returns
    -------
    Vd : torch.Tensor
        Derivatives of V of dimension N x M
    Qd : torch.Tensor
        Derivatives of Q of dimension N x M x S
    """
    if not use_numba or operator != 'softmax':
        m, x, y = 1, 0, 2
        operator = operators[operator]
        new = Ztheta.new
        N, M = Ztheta.size()
        N, M = N - 2, M - 2
        Vd = new(N + 1, M + 1).zero_()     # N x M
        Qd = new(N + 2, M + 2, 3).zero_()  # N x M x S
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                Vd[i, j] = Ztheta[i, j] + \
                    Q[i, j, x] * (ZA[i - 1, j - 1] + Vd[i - 1, j]) + \
                    Q[i, j, m] * Vd[i - 1, j - 1] + \
                    Q[i, j, y] * (ZA[i - 1, j - 1] + Vd[i, j - 1])
                vd = torch.Tensor([(ZA[i - 1, j - 1] + Vd[i - 1, j]),
                                   Vd[i - 1, j - 1],
                                   (ZA[i - 1, j - 1] + Vd[i, j - 1])])
                Qd[i, j] = operator.hessian_product(Q[i, j], vd)
        return Vd[N, M], Qd
    else:
        Vd, Qd = _adjoint_forward_pass_numba(
            Q.detach().cpu().numpy(), Ztheta.detach().cpu().numpy(),
            ZA.detach().cpu().numpy())
        Vd = torch.tensor(Vd, dtype=Ztheta.dtype)
        Qd = torch.from_numpy(Qd)
        return Vd, Qd


@numba.njit
def _adjoint_backward_pass_numba(E, Q, Qd):
    m, x, y = 1, 0, 2
    n_1, m_1, _ = Q.shape
    N, M = n_1 - 2, m_1 - 2
    Ed = np.zeros((N + 2, M + 2))
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
    return Ed


def _adjoint_backward_pass(E, Q, Qd):
    """ Calculate directional derivatives and Hessians.

    Parameters
    ----------
    E : torch.Tensor
        Traceback matrix of dimension N x M
    Q : torch.Tensor
        Derivatives of max theta + v of dimension N x M x S
    Qd : torch.Tensor
        Derivatives of Q of dimension N x M

    Returns
    -------
    Ed : torch.Tensor
        Derivative of traceback matrix of dimension N x M.

    Notes
    -----
    Careful with Ztheta, it actually has dimensions (N + 2)  x (M + 2).
    The border elements aren't useful, only need Ztheta[1:-1, 1:-1]
    """
    if not use_numba:
        m, x, y = 1, 0, 2
        n_1, m_1, _ = Q.shape
        new = Q.new
        N, M = n_1 - 2, m_1 - 2
        Ed = new(N + 2, M + 2).zero_()
        for i in reversed(range(1, N + 1)):
            for j in reversed(range(1, M + 1)):
                Ed[i, j] = Qd[i + 1, j, x] * E[i + 1, j] + \
                    Q[i + 1, j, x] * Ed[i + 1, j] + \
                    Qd[i + 1, j + 1, m] * E[i + 1, j + 1] + \
                    Q[i + 1, j + 1, m] * Ed[i + 1, j + 1] + \
                    Qd[i, j + 1, y] * E[i, j + 1] + \
                    Q[i, j + 1, y] * Ed[i, j + 1]
    else:
        Ed = _adjoint_backward_pass_numba(
            E.detach().cpu().numpy(), Q.detach().cpu().numpy(),
            Qd.detach().cpu().numpy())
        Ed = torch.tensor(Ed)

    return Ed


class NeedlemanWunschFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, theta, A, operator):
        # Return both the alignment matrix
        Vt, Q = _forward_pass(theta, A, operator)
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
        E, A = NeedlemanWunschFunctionBackward.apply(
            theta, A, Et, Q, operator)
        return E[1:-1, 1:-1], A, None, None, None


class NeedlemanWunschFunctionBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, theta, A, Et, Q, operator):
        E = _backward_pass(Et, Q)
        ctx.save_for_backward(E, Q)
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
            Derivative of affine gap matrix
        """
        E, Q = ctx.saved_tensors
        operator = ctx.others
        Vtd, Qd = _adjoint_forward_pass(Q, Ztheta, ZA, operator)
        Ed = _adjoint_backward_pass(E, Q, Qd)
        Ed = Ed[1:-1, 1:-1]
        return Ed, None, Vtd, None, None, None


class NeedlemanWunschDecoder(nn.Module):

    def __init__(self, operator):
        super().__init__()
        self.operator = operator

    def forward(self, theta, A):
        theta = theta.cpu()
        A = A.cpu()
        return NeedlemanWunschFunction.apply(
            theta, A, self.operator)

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
        states = [(i, j, m)]
        max_ = -100000
        while True:
            idx = torch.Tensor([[i - 1, j], [i - 1, j - 1], [i, j - 1]]).long()
            left = max_ if i <= 0 else grad[i - 1, j]
            diag = max_ if (i <= 0 and j <= 0) else grad[i - 1, j - 1]
            upper = max_ if j <= 0 else grad[i, j - 1]
            if diag == max_ and upper == max_ and left == max_:
                break
            ij = torch.argmax(torch.Tensor([left, diag, upper]))
            xmy = torch.Tensor([x, m, y])
            i, j = int(idx[ij][0]), int(idx[ij][1])
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
        # data, batch_sizes = theta
        theta = theta.cpu()
        A = A.cpu()
        with torch.enable_grad():
            # data.requires_grad_()
            nll = self.forward(theta, A)
            v = torch.sum(nll)
            v_grad, _ = torch.autograd.grad(
                v, (theta, A),
                create_graph=True)
        return v_grad
