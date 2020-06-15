import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from deepblast.ops import operators
import numpy as np


def _forward_pass(theta, A, operator='softmax'):
    """  Forward pass to calculate DP alignment matrix

    Parameters
    ----------
    theta : torch.Tensor
        Input potentials of dimension B x N x M. This represents the
        pairwise residue match scores.
    A : torch.Tensor
        Gap penality of dimension B
    operator : str
        The smoothed maximum operator.

    Returns
    -------
    Vt : torch.Tensor
        Terminal alignment score of dimension B
    Q : torch.Tensor
        Derivatives of max theta + v of dimension B x N x M x S.
    """
    operator = operators[operator]
    new = theta.new
    B, N, M = theta.size()
    V = new(B, N + 1, M + 1).zero_()     # N x M
    Q = new(B, N + 2, M + 2, 3).zero_()  # N x M x S
    V[:, :, 0] = -1e10
    V[:, 0, :] = -1e10
    V[:, 0, 0] = 0.
    Q[:, N+1, M+1] = 1
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            v1 = A + V[:, i-1, j]
            v2 = V[:, i-1, j-1]
            v3 = A + V[:, i, j-1]
            v = torch.stack((v1, v2, v3), dim=1)
            v, Q[:, i, j] = operator.max(v)
            V[:, i, j] = theta[:, i-1, j-1] + v
    Vt = V[:, N, M]
    return Vt, Q


def _backward_pass(Et, Q):
    """ Backward pass to calculate grad DP

    Parameters
    ----------
    Et : torch.Tensor
        Terminal alignment edges of dimension B
    Q : torch.Tensor
        Derivatives of max (theta + v) of dimension B x N x M x S.

    Returns
    -------
    E : torch.Tensor
        Traceback matrix of dimension B x N x M
    """
    m, x, y = 1, 0, 2
    B, n_1, m_1, _ = Q.shape
    new = Q.new
    N, M = n_1 - 2, m_1 - 2
    E = new(B, N + 2, M + 2).zero_()
    E[:, N+1, M+1] = 1 * Et
    for i in reversed(range(1, N + 1)):
        for j in reversed(range(1, M + 1)):
            E[:, i, j] = Q[:, i + 1, j, x] * E[:, i + 1, j] + \
                Q[:, i + 1, j + 1, m] * E[:, i + 1, j + 1] + \
                Q[:, i, j + 1, y] * E[:, i, j + 1]
    return E


def _adjoint_forward_pass(Q, Ztheta, ZA, operator='softmax'):
    """ Calculate directional derivatives and Hessians.

    Parameters
    ----------
    Q : torch.Tensor
        Derivatives of max theta + v of dimension B x N x M x S
    Ztheta : torch.Tensor
        Derivative of theta of dimension B x N x M
    ZA : torch.Tensor
        Derivative of gap score of dimension B
    operator : str
        The smoothed maximum operator.

    Returns
    -------
    Vd : torch.Tensor
        Derivatives of V of dimension N x M x S
    Qd : torch.Tensor
        Derivatives of Q of dimension N x M x S x S
    """
    m, x, y = 1, 0, 2
    operator = operators[operator]
    new = Ztheta.new
    B, N, M = Ztheta.size()
    N, M = N - 2, M - 2
    Vd = new(B, N + 1, M + 1).zero_()     # N x M
    Qd = new(B, N + 2, M + 2, 3).zero_()  # N x M x S
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            v1 = ZA + Vd[:, i - 1, j]
            v2 = Vd[:, i - 1, j - 1]
            v3 = ZA + Vd[:, i, j - 1]
            Vd[:, i, j] = Ztheta[:, i, j] + \
                       Q[:, i, j, x] * v1 + \
                       Q[:, i, j, m] * v2 + \
                       Q[:, i, j, y] * v3
            v = torch.stack((v1, v2, v3), dim=1)
            Qd[:, i, j] = operator.hessian_product(Q[:, i, j], v)

    return Vd[:, N, M], Qd


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
    """
    m, x, y = 1, 0, 2
    B, n_1, m_1, _ = Q.shape
    new = Q.new
    N, M = n_1 - 2, m_1 - 2
    Ed = new(B, N + 2, M + 2).zero_()
    for i in reversed(range(1, N + 1)):
        for j in reversed(range(1, M + 1)):
            Ed[:, i, j] = Qd[:, i + 1, j, x] * E[:, i + 1, j] + \
                       Q[:, i + 1, j, x] * Ed[:, i + 1, j] + \
                       Qd[:, i + 1, j + 1, m] * E[:, i + 1, j + 1] + \
                       Q[:, i + 1, j + 1, m] * Ed[:, i + 1, j + 1] + \
                       Qd[:, i, j + 1, y] * E[:, i, j + 1] + \
                       Q[:, i, j + 1, y] * Ed[:, i, j + 1]
    return Ed


class NeedlemanWunschFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, theta, A, operator):
        # Return both the alignment matrix
        inputs = (theta, A)
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
        return E[:, 1:-1, 1:-1], A, None, None, None


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
            Derivative of transition matrix of dimension 3 x 3
        """
        E, Q = ctx.saved_tensors
        operator = ctx.others
        Vtd, Qd = _adjoint_forward_pass(Q, Ztheta, ZA, operator)
        Ed = _adjoint_backward_pass(E, Q, Qd)
        Ed = Ed[:, 1:-1, 1:-1]
        return Ed, None, Vtd, None, None, None


class NeedlemanWunschDecoder(nn.Module):

    def __init__(self, operator):
        super().__init__()
        self.operator = operator

    def forward(self, theta, A):
        return NeedlemanWunschFunction.apply(
            theta, A, self.operator)

    def traceback(self, grad):
        """ Computes traceback

        Parameters
        ----------
        grad : torch.Tensor
            Gradients of the alignment matrix of dimension N x M.

        Returns
        -------
        states : list of tuple
            Indices representing matches.
        """
        m, x, y = 1, 0, 2
        N, M = grad.shape
        states = torch.zeros(max(N, M))
        T = max(N, M)
        i, j = N - 1, M - 1
        states = [(i, j)]
        for t in reversed(range(T)):
            idx = torch.Tensor([
                [i - 1, j],
                [i - 1, j - 1],
                [i, j - 1]
            ]).long()
            ij = torch.argmax(
                   torch.Tensor([
                       grad[i - 1, j],
                       grad[i - 1, j - 1],
                       grad[i, j - 1]
                   ])
            )
            i, j = int(idx[ij][0]), int(idx[ij][1])
            states.append((i, j))
        return states[::-1]

    def decode(self, theta, A):
        """ Shortcut for doing inference. """
        # data, batch_sizes = theta
        with torch.enable_grad():
            # data.requires_grad_()
            nll = self.forward(theta, A)
            v = torch.sum(nll)
            v_grad, _ = torch.autograd.grad(
                v, (theta, A),
                create_graph=True)
        return v_grad
