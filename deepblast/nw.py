import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from deepblast.ops import operators


def _forward_pass(theta, A, operator='softmax'):
    """  Forward pass to calculate DP alignment matrix

    Parameters
    ----------
    theta : torch.Tensor
        Input potentials of dimension N x M. This represents the
        pairwise residue match scores.
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
    operator = operators[operator]
    new = theta.new
    N, M = theta.size()
    V = new(N + 1, M + 1).zero_()     # N x M
    Q = new(N + 2, M + 2, 3).zero_()  # N x M x S
    V[:, 0] = -1e10
    V[0, :] = -1e10
    V[0, 0] = 0.
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            v = torch.Tensor([
                A + V[i-1, j],
                V[i-1, j-1],
                A + V[i, j-1]
            ])
            v, Q[i, j] = operator.max(v)
            V[i, j] = theta[i-1, j-1] + v
    Vt = V[N, M]
    return Vt, Q


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
    m, x, y = 1, 0, 2
    n_1, m_1, _ = Q.shape
    new = Q.new
    N, M = n_1 - 2, m_1 - 2
    E = new(N + 2, M + 2).zero_()
    E[N+1, M+1] = 1 * Et
    Q[N+1, M+1, m] = 1
    for i in reversed(range(1, N + 1)):
        for j in reversed(range(1, M + 1)):
            E[i, j] = Q[i + 1, j, x] * E[i + 1, j] + \
                Q[i + 1, j + 1, m] * E[i + 1, j + 1] + \
                Q[i, j + 1, y] * E[i, j + 1]
            #print(Q[i + 1, j, x], Q[i + 1, j + 1, m], Q[i, j + 1, m])
    return E


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
        Derivatives of V of dimension N x M x S
    Qd : torch.Tensor
        Derivatives of Q of dimension N x M x S x S
    """
    pass

def _adjoint_backward_pass(Et, E, Q, Qd):
    """ Calculate directional derivatives and Hessians.

    Parameters
    ----------
    Et : torch.Tensor
        Terminal alignment edge (scalar valued)..
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
    pass


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
        return E[1:-1, 1:-1], A, None, None, None


class NeedlemanWunschFunctionBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, theta, A, Et, Q, operator):
        E = _backward_pass(Et, Q)
        ctx.save_for_backward(Et, E, Q)
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
        Et, E, Qt, Q = ctx.saved_tensors
        operator = ctx.others
        Vtd, Qtd, Qd = _adjoint_forward_pass(Qt, Q, Ztheta, ZA, operator)
        Ed = _adjoint_backward_pass(Q, Qd, E)
        Ed = Ed[1:-1, 1:-1]
        return Ed, Vtd, None, None, None, None


class NeedlemanWunschDecoder(nn.Module):

    def __init__(self, operator):
        super().__init__()
        self.operator = operator

    def forward(self, theta, A):
        return NeedlemanWunschFunction.apply(
            theta, A, self.operator)

    def decode(self, theta, A):
        """ Shortcut for doing inference. """
        # data, batch_sizes = theta
        with torch.enable_grad():
            # data.requires_grad_()
            nll = self.forward(theta, A)
            v = torch.sum(nll)
            v_grad, = torch.autograd.grad(
                v, (theta.data, A.data,),
                create_graph=True)
        return v_grad
