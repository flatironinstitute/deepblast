import torch
import torch.nn as nn
from deepblast.ops import operators
# from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
# from deepblast.constants import x, m, y, s


def _forward_pass(theta, pos, operator='softmax'):
    """ Forward algorithm for K state HMM.

    Parameters
    ----------
    theta : torch.Tensor
        Input Potentials of dimension N x M x S x S.
        This represents the combination of transition
        and emission probabilities.
    pos : list of int
        Specifies the differential indices for each state.
    operator : str
        The smoothed maximum operator.
    """
    op = operators[operator]
    new = theta.new
    N, M, S, S = theta.size()
    V = new(N + 1, M + 1, S).zero_()     # N x M x S
    Q = new(N + 2, M + 2, S, S).zero_()  # N x M x S x S
    # Q[:, :] = 1 / S  # uniform distribution
    Vt = new(S).zero_()    # S
    neg_inf = -1e8   # very negative number
    V = V + neg_inf
    V[0, 0] = 0   # Make all states equally likely to enter
    assert S == len(pos), f'{pos} does not have length {S}'
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            for s in range(S):
                di, dj = pos[s]
                res = op.max(V[i + di, j + dj] + theta[i - 1, j - 1, s])
                V[i, j, s], Q[i, j, s] = res
    # Terminate. First state *is* terminal state
    Vt, Q[N + 1, M + 1, 0] = op.max(V[N, M])
    return Vt, Q


def _backward_pass(Et, Q, pos):
    """ Backward pass to calculate grad DP.

    This is the derivative of the forward pass.

    Parameters
    ----------
    Et : torch.Tensor
        Input scalar derivative from upstream step.
    Q : torch.Tensor
        Derivatives of max theta + v of dimension N x M x S x S.
    pos : list of int
        Specifies the differential indices for each state.

    Returns
    -------
    E : torch.Tensor
        Traceback matrix of dimension N x M x S
    """
    n_1, m_1, S, _ = Q.shape
    new = Q.new
    N, M = n_1 - 2, m_1 - 2
    U = new(N + 2, M + 2, S).zero_()
    E = new(N + 2, M + 2, S, S).zero_()
    # Initial conditions (note first state is terminal state)
    E[N + 1, M + 1, 0] = Et / S  # not sure about this...
    U[N + 1, M + 1] = torch.sum(E[N + 1, M + 1, :, 0])
    # Backward pass
    for i in reversed(range(1, N + 1)):
        for j in reversed(range(1, M + 1)):
            for s in range(S):
                di, dj = pos[s]
                E[i, j, :, s] = Q[i - di, j - dj, :, s] * U[i - di, j - dj]
                U[i, j, s] = E[i, j, :, s].sum()
                # print(i, j, s, i - di, j - dj,
                #       'E', E[i, j, s].detach().numpy(),
                #       'Q', Q[i - di, j - dj, s].detach().numpy(),
                #       'U', U[i - di, j - dj].detach().numpy(),
                #       'U', U[i, j, s].detach().numpy())
    return E


def _adjoint_forward_pass(Q, Ztheta, pos, operator):
    """ Calculate directional derivatives and Hessians.

    Parameters
    ----------
    Q : torch.Tensor
        Derivatives of max theta + v of dimension N x M x S x S
    Ztheta : torch.Tensor
        Derivative of theta of dimension N x M x S x S
    operator : str
        The smoothed maximum operator.
    pos :str
        Differential indexes to help facilitate the construction of
        alternative HMM architectures.

    Returns
    -------
    Vd : torch.Tensor
        Derivatives of V of dimension N x M x S
    Qd : torch.Tensor
        Derivatives of Q of dimension N x M x S x S
    """
    op = operators[operator]
    new = Ztheta.new
    N, M, S, S = Ztheta.size()
    N, M = N - 2, M - 2
    Vd = new(N + 1, M + 1, S).zero_()     # N x M
    Qd = new(N + 2, M + 2, S, S).zero_()  # N x M x S
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            for s in range(S):
                di, dj = pos[s]
                qvd = torch.zeros(S)
                for k in range(S):
                    qvd[k] = Vd[i + di, j + dj, k] + Ztheta[i, j, s, k]
                    Vd[i, j, s] += Q[i, j, s, k] * qvd[k]
                Qd[i, j, s] = op.hessian_product(Q[i, j, s], qvd)
    # Terminate. First state *is* terminal state
    Vdt = Q[N + 1, M + 1, 0] @ Vd[N, M]
    Qd[N + 1, M + 1, 0] = op.hessian_product(Q[N + 1, M + 1, 0], Vd[N, M])
    return Vdt, Qd


def _adjoint_backward_pass(E, Q, Qd, pos):
    """ Calculate directional derivatives and Hessians.

    Parameters
    ----------
    E : torch.Tensor
        Traceback matrix of dimension N x M x S
    Q : torch.Tensor
        Derivatives of max theta + v of dimension N x M x S x S
    Qd : torch.Tensor
        Derivatives of Q of dimension N x M x S

    Returns
    -------
    Ed : torch.Tensor
        Derivative of traceback matrix of dimension N x M x S

    Notes
    -----
    Careful with Ztheta, it actually has dimensions (N + 2)  x (M + 2).
    The border elements aren't useful, only need Ztheta[1:-1, 1:-1]
    """
    n_1, m_1, S, _ = Q.shape
    new = Q.new
    N, M = n_1 - 2, m_1 - 2
    Ed = new(N + 2, M + 2, S, S).zero_()
    for i in reversed(range(1, N + 1)):
        for j in reversed(range(1, M + 1)):
            for s in range(S):
                di, dj = pos[s]
                Ed[i, j] += Qd[i - di, j - dj, s] * E[i - di, j - dj, s] + \
                    Q[i - di, j - dj, s] * Ed[i - di, j - dj, s]
    return Ed


class ForwardFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, theta, pos, operator):
        # Return both the alignment matrix
        Vt, Q = _forward_pass(theta, pos, operator)
        ctx.save_for_backward(theta, Q)
        ctx.others = pos, operator
        return Vt

    @staticmethod
    def backward(ctx, Et):
        theta, Q = ctx.saved_tensors
        pos, operator = ctx.others
        E = ForwardFunctionBackward.apply(
            theta, Et, Q, pos, operator)
        return E[1:-1, 1:-1], None, None, None


class ForwardFunctionBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, Et, Q, pos, operator):
        E = _backward_pass(Et, Q, pos)
        ctx.save_for_backward(E, Q)
        ctx.others = pos, operator
        return E

    @staticmethod
    def backward(ctx, Ztheta):
        """
        Parameters
        ----------
        ctx : ?
           Some autograd context object
        Ztheta : torch.Tensor
            Derivative of theta of dimension N x M
        """
        E, Q = ctx.saved_tensors
        pos, operator = ctx.others
        Vtd, Qd = _adjoint_forward_pass(Q, Ztheta, pos, operator)
        Ed = _adjoint_backward_pass(E, Q, Qd, pos)
        Ed = Ed[1:-1, 1:-1]
        return Ed, Vtd, None, None, None


class ForwardDecoder(nn.Module):
    def __init__(self, operator, pos):
        super().__init__()
        self.operator = operator
        self.pos = pos

    def forward(self, theta):
        # offloading to CPU for now
        theta = theta.cpu()
        return ForwardFunction.apply(theta, self.operator, self.pos)


class ViterbiDecoder(nn.Module):

    def __init__(self, operator, pos):
        super().__init__()
        self.operator = operator
        self.pos = pos

    def forward(self, theta):
        # offloading to CPU for now
        theta = theta.cpu()
        # Forward and Backward algorithm (aka Baum-Welch)
        return baumwelch(theta, A, self.operator)

    def decode(self, theta, A):
        """ Shortcut for doing inference. """
        # data, batch_sizes = theta
        with torch.enable_grad():
            # data.requires_grad_()
            nll = self.forward(theta, A)
            v = torch.sum(nll)
            v_grad, = torch.autograd.grad(
                v, (theta, A), create_graph=True)
        return v_grad
