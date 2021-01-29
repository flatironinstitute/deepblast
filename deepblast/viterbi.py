import torch
import torch.nn as nn
from deepblast.ops import operators
# from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
# from deepblast.constants import x, m, y, s


def _forward_pass(theta, A, pos, operator='softmax'):
    """ Forward algorithm for K state HMM.

    Parameters
    ----------
    theta : torch.Tensor
        Input Potentials of dimension N x M x S. This represents the
        pairwise residue distance across states.
    A : torch.Tensor
        Transition probabilities of dimensions N x M x S x S.
        All of these parameters are assumed to be in log units
    pos : list of int
        Specifies the differential indices for each state.
    operator : str
        The smoothed maximum operator.
    """
    op = operators[operator]
    new = theta.new
    N, M, S = theta.size()
    V = new(N + 1, M + 1, S).zero_()     # N x M x S
    Q = new(N + 2, M + 2, S, S).zero_()  # N x M x S x S
    Vt = new(S).zero_()    # S
    neg_inf = -1e8   # very negative number
    V = V + neg_inf
    V[0, 0] = 0   # Make all states equally likely to enter
    assert S == len(pos), f'{pos} does not have length {S}'
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            for k in range(S):
                di, dj = pos[k]
                res = op.max(V[i + di, j + dj] + A[i - 1, j - 1, k])
                V[i, j, k], Q[i, j, k] = res
            V[i, j] += theta[i - 1, j - 1]
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
    E = new(N + 2, M + 2, S).zero_()
    # Initial conditions (note first state is terminal state)
    E[N + 1, M + 1, 0] = Et
    # Backward pass
    for i in reversed(range(1, N + 1)):
        for j in reversed(range(1, M + 1)):
            for k in range(S):
                di, dj = pos[k]
                E[i, j] += Q[i - di, j - dj, k] * E[i - di, j - dj, k]
    return E


def _adjoint_forward_pass(Q, Ztheta, ZA, pos, operator):
    """ Calculate directional derivatives and Hessians.

    Parameters
    ----------
    Q : torch.Tensor
        Derivatives of max theta + v of dimension N x M x S x S
    Ztheta : torch.Tensor
        Derivative of theta of dimension N x M x S
    ZA : torch.Tensor
        Derivative of transition probabilities of dimension N x M x S x S
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
    N, M, S = Ztheta.size()
    N, M = N - 2, M - 2
    Vd = new(N + 1, M + 1, S).zero_()     # N x M
    Qd = new(N + 2, M + 2, S, S).zero_()  # N x M x S
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            for k in range(S):
                di, dj = pos[k]
                vd = Vd[i + di, j + dj] + ZA[i - 1, j - 1, k]
                q = Q[i, j, k]
                Vd[i, j, k] = q @ vd
                Qd[i, j, k] = op.hessian_product(q, vd)
            Vd[i, j] += Ztheta[i - 1, j - 1]
    # Terminate. First state *is* terminal state
    vd = Vd[N, M]  # + ZA[N, M, 0]  # Q: why does ZA have weird dims?
    q = Q[N, M, 0]
    Vdt = Ztheta[N, M, 0] + q @ vd
    Qd[N + 1, M + 1, 0] = op.hessian_product(q, vd)
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
    Ed = new(N + 2, M + 2, S).zero_()
    for i in reversed(range(1, N + 1)):
        for j in reversed(range(1, M + 1)):
            for k in range(S):
                di, dj = pos[k]
                Ed[i, j] += Qd[i - di, j - dj, k] * E[i - di, j - dj, k] + \
                    Q[i - di, j - dj, k] * Ed[i - di, j - dj, k]
    return Ed


class ForwardFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, theta, A, pos, operator):
        # Return both the alignment matrix
        Vt, Q = _forward_pass(theta, A, pos, operator)
        ctx.save_for_backward(theta, A, Q)
        ctx.others = pos, operator
        return Vt

    @staticmethod
    def backward(ctx, Et):
        theta, A, Q = ctx.saved_tensors
        pos, operator = ctx.others
        E, A = ForwardFunctionBackward.apply(
            theta, A, Et, Q, pos, operator)
        return E[1:-1, 1:-1], A, None, None, None


class ForwardFunctionBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, A, Et, Q, pos, operator):
        E = _backward_pass(Et, Q, pos)
        ctx.save_for_backward(E, Q)
        ctx.others = pos, operator
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
        pos, operator = ctx.others
        Vtd, Qd = _adjoint_forward_pass(Q, Ztheta, ZA, pos, operator)
        Ed = _adjoint_backward_pass(E, Q, Qd, pos)
        Ed = Ed[1:-1, 1:-1]
        return Ed, None, Vtd, None, None, None


def baumwelch(theta, A, operator):
    """ Implements the forward-backward algorithm.

    This is used to estimate the posterior distribution of
    states given the observed sequence.

    Parameters
    ----------
    theta : torch.Tensor
       Emission log probabilities of dimension N x M x S
    A : torch.Tensor
       Transition log probabilities of dimension N x M x S x S

    Returns
    -------
    posterior : torch.Tensor
       Posterior distribution across all states.  This is a
       N x M x S tensor of log probabilities.
    """
    fwd = ForwardFunction.apply(theta, A, operator)
    bwd = ForwardFunction.apply(
        theta[::-1, ::-1], A[::-1, ::-1].permute(0, 1, 3, 2),
        operator)
    posterior = fwd + bwd
    return posterior


class ForwardDecoder(nn.Module):
    def __init__(self, operator, pos):
        super().__init__()
        self.operator = operator
        self.pos = pos

    def forward(self, theta, A):
        # offloading to CPU for now
        theta = theta.cpu()
        A = A.cpu()
        return ForwardFunction.apply(theta, A, self.operator, self.pos)


class ViterbiDecoder(nn.Module):

    def __init__(self, operator, pos):
        super().__init__()
        self.operator = operator
        self.pos = pos

    def forward(self, theta, A):
        # offloading to CPU for now
        theta = theta.cpu()
        A = A.cpu()
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
