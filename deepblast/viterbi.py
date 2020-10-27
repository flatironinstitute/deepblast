import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from deepblast.ops import operators
from deepblast.constants import x, m, y, s


def hmmforward(theta, A, pos, operator='softmax'):
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
    new = theta.new
    N, M, S = theta.size()
    V = new(N + 1, M + 1, S).zero_()    # N x M x S
    Vt = new(S).zero_()    #S
    Q = new(N + 2, M + 2, S, S).zero_() # N x M x S x S
    assert S == len(pos)
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            for k in range(S):
                for k_ in range(S):
                    di, dj = pos[k_], pos[k_]
                    V[i, j, k], Q[i, j, k] += op.max(
                        V[i - di, j - dj] + A[i - 1, j - 1, k_])
                    V[i, j] += theta[i - di, j - dj]
    # Terminate
    i, j = N + 1, M + 1
    for k in range(S):
        for k_ in range(S):
            di, dj = pos[k_], pos[k_]
            Vt[k], Q[i, j, k] += op.max(
                V[i - di, j - dj] + A[i - 1, j - 1, k_])
    return Vt, Q


def _forward_pass(theta, A, operator='softmax'):
    """  Forward pass to calculate DP alignment matrix.

    If the softmax op is used, this is an explicit implementation
    of the Forward algorithm commonly used for Expectation-Maximization
    optimization of HMMs.

    Parameters
    ----------
    theta : torch.Tensor
        Input Potentials of dimension N x M x S. This represents the
        pairwise residue distance across states.
    A : torch.Tensor
        Transition probabilities of dimensions N x M x S x S.
        All of these parameters are assumed to be in log units
    operator : str
        The smoothed maximum operator.

    Returns
    -------
    V : torch.Tensor
        Estimated alignment matrix of dimension N x M x S.
        S is the number of states (here it is 3 for M, X, Y).
    Q : torch.Tensor
        Derivatives of max theta + v of dimension N x M x S x S.

    Notes
    -----
    This is currently coded for 4 explicit states, namely matches,
    insertions, deletions and slips. A slip here indicates that
    two residues don't match and skipped over.
    """
    op = operators[operator]
    new = theta.new
    N, M, S = theta.size()
    # Initialize the matrices of interest.
    V = new(N + 1, M + 1, S).zero_()    # N x M x S
    Q = new(N + 2, M + 2, S, S).zero_() # N x M x S x S
    neg_inf = -1e8   # very negative number
    V = V + neg_inf
    V[0, 0] = 0   # Make all 4 states equally likely to enter
    # Forward pass
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            V[i, j, m], Q[i, j, m] = op.max(V[i-1, j-1] + A[i-1, j-1, m])
            V[i, j, s], Q[i, j, s] = op.max(V[i-1, j-1] + A[i-1, j-1, s])
            V[i, j, x], Q[i, j, x] = op.max(V[i-1, j] + A[i-1, j-1, x])
            V[i, j, y], Q[i, j, y] = op.max(V[i, j-1] + A[i-1, j-1, y])
            V[i, j] += theta[i-1, j-1]  # give emission probs to all states
    Vt, Q[N + 1, M + 1, m] = op.max(V[N, M])
    return Vt, Q


def _backward_pass(Et, Q):
    """ Backward pass to calculate grad DP.

    This is the derivative of the forward pass.

    Parameters
    ----------
    Et : torch.Tensor
        Input scalar derivative from upstream step.
    Q : torch.Tensor
        Derivatives of max theta + v of dimension N x M x S x S.

    Returns
    -------
    E : torch.Tensor
        Traceback matrix of dimension N x M x S
    """
    n_1, m_1, S, _ = Q.shape
    new = Q.new
    N, M = n_1 - 2, m_1 - 2
    E = new(N + 2, M + 2, S).zero_()
    # Initial conditions
    E[N + 1, M + 1, m] = Et
    # Backward pass
    for i in reversed(range(1, N + 1)):
        for j in reversed(range(1, M + 1)):
            E[i, j] = Q[i + 1, j + 1, m] * E[i + 1, j + 1, m] + \
                      Q[i + 1, j + 1, s] * E[i + 1, j + 1, s] + \
                      Q[i + 1, j, x] * E[i + 1, j, x] + \
                      Q[i, j + 1, y] * E[i, j + 1, y]
    return E


class ForwardFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, theta, A, operator):
        # Return both the alignment matrix
        Vt, Q = _forward_pass(theta, A, operator)
        ctx.save_for_backward(theta, A, Q)
        ctx.others = operator
        return Vt

    @staticmethod
    def backward(ctx, Et):
        theta, A, Q = ctx.saved_tensors
        operator = ctx.others
        E = _adjoint_forward_pass(Et, Q)
        return E[1:-1, 1:-1], A, None, None, None


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
    fwd =  ForwardFunction.apply(
        theta, A, operator)
    bwd =  ForwardFunction.apply(
        theta[::-1, ::-1], A[::-1, ::-1].permute(0, 1, 3, 2),
        operator)
    posterior = fwd + bwd
    return posterior


class ViterbiDecoder(nn.Module):

    def __init__(self, operator):
        super().__init__()
        self.operator = operator

    def forward(self, theta, A):
        # offloading to CPU for now
        theta = theta.cpu()
        A = A.cpu()
        # Forward and Backward algorithm (aka Baum-Welch)
        exp = baumwelch(theta, A, self.operator)
        return exp

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
