import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from deepblast.ops import operators
from deepblast.constants import x, m, y


def _forward_pass(theta, A, operator='softmax'):
    """  Forward pass to calculate DP alignment matrix

    Parameters
    ----------
    theta : torch.Tensor
        Input Potentials of dimension N x M x S. This represents the
        pairwise residue distance across states.
    A : torch.Tensor
        Transition probabilities of dimensions S x S.
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
    """
    # m, x, y = 0, 1, 2
    operator = operators[operator]
    new = theta.new
    N, M, _ = theta.size()

    # Initialize the matrices of interest. The 3rd axis here represents the
    # states that corresponds to (0) match, (1) gaps in X and (2) gaps in Y.
    V = new(N + 1, M + 1, 3).zero_()    # N x M x S
    Q = new(N + 2, M + 2, 3, 3).zero_() # N x M x S x S
    # Q = Q + (1 / 3)
    neg_inf = -1e8   # very negative number
    V = V + neg_inf
    V[0, 0, m] = 0   # force first state to be a match
    # Forward pass

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            V[i, j, m], Q[i, j, m] = operator.max(V[i - 1, j - 1] + A[m])
            V[i, j, x], Q[i, j, x] = operator.max(V[i - 1, j] + A[x])
            V[i, j, y], Q[i, j, y] = operator.max(V[i, j - 1] + A[y])
            V[i, j] += theta[i - 1, j - 1]  # give emission probs to all states

    # print('M', V[:, :, m])
    # print('X', V[:, :, x])
    # print('Y', V[:, :, y])
    Vt, Q[N + 1, M + 1, m] = operator.max(V[N, M])
    return Vt, Q


def _backward_pass(Et, Q):
    """ Backward pass to calculate grad DP

    Parameters
    ----------
    Et : torch.Tensor
        Input derivative from upstream step.
    Q : torch.Tensor
        Derivatives of max theta + v of dimension N x M x S x S.

    Returns
    -------
    E : torch.Tensor
        Traceback matrix of dimension N x M x S
    """
    # m, x, y = 0, 1, 2
    n_1, m_1, _, _ = Q.shape
    new = Q.new
    N, M = n_1 - 2, m_1 - 2
    E = new(N + 2, M + 2, 3).zero_()
    # Initial conditions
    E[N + 1, M + 1, m] = Et
    # Backward pass
    for i in reversed(range(1, N + 1)):
        for j in reversed(range(1, M + 1)):
            E[i, j] = Q[i + 1, j + 1, m] * E[i + 1, j + 1, m] + \
                      Q[i + 1, j, x] * E[i + 1, j, x] + \
                      Q[i, j + 1, y] * E[i, j + 1, y]
    return E


def _adjoint_forward_pass(Q, Ztheta, ZA, operator='softmax'):
    """ Calculate directional derivatives and Hessians.

    Parameters
    ----------
    Q : torch.Tensor
        Derivatives of max theta + v of dimension N x M x S x S
    Ztheta : torch.Tensor
        Derivative of theta of dimension N x M x S
    ZA : torch.Tensor
        Derivative of transition probabilities A
    operator : str
        The smoothed maximum operator.

    Returns
    -------
    Vd : torch.Tensor
        Derivatives of V of dimension N x M x S
    Qd : torch.Tensor
        Derivatives of Q of dimension N x M x S x S
    """
    neg_inf = -1e8   # very negative number
    operator = operators[operator]
    new = Ztheta.new
    N, M, _ = Ztheta.size()
    N, M = N - 2, M - 2
    Zn, Zd, Ze, Zt = ZA[0], ZA[1], ZA[2], ZA[3]
    Zeps = (1 - Zt) * torch.exp(-Ze)
    Znet = (1 - Zn) / (1 - Ze - Zt)
    Zdelta = (1 - 2 * torch.exp(-Zd) * Znet)  * Znet * (1 - Zt) * torch.exp(-Zd)
    # TODO : Zc maybe calculated incorrectly
    Zc = torch.log(1 - 2 * Zdelta - Zt + ie) - torch.log(1 - Zeps - Zt + ie)
    Vd = new(N + 1, M + 1, 3).zero_()
    Qd = new(N + 2, M + 2, 3, 3).zero_()
    Vd[0, 0, m] = 2 * Zn
    # Forward pass
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            # vm = torch.Tensor([Vd[i - 1, j - 1, m], Vd[i - 1, j - 1, x], Vd[i - 1, j - 1, y]])
            vm = Vd[i - 1, j - 1, :]
            vx = torch.Tensor([Vd[i - 1, j, m] - Zd, Vd[i - 1, j, x] - Ze, 0])
            vy = torch.Tensor([Vd[i, j - 1, m] - Zd, 0, Vd[i, j - 1, y] - Ze])
            Vd[i, j, m] = Ztheta[i, j, m] + Q[i, j, m] @ vm
            Vd[i, j, x] = Ztheta[i, j, x] + Q[i, j, x] @ vx
            Vd[i, j, y] = Ztheta[i, j, y] + Q[i, j, y] @ vy
            Qd[i, j, m] = operator.hessian_product(Q[i, j, m], vm)
            Qd[i, j, x] = operator.hessian_product(Q[i, j, x], vx)
            Qd[i, j, y] = operator.hessian_product(Q[i, j, y], vy)

    vm = torch.Tensor([
        Vd[N - 1, M - 1, m], Vd[N - 1, M - 1, x] + Zc, Vd[N - 1, M - 1, y] + Zc
    ])
    Vtd = Q[N, M, m] @ vm
    Qd[N + 1, M + 1, m] = operator.hessian_product(Q[N, M, m], vm)
    return Vtd, Qd


def _adjoint_backward_pass(E, Q, Qd):
    """ Calculate directional derivatives and Hessians.

    Parameters
    ----------
    Q : torch.Tensor
        Derivatives of max theta + v of dimension N x M x S x S
    Qd : torch.Tensor
        Derivatives of Q of dimension N x M
    E : torch.Tensor
        Traceback matrix of dimension N x M

    Returns
    -------
    Ed : torch.Tensor
        Derivative of traceback matrix of dimension N x M.
    """
    N_1, M_1, _, _ = Q.size()
    N, M = N_1 - 2, M_1 - 2
    new = Qd.new
    Ed = new(N + 2, M + 2, 3).zero_()

    # Backward pass
    for i in reversed(range(1, N + 1)):
        for j in reversed(range(1, M + 1)):
            Ed[i, j, m] = Qd[i+1, j+1, m] @ E[i+1, j+1] + Q[i+1, j+1, m] @ Ed[i+1, j+1]
            Ed[i, j, x] = Qd[i+1, j, x] @ E[i+1, j] + Q[i+1, j, x] @ Ed[i+1, j]
            Ed[i, j, y] = Qd[i, j+1, y] @ E[i, j+1] + Q[i, j+1, y] @ Ed[i, j+1]
    return Ed


class ViterbiFunction(torch.autograd.Function):

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
        E, A = ViterbiFunctionBackward.apply(
            theta, A, Et, Q, operator)
        return E[1:-1, 1:-1], A, None, None, None


class ViterbiFunctionBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, theta, A, Et, Q, operator):
        E = _backward_pass(Et, Q)
        ctx.save_for_backward(E, Q)
        ctx.others = operator
        return E, A

    @staticmethod
    def backward(ctx, Ztheta, ZA):
        E, Q = ctx.saved_tensors
        operator = ctx.others
        Vtd, Qd = _adjoint_forward_pass(Q, Ztheta, ZA, operator)
        Ed = _adjoint_backward_pass(E, Q, Qd)
        Ed = Ed[1:-1, 1:-1]
        return Ed, None, Vtd, None, None, None


class ViterbiDecoder(nn.Module):

    def __init__(self, operator):
        super().__init__()
        self.operator = operator

    def forward(self, theta, A):
        theta = theta.cpu()
        A = A.cpu()
        res =  ViterbiFunction.apply(
            theta, A, self.operator)
        return res

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
