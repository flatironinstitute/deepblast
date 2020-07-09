import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from deepblast.ops import operators


def _forward_pass(theta, A, operator='softmax'):
    """  Forward pass to calculate DP alignment matrix

    Parameters
    ----------
    theta : torch.Tensor
        Input Potentials of dimension N x M x S. This represents the
        pairwise residue distance across states.
    psi : torch.Tensor
        Input Potentials of dimension N. This represents the
        gap score for the first sequence X.
    phi : torch.Tensor
        Input Potentials of dimension M. This represents the
        gap score for the second sequence Y.
    A : torch.Tensor
        Transition probabilities. Contains 4 elements [n, d, e, t].
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
    ie = 1e-10  # a very small number for numerical stability
    operator = operators[operator]
    new = theta.new
    N, M, _ = theta.size()
    n, d, e, t = A[0], A[1], A[2], A[3]
    eps = (1 - t) * torch.exp(-e)
    net = (1-n) / (1 - e - t)
    delta = (1 - 2 * torch.exp(-d) * net)  * net * (1 - t) * torch.exp(-d)
    c = torch.log(1 - 2 * delta - t + ie) - torch.log(1 - eps - t + ie)

    # Initialize the matrices of interest. The 3rd axis here represents the
    # states that corresponds to (0) match, (1) gaps in X and (2) gaps in Y.
    V = new(N + 1, M + 1, 3).zero_()    # N x M x S
    Q = new(N + 2, M + 2, 3, 3).zero_() # N x M x S x S

    m, x, y = 0, 1, 2 # state numbering
    neg_inf = -1e10   # very negative number
    V[0, 0, m] = 2 * n
    # Forward pass
    for i in range(1, N):
        for j in range(1, M):
            V[i, j, m], Q[i, j, m] = operator.max(
                torch.Tensor([
                    V[i-1, j-1, m],
                    V[i-1, j-1, x],
                    V[i-1, j-1, y]
                ])
            )
            V[i, j, x], Q[i, j, x] = operator.max(
                torch.Tensor([
                    V[i-1, j, m] - d,
                    V[i-1, j, x] - e,
                    neg_inf
                ])
            )
            V[i, j, y], Q[i, j, y] = operator.max(
                torch.Tensor([
                    V[i, j-1, m] - d,
                    neg_inf,
                    V[i, j-1, m] - e
                ])
            )
            V[i, j] += theta[i-1, j-1]

    V[N, M, m], Q[N, M, m] = operator.max(
        torch.Tensor([
            V[N, M, m],
            V[N, M, x] + c,
            V[N, M, y] + c
        ])
    )
    return V[N, M, m], Q


def _backward_pass(Et, Q):
    """ Backward pass to calculate grad DP

    Parameters
    ----------
    Q : torch.Tensor
        Derivatives of max theta + v of dimension N x M x S x S.

    Returns
    -------
    E : torch.Tensor
        Traceback matrix of dimension N x M x S
    """
    n_1, m_1, _, _ = Q.shape
    new = Q.new
    N, M = n_1 - 2, m_1 - 2
    m, x, y = 0, 1, 2 # state numbering

    E = new(N + 2, M + 2, 3).zero_()
    # Initial conditions
    E[N + 1, M + 1] = Et

    # Backward pass
    for i in reversed(range(1, N + 1)):
        for j in reversed(range(1, M + 1)):
            E[i, j, m] = Q[i + 1, j + 1, m] @ E[i + 1, j + 1]
            E[i, j, x] = Q[i + 1, j, x] @ E[i + 1, j]
            E[i, j, y] = Q[i, j + 1, y] @ E[i, j + 1]
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
    operator = operators[operator]
    new = Ztheta.new
    N, M, _ = Ztheta.size()
    Zn, Zd, Ze, Zt = ZA[0], ZA[1], ZA[2], ZA[3]

    Vd = new(N + 1, M + 1, 3).zero_()
    Qd = new(N + 2, M + 2, 3, 3).zero_()
    m, x, y = 0, 1, 2 # state numbering
    Vd[0, 0, m] = Zn
    # Forward pass
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            vm = torch.Tensor([Vd[i - 1, j - 1, m], Vd[i - 1, j - 1, x], Vd[i - 1, j - 1, y]])
            vx = torch.Tensor([Vd[i - 1, j, m], Vd[i - 1, j, x], 0])
            vy = torch.Tensor([Vd[i, j - 1, m], 0, Vd[i, j - 1, y]])
            # Need to double check these indices - they maybe wrong.
            Vd[i, j, m] = Ztheta[i - 1, j - 1, m] + Q[i, j, m] @ vm
            Vd[i, j, x] = Ztheta[i - 1, j - 1, x] + Q[i, j, x] @ vx
            Vd[i, j, y] = Ztheta[i - 1, j - 1, y] + Q[i, j, y] @ vy
            Qd[i, j, m] = operator.hessian_product(Q[i, j, m], vm)
            Qd[i, j, x] = operator.hessian_product(Q[i, j, x], vx)
            Qd[i, j, y] = operator.hessian_product(Q[i, j, y], vy)
    # TODO: Need to figure out how to terminate
    return Vd, Qd


def _adjoint_backward_pass(Q, Qd, E):
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
    m, x, y = 0, 1, 2 # state numbering
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
        Vd, Qd = _adjoint_forward_pass(Q, Ztheta, ZA, operator)
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
        return ViterbiFunction.apply(
            theta, A, self.operator)

    def decode(self, theta, psi, phi):
        """ Shortcut for doing inference. """
        # data, batch_sizes = theta
        with torch.enable_grad():
            # data.requires_grad_()
            nll = self.forward(theta, psi, phi)
            v = torch.sum(nll)
            v_grad, = torch.autograd.grad(
                v, (theta.data, psi.data, phi.data,),
                create_graph=True)
        return v_grad
