import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from deepblast.ops import operators


def _forward_pass(theta, A, operator='softmax'):
    """  Forward pass to calculate DP alignment matrix

    Parameters
    ----------
    theta : torch.Tensor
        Input Potentials of dimension N x M. This represents the
        pairwise residue distance.
    psi : torch.Tensor
        Input Potentials of dimension N. This represents the
        gap score for the first sequence X.
    phi : torch.Tensor
        Input Potentials of dimension M. This represents the
        gap score for the second sequence Y.
    A : torch.Tensor
        Transition probabilities. Contains 4 elements [eta, delta, eps, tau].
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
    operator = operators[operator]
    new = theta.new
    N, M = theta.size()
    eta, delta, eps, tau = A[0], A[1], A[2], A[3]
    c = torch.log(1 - 2 * delta - tau) - torch.log(1 - eps - tau)
    # Initialize the matrices of interest. The 3rd axis here represents the
    # states that corresponds to (0) match, (1) gaps in X and (2) gaps in Y.
    V = new(N + 1, M + 1, 3).zero_()    # N x M x S
    Q = new(N + 2, M + 2, 3, 3).zero_() # N x M x S x S

    m, x, y = 0, 1, 2 # state numbering
    e = 1e-10
    neg_inf = -1e10
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
                    V[i-1, j, m] - delta,
                    V[i-1, j, x] - eps,
                    neg_inf
                ])
            )
            V[i, j, y], Q[i, j, y] = operator.max(
                torch.Tensor([
                    V[i, j-1, m] - d,
                    neg_inf
                    V[i, j-1, m] - e
                ])
            )
            V[i, j, m] += theta[i-1, j-1]

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
    Q[N + 1, M + 1] = 1  # Is this initialization correct?

    # Backward pass
    for i in reversed(range(1, N + 1)):
        for j in reversed(range(1, M + 1)):
            E[i, j, m] = Q[i + 1, j + 1, m] @ E[i + 1, j + 1]
            E[i, j, x] = Q[i + 1, j, x] @ E[i + 1, j]
            E[i, j, y] = Q[i, j + 1, y] @ E[i, j + 1]
    return E


def _adjoint_forward_pass(Q, E, Ztheta, ZA, operator='softmax'):
    """ Calculate directional derivatives and Hessians.

    Parameters
    ----------
    Q : torch.Tensor
        Derivatives of max theta + v of dimension N x M x S x S
    E : torch.Tensor
        Traceback matrix of dimension N x M x S
    Ztheta : torch.Tensor
        Derivative of theta of dimension N x M
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
    N, M = Ztheta.size()
    Zeta, Zdelta, Zeps, Ztau = ZA[0], ZA[1], ZA[2], ZA[3]

    Vd = new(N + 1, M + 1, 3).zero_()
    Qd = new(N + 2, M + 2, 3, 3).zero_()
    m, x, y = 0, 1, 2 # state numbering

    # Forward pass
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            # Need to double check these indices - they maybe wrong.
            Vd[i, j, m] = Ztheta[i - 1, j - 1] + \
                Q[i, j, m] @  torch.Tensor([Vd[i - 1, j - 1, m],
                                            Vd[i - 1, j - 1, x],
                                            Vd[i - 1, j - 1, y]])
            Vd[i, j, x] = Zpsi[i - 1] + \
                Q[i, j, x] @ torch.Tensor([Vd[i - 1, j, m] - Zd,
                                           Vd[i - 1, j, x] - Ze,
                                           0])
            Vd[i, j, y] = Zphi[j - 1] + \
                Q[i, j, y] @ torch.Tensor([Vd[i, j - 1, m] - Zd,
                                           0,
                                           Vd[i, j - 1, y] - Ze])
            vm = torch.Tensor([Vd[i - 1, j - 1, m], Vd[i - 1, j - 1, x], Vd[i - 1, j - 1, y]])
            vx = torch.Tensor([Vd[i - 1, j, m], Vd[i - 1, j, x], 0])
            vy = torch.Tensor([Vd[i, j - 1, m], 0, Vd[i, j - 1, y]])
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
    def forward(ctx, theta, psi, phi, A, operator):
        # Return both the alignment matrix
        V, Q = _forward_pass(theta, psi, phi, A, operator)
        ctx.save_for_backward(theta, psi, phi, Q)
        ctx.others = operator
        return V

    @staticmethod
    def backward(ctx, E):
        theta, psi, phi, Q = ctx.saved_tensors
        operator = ctx.others
        return ViterbiFunctionBackward.apply(
            theta, psi, phi, Q, E, operator)


class ViterbiFunctionBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, theta, psi, phi, Q, E, operator):
        E = _backward_pass(Q)
        ctx.save_for_backward(Q, E)
        ctx.others = operator
        return E

    @staticmethod
    def backward(ctx, Ztheta, Zphi, Zpsi):
        Q, E = ctx.saved_tensors
        operator = ctx.others
        Vd, Qd = _adjoint_forward_pass(Q, E, Ztheta, Zpsi, Zphi, operator)
        Ed = _adjoint_backward_pass(Q, Qd, E)
        return Ed, Vd


class ViterbiDecoder(nn.Module):

    def __init__(self, operator):
        super().__init__()
        self.operator = operator

    def forward(self, theta, psi, phi):
        return ViterbiFunction.apply(
            theta.data, psi.data, phi.data, self.operator)

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
