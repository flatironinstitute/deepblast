import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence


class HardMaxOp:
    @staticmethod
    def max(X):
        M, _ = torch.max(X, keepdim=True)
        A = (M == X).float()
        A = A / torch.sum(A, keepdim=True)

        return M.squeeze(), A.squeeze()

    @staticmethod
    def hessian_product(P, Z):
        return torch.zeros_like(Z)


class SoftMaxOp:
    @staticmethod
    def max(X):
        M = torch.max(X)
        X = X - M
        A = torch.exp(X)
        S = torch.sum(A)
        M = M + torch.log(S)
        A /= S
        return M.squeeze(), A.squeeze()

    @staticmethod
    def hessian_product(P, Z):
        prod = P * Z
        return prod - P * torch.sum(prod, keepdim=True)


class SparseMaxOp:
    @staticmethod
    def max(X):
        seq_len, n_batch, n_states = X.shape
        X_sorted, _ = torch.sort(X, descending=True)
        cssv = torch.cumsum(X_sorted) - 1
        ind = X.new(n_states)
        for i in range(n_states):
            ind[i] = i + 1
        cond = X_sorted - cssv / ind > 0
        rho = cond.long().sum()
        cssv = cssv.view(-1, n_states)
        rho = rho.view(-1)

        tau = (torch.gather(cssv, dim=1, index=rho[:, None] - 1)[:, 0]
               / rho.type(X.type()))
        tau = tau.view(seq_len, n_batch)
        A = torch.clamp(X - tau[:, :, None], min=0)
        # A /= A.sum(dim=2, keepdim=True)

        M = torch.sum(A * (X - .5 * A))

        return M.squeeze(), A.squeeze()

    @staticmethod
    def hessian_product(P, Z):
        S = (P > 0).type(Z.type())
        support = torch.sum(S, keepdim=True)
        prod = S * Z
        return prod - S * torch.sum(prod, keepdim=True) / support


operators = {'softmax': SoftMaxOp, 'sparsemax': SparseMaxOp,
             'hardmax': HardMaxOp}


def _forward_pass(theta, psi, phi, A, operator='softmax'):
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
        Transition probabilities. Only contains 2 elements [delta, eps].
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
    delta, eps = A[0], A[1]
    # Initialize the matrices of interest. The 3rd axis here represents the
    # states that corresponds to (0) match, (1) gaps in X and (2) gaps in Y.
    V = new(N + 1, M + 1, 3).zero_()    # N x M x S
    Q = new(N + 2, M + 2, 3, 3).zero_() # N x M x S x S

    m, x, y = 0, 1, 2 # state numbering
    e = 1e-10
    log1_2delta = torch.log(1 - 2 * delta + e)
    loge_1 = torch.log(1 - eps + e)
    logd, loge = torch.log(delta + e), torch.log(eps + e)

    # Forward pass
    for i in range(1, N):
        for j in range(1, M):
            V[i, j, m], Q[i, j, m] = operator.max(
                torch.Tensor([
                    theta[i, j] + log1_2delta + V[i-1, j-1, m],
                    theta[i, j] + loge_1 + V[i-1, j-1, x],
                    theta[i, j] + loge_1 + V[i-1, j-1, y]
                ])
            )
            V[i, j, x], Q[i, j, x] = operator.max(
                torch.Tensor([
                    psi[i] + logd + V[i-1, j, m],
                    psi[i] + loge + V[i-1, j, x],
                    0
                ])
            )
            V[i, j, y], Q[i, j, y] = operator.max(
                torch.Tensor([
                    phi[i] + logd + V[i, j-1, m],
                    0,
                    phi[i] + loge + V[i, j-1, y]
                ])
            )
    return V, Q


def _backward_pass(Q):
    """ Backward pass to calculate grad DP

    Parameters
    ----------
    Q : torch.Tensor
        Derivatives of max theta + v of dimension N x M x 3.

    Returns
    -------
    E : torch.Tensor
        Traceback matrix of dimension N x M
    """
    n_1, m_1, _, _ = Q.shape
    new = Q.new
    N, M = n_1 - 2, m_1 - 2
    m, x, y = 0, 1, 2 # state numbering

    E = new(N + 2, M + 2, 3).zero_()
    # Initial conditions
    E[N + 1, M + 1] = 1
    Q[N + 1, M + 1] = 1  # Is this initialization correct?

    # Backward pass
    for i in reversed(range(1, N + 1)):
        for j in reversed(range(1, M + 1)):
            E[i, j, m] = Q[i + 1, j + 1, m] @ E[i + 1, j + 1]
            E[i, j, x] = Q[i + 1, j, x] @ E[i + 1, j]
            E[i, j, y] = Q[i, j + 1, y] @ E[i, j + 1]
    return E[1:, 1:]


def _adjoint_forward_pass(Q, E, Ztheta, Zpsi, Zphi, operator='softmax'):
    """ Calculate directional derivatives and Hessians.

    Parameters
    ----------
    Q : torch.Tensor
        Derivatives of max theta + v of dimension N x M
    E : torch.Tensor
        Traceback matrix of dimension N x M
    Ztheta : torch.Tensor
        Derivative of theta
    Zpsi : torch.Tensor
        Derivative of psi
    Zphi : torch.Tensor
        Derivative of phi
    operator : str
        The smoothed maximum operator.

    Returns
    -------
    Vd : torch.Tensor
        Derivatives of V of dimension N x M
    Qd : torch.Tensor
        Derivatives of Q of dimension N x M
    """
    operator = operators[operator]
    new = Ztheta.new
    N, M = Ztheta.size()

    Vd = new(N + 1, M + 1, 3).zero_()
    Qd = new(N + 2, M + 2, 3).zero_()
    m, x, y = 0, 1, 2 # state numbering

    # Forward pass
    for i in range(1, M + 1):
        for j in range(1, M + 1):
            Vd[i, j, m] = Ztheta[i - 1, j - 1] + \
                Q[i, j, m] * Vd[i - 1, j - 1, m] + \
                Q[i, j, x] * Vd[i - 1, j - 1, x] + \
                Q[i, j, y] * Vd[i - 1, j - 1, y]
            Vd[i, j, x] = Zpsi[i - 1] + \
                Q[i, j, m] * Vd[i - 1, j, m] + \
                Q[i, j, x] * Vd[i - 1, j, x]
            Vd[i, j, y] = Zphi[j - 1] + \
                Q[i, j, m] * Vd[i, j - 1, m] + \
                Q[i, j, y] * Vd[i, j - 1, y]
            vm = torch.Tensor([Vd[i - 1, j - 1, m], Vd[i - 1, j - 1, x], Vd[i - 1, j - 1, y]])
            vx = torch.Tensor([Vd[i - 1, j, m], Vd[i - 1, j, x]])
            vy = torch.Tensor([Vd[i, j - 1, m], Vd[i, j - 1, y]])
            Qd[i, j, m] = operator.hessian_product(Q[i, j, m], vm)
            Qd[i, j, x] = operator.hessian_product(Q[i, j, x], vx)
            Qd[i, j, y] = operator.hessian_product(Q[i, j, y], vy)
    return Vd, Qd


def _adjoint_backward_pass(Q, Qd, E):
    """ Calculate directional derivatives and Hessians.

    Parameters
    ----------
    Q : torch.Tensor
        Derivatives of max theta + v of dimension N x M
    Qd : torch.Tensor
        Derivatives of Q of dimension N x M
    E : torch.Tensor
        Traceback matrix of dimension N x M

    Returns
    -------
    Ed : torch.Tensor
        Derivative of traceback matrix of dimension N x M.
    """
    N_1, M_1, _ = Vd.shape()
    N, M = N_1 - 1, M_1 - 1
    Ed = new(N + 2, M + 2, 3).zero_()

    # Backward pass
    for i in reversed(range(1, N + 1)):
        for j in reversed(range(1, M + 1)):
            Ed[i, j, m] = \
                Qd[i+1, j+1, m]*E[i+1, j+1, m] + Q[i+1, j+1, m]*Ed[i+1, j+1, m] + \
                Qd[i+1, j+1, x]*E[i+1, j+1, x] + Q[i+1, j+1, x]*Ed[i+1, j+1, x] + \
                Qd[i+1, j+1, y]*E[i+1, j+1, y] + Q[i+1, j+1, y]*Ed[i+1, j+1, y]
            Ed[i, j, x] = \
                Qd[i+1, j, m]*E[i+1, j, m] + Q[i+1, j, m]*Ed[i+1, j, m] + \
                Qd[i+1, j, x]*E[i+1, j, x] + Q[i+1, j, x]*Ed[i+1, j, x]
            Ed[i, j, y] = \
                Qd[i, j+1, m]*E[i, j+1, m] + Q[i, j+1, m]*Ed[i, j+1, m] + \
                Qd[i, j+1, y]*E[i, j+1, y] + Q[i, j+1, y]*Ed[i, j+1, y]
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
