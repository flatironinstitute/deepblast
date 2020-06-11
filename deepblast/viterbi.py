import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence


class HardMaxOp:
    @staticmethod
    def max(X):
        M, _ = torch.max(X)
        A = (M == X).float()
        A = A / torch.sum(A)

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
        return prod - P * torch.sum(prod)


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
        support = torch.sum(S)
        prod = S * Z
        return prod - S * torch.sum(prod) / support


operators = {'softmax': SoftMaxOp, 'sparsemax': SparseMaxOp,
             'hardmax': HardMaxOp}


def _forward_pass(theta, A, operator='softmax'):
    """  Forward pass to calculate DP alignment matrix

    Parameters
    ----------
    theta : torch.Tensor
        Input potentials of dimension N x M x S. This represents the
        pairwise residue match and gap scores.
    A : torch.Tensor
        3x3 matrix of transition log probabilities.
        log([[1-2d, d, d],
             [1-e, e, 0],
             [1-e, 0, e]])
        This has to be specified by the user.
    operator : str
        The smoothed maximum operator.

    Returns
    -------
    Vt : torch.Tensor
        Terminal alignment score (just 1 dimension)
    Qt : torch.Tensor
        Terminal alignment tracebacks of dimension S
    Q : torch.Tensor
        Derivatives of max theta + v of dimension N x M x S x S.
    """
    operator = operators[operator]
    new = theta.new
    N, M, _ = theta.size()
    m, x, y = 1, 0, 2              # state numbering
    # Note that theta is indexed from zero, whereas V is indexed from 1.
    # Initialize the matrices of interest. The 3rd axis here represents the
    # states that corresponds to (0) match, (1) gaps in X and (2) gaps in Y.
    V = new(N + 1, M + 1, 3).zero_()     # N x M x S
    Q = new(N + 2, M + 2, 3, 3).zero_()  # N x M x S x S
    V[0, 0, m] = 1
    # Forward pass
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            V[i, j, m], Q[i, j, m] = operator.max(A[m] + V[i-1, j-1])
            V[i, j, x], Q[i, j, x] = operator.max(A[x] + V[i-1, j])
            V[i, j, y], Q[i, j, y] = operator.max(A[y] + V[i, j-1])
            V[i, j, m] += theta[i-1, j-1, m]
            V[i, j, x] += theta[i-1, j-1, x]
            V[i, j, y] += theta[i-1, j-1, y]
    # Compute terminal score
    Vt, Qt = operator.max(V[N, M])
    # Q[N + 1, M + 1, m, m] = 1
    return Vt, Qt, Q


def _backward_pass(Et, Qt, Q):
    """ Backward pass to calculate grad DP

    Parameters
    ----------
    Et : torch.Tensor
        Terminal alignment edges of dimension S.
    Qt : torch.Tensor
        Terminal alignment traceback vector of dimension S.
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
    m, x, y = 1, 0, 2 # state numbering
    E = new(N + 2, M + 2, 3).zero_()
    # Initial conditions
    E[N, M] = Et * Qt

    # Backward pass
    for i in reversed(range(1, N)):
        for j in reversed(range(1, M)):
            # Beware, there is an indexing issue with Q and E
            E[i, j, m] = Q[i, j, m] @ E[i + 1, j + 1]
            E[i, j, x] = Q[i + 1, j, x] @ E[i + 1, j]
            E[i, j, y] = Q[i, j + 1, y] @ E[i, j + 1]
    return E


def _adjoint_forward_pass(Qt, Q, Ztheta, ZA, operator='softmax'):
    """ Calculate directional derivatives and Hessians.

    Parameters
    ----------
    Qt : torch.Tensor
        Terminal Derivatives of max theta + v of dimension S
    Q : torch.Tensor
        Derivatives of max theta + v of dimension N x M x S x S
    Ztheta : torch.Tensor
        Derivative of theta of dimension N x M
    ZA : torch.Tensor
        Derivative of log transition probabilities of dimension S x S.
    operator : str
        The smoothed maximum operator.

    Returns
    -------
    Vd : torch.Tensor
        Derivatives of V of dimension N x M x S
    Qd : torch.Tensor
        Derivatives of Q of dimension N x M x S x S
    """
    m, x, y = 1, 0, 2 # state numbering
    operator = operators[operator]
    N, M, _, _ = Q.size()
    N, M = N - 2, M - 2
    new = Ztheta.new
    Vd = new(N + 1, M + 1, 3).zero_()
    Vtd = new(3).zero_()
    Qd = new(N + 2, M + 2, 3, 3).zero_()
    Qtd = new(3, 3).zero_()
    # Forward pass
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            vm = Vd[i - 1, j - 1] + ZA[m]
            vx = Vd[i - 1, j] + ZA[x]
            vy = Vd[i, j - 1] + ZA[y]
            # nullify illegal actions (TODO make unittest for this)
            # vx[y] = 0 vy[x] = 0
            Vd[i, j, m] = Ztheta[i - 1, j - 1, m] + Q[i, j, m] @ vm
            Vd[i, j, x] = Ztheta[i - 1, j, x] + Q[i, j, x] @ vx
            Vd[i, j, y] = Ztheta[i, j - 1, y] + Q[i, j, y] @ vy
            Qd[i, j, m] = operator.hessian_product(Q[i, j, m], vm)
            Qd[i, j, x] = operator.hessian_product(Q[i, j, x], vx)
            Qd[i, j, y] = operator.hessian_product(Q[i, j, y], vy)
    # Compute terminal score
    Vtd = Q[N, M] @ Vd[N, M]
    vtm = Vd[N - 1, M - 1]
    vtx = torch.Tensor([Vd[N - 1, M, x], Vd[N - 1, M, m], 0])
    vty = torch.Tensor([0, Vd[N, M - 1, m], Vd[N, M - 1, y]])
    Qtd[m] = operator.hessian_product(Qt[m], vm)
    Qtd[x] = operator.hessian_product(Qt[x], vx)
    Qtd[y] = operator.hessian_product(Qt[y], vy)
    return Vtd, Qtd, Qd


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
    m, x, y = 1, 0, 2 # state numbering
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
        inputs = (theta, A)
        Vt, Qt, Q = _forward_pass(theta, A, operator)
        ctx.save_for_backward(theta, A, Qt, Q)
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
        theta, A, Qt, Q = ctx.saved_tensors
        operator = ctx.others
        E, A = ViterbiFunctionBackward.apply(
            theta, A, Et, Qt, Q, operator)
        return E[1:-1, 1:-1], A, None, None, None


class ViterbiFunctionBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, theta, A, Et, Qt, Q, operator):
        E = _backward_pass(Et, Qt, Q)
        ctx.save_for_backward(Et, E, Qt, Q)
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


class ViterbiDecoder(nn.Module):

    def __init__(self, operator):
        super().__init__()
        self.operator = operator

    def forward(self, theta, A):
        return ViterbiFunction.apply(
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
