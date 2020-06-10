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
    N, M = theta.size()

    # Initialize the matrices of interest. The 3rd axis here represents the
    # states that corresponds to (0) match, (1) gaps in X and (2) gaps in Y.
    V = torch.zeros(N + 1, M + 1, 3)    # N x M x S
    Q = torch.zeros(N + 2, M + 2, 3, 3) # N x M x S x S

    m, x, y = 0, 1, 2 # state numbering
    e = 1e-10
    # Forward pass
    for i in range(1, N):
        for j in range(1, M):
            V[i, j, m], Q[i, j, m] = operator.max(
                theta[i, j] + A[m] + V[i-1, j-1]
            )
            V[i, j, m], Q[i, j, x] = operator.max(
                psi[i] + A[x] + V[i-1, j]
            )
            V[i, j, m], Q[i, j, m] = operator.max(
                phi[j] + A[y] + V[i, j-1]
            )
    # Compute terminal score
    Vt, Qt = operator.max(V[N, M])
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
    m, x, y = 0, 1, 2 # state numbering
    E = new(N + 2, M + 2, 3).zero_()
    # Initial conditions
    E[N + 1, M + 1] = Qt @ Et  # these dimensions look weird
    Q[N + 1, M + 1] = 1        # this may also be wrong
    # Backward pass
    for i in reversed(range(1, N + 1)):
        for j in reversed(range(1, M + 1)):
            E[i, j, m] = Q[i + 1, j + 1, m] @ E[i + 1, j + 1]
            E[i, j, x] = Q[i + 1, j, x] @ E[i + 1, j]
            E[i, j, y] = Q[i, j + 1, y] @ E[i, j + 1]
    return E


def _adjoint_forward_pass(Q, Ztheta, Zpsi, Zphi, ZA, operator='softmax'):
    """ Calculate directional derivatives and Hessians.

    Parameters
    ----------
    Q : torch.Tensor
        Derivatives of max theta + v of dimension N x M x S x S
    E : torch.Tensor
        Traceback matrix of dimension N x M x S
    Ztheta : torch.Tensor
        Derivative of theta of dimension N x M
    Zpsi : torch.Tensor
        Derivative of psi of dimension N
    Zphi : torch.Tensor
        Derivative of phi of dimension M
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
    operator = operators[operator]
    new = Ztheta.new
    N, M = Ztheta.size()
    Vd = new(N + 1, M + 1, 3).zero_()
    Vtd = new(3).zero_()
    Qd = new(N + 2, M + 2, 3, 3).zero_()
    Qtd = new(N + 2, M + 2, 3).zero_()
    m, x, y = 0, 1, 2 # state numbering
    # Forward pass
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            vm = Vd[i - 1, j - 1] + ZA[m]
            vx = Vd[i - 1, j] + ZA[x]
            vy = Vd[i, j - 1] + ZA[y]
            # nullify illegal actions (TODO make unittest for this)
            # vx[y] = 0 vy[x] = 0
            Vd[i, j, m] = Ztheta[i - 1, j - 1] + Q[i, j, m] @ vm
            Vd[i, j, x] = Zpsi[i - 1] + Q[i, j, x] @ vx
            Vd[i, j, y] = Zphi[j - 1] + Q[i, j, y] @ vy
            Qd[i, j, m] = operator.hessian_product(Q[i, j, m], vm)
            Qd[i, j, x] = operator.hessian_product(Q[i, j, x], vx)
            Qd[i, j, y] = operator.hessian_product(Q[i, j, y], vy)
    # Compute terminal score
    Vtd = Q[N, M, m] @ Vd[N, M] + Q[N, M, x] @ Vd[N, M] + Q[N, M, y] @ Vd[N, M]
    vtm = Vd[N - 1, M - 1]
    vtx = torch.Tensor([Vd[N - 1, M, m], Vd[N - 1, M, x], 0])
    vty = torch.Tensor([Vd[N, M - 1, m], 0, Vd[N, M - 1, y]])
    Qtd[N, M, m] = operator.hessian_product(Qt[N, M, m], vm)
    Qtd[N, M, x] = operator.hessian_product(Qt[N, M, x], vx)
    Qtd[N, M, y] = operator.hessian_product(Qt[N, M, y], vy)
    return Vtd, Qtd, Q


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
        inputs = (theta, psi, phi, A)
        Vt, Qt, Q = _forward_pass(theta, psi, phi, A, operator)
        ctx.save_for_backward(theta, psi, phi, Qt, Q)
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
           Derivative of Vt (essentially an arg max) of dimension S.
        """
        theta, psi, phi, Q = ctx.saved_tensors
        operator = ctx.others
        E = ViterbiFunctionBackward.apply(
            theta, psi, phi, Et, Qt, Q, operator)
        return E[1:-1, 1:-1]


class ViterbiFunctionBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, theta, psi, phi, Et, Qt, Q, operator):
        E = _backward_pass(Et, Qt, Q)
        ctx.save_for_backward(Et, E, Qt, Q)
        ctx.others = operator
        return E

    @staticmethod
    def backward(ctx, Ztheta, Zphi, Zpsi):
        """
        Parameters
        ----------
        ctx : ?
           Some autograd context object
        Ztheta : torch.Tensor
            Derivative of theta of dimension N x M
        Zpsi : torch.Tensor
            Derivative of psi of dimension N
        Zphi : torch.Tensor
            Derivative of phi of dimension M
        """
        Et, E, Qt, Q = ctx.saved_tensors
        operator = ctx.others
        Vd, Qd = _adjoint_forward_pass(Q, E, Ztheta, Zpsi, Zphi, operator)
        Ed = _adjoint_backward_pass(Q, Qd, E)
        return Ed, Vd, None


class ViterbiDecoder(nn.Module):

    def __init__(self, operator):
        super().__init__()
        self.operator = operator

    def forward(self, theta, psi, phi, A):
        return ViterbiFunction.apply(
            theta, psi, phi, A, self.operator)

    def decode(self, theta, psi, phi, A):
        """ Shortcut for doing inference. """
        # data, batch_sizes = theta
        with torch.enable_grad():
            # data.requires_grad_()
            nll = self.forward(theta, psi, phi, A)
            v = torch.sum(nll)
            v_grad, = torch.autograd.grad(
                v, (theta.data, psi.data, phi.data,),
                create_graph=True)
        return v_grad
