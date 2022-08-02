import torch


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

        tau = (
            torch.gather(
                cssv, dim=1, index=rho[:, None] - 1
            )[:, 0] / rho.type(X.type()))
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
