import torch


class HardMaxOp:
    @staticmethod
    def max(X):
        M, _ = torch.max(X, dim=1, keepdim=True)
        A = (M == X).float()
        A = A / torch.sum(A, dim=1, keepdim=True)

        return M.squeeze(), A.squeeze()

    @staticmethod
    def hessian_product(P, Z):
        return torch.zeros_like(Z)


class SoftMaxOp:
    @staticmethod
    def max(X):
        M, _ = torch.max(X, dim=1)
        X = X - M.view(-1, 1)
        A = torch.exp(X)
        S = torch.sum(A, dim=1)
        M = M + torch.log(S)
        A /= S.view(-1, 1)
        return M.squeeze(), A.squeeze()

    @staticmethod
    def hessian_product(P, Z):
        prod = P * Z
        return prod - P * torch.sum(prod, dim=1, keepdim=True)


operators = {'softmax': SoftMaxOp, 'hardmax': HardMaxOp}

