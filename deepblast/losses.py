import torch
from torch.distributions import Normal
from deepblast.constants import match_mean, match_std, gap_mean, gap_std


def mask_tensor(A, x_mask, y_mask):
    return A[x_mask][:, y_mask]


class AlignmentAccuracy:
    def __call__(self, true_edges, pred_edges):
        pass


class L2MatrixCrossEntropy:
    def __call__(self, Ytrue, Ypred, M, G, x_mask, y_mask):
        """ Computes binary cross entropy on the matrix with regularizers.

        The matrix cross entropy loss is given by

        d(ypred, ytrue) = - (mean(ytrue x log(ypred))
             + mean((1 - ytrue) x log(1 - ypred)))

        Parameters
        ----------
        Ytrue : torch.Tensor
            Ground truth alignment matrix of dimension N x M.
            All entries are marked by 0 and 1.
        Ypred : torch.Tensor
            Predicted alignment matrix of dimension N x M.
        M : torch.Tensor
            Match score matrix
        G : torch.Tensor
            Gap score matrix
        """
        score = 0
        eps = 3e-8   # unfortunately, this is the smallest eps we can have :(
        Ypred = torch.clamp(Ypred, min=eps, max=1 - eps)
        for b in range(len(x_mask)):
            pos = torch.mean(
                mask_tensor(Ytrue[b], x_mask[b], y_mask[b]) * torch.log(
                    mask_tensor(Ypred[b], x_mask[b], y_mask[b]))
            )
            neg = torch.mean(
                (1 - mask_tensor(Ytrue[b], x_mask[b], y_mask[b])) * torch.log(
                    1 - mask_tensor(Ypred[b], x_mask[b], y_mask[b]))
            )
            score += -(pos + neg)

        match_prior = Normal(match_mean, match_std)
        gap_prior = Normal(gap_mean, gap_std)
        log_like = score / len(x_mask)
        match_log = match_prior.log_prob(M).mean()
        gap_log = gap_prior.log_prob(G).mean()
        score = log_like + match_log + gap_log
        return score


class MatrixCrossEntropy:
    def __call__(self, Ytrue, Ypred, x_mask, y_mask):
        """ Computes binary cross entropy on the matrix

        The matrix cross entropy loss is given by

        d(ypred, ytrue) = - (mean(ytrue x log(ypred))
             + mean((1 - ytrue) x log(1 - ypred)))

        Parameters
        ----------
        Ytrue : torch.Tensor
            Ground truth alignment matrix of dimension N x M.
            All entries are marked by 0 and 1.
        Ypred : torch.Tensor
            Predicted alignment matrix of dimension N x M.
        """
        score = 0
        eps = 3e-8   # unfortunately, this is the smallest eps we can have :(
        Ypred = torch.clamp(Ypred, min=eps, max=1 - eps)
        for b in range(len(x_mask)):
            pos = torch.mean(
                mask_tensor(Ytrue[b], x_mask[b], y_mask[b]) * torch.log(
                    mask_tensor(Ypred[b], x_mask[b], y_mask[b]))
            )
            neg = torch.mean(
                (1 - mask_tensor(Ytrue[b], x_mask[b], y_mask[b])) * torch.log(
                    1 - mask_tensor(Ypred[b], x_mask[b], y_mask[b]))
            )
            # pos = torch.mean(
            #     Ytrue[b, x_mask[b], y_mask[b]] * torch.log(
            #         Ypred[b, x_mask[b], y_mask[b]])
            # )
            # neg = torch.mean(
            #     (1 - Ytrue[b, x_mask[b], y_mask[b]]) * torch.log(
            #         1 - Ypred[b, x_mask[b], y_mask[b]])
            # f)
            score += -(pos + neg)
        return score / len(x_mask)


class SoftPathLoss:
    def __call__(self, Pdist, Ypred, x_mask, y_mask):
        """ Computes a soft path loss

        The soft path loss is given by

        d(ypred, ytrue) = frobieus_norm(ypred x path(ytrue))

        where path(ytrue) gives a pairwise distance matrix yielding
        the distance between a point and nearest point in the path
        for every element in the matrix.


        Parameters
        ----------
        Pdist : torch.Tensor
            Pairwise path distances.
        Ypred : torch.Tensor
            Predicted alignment matrix of dimension N x M.
        """
        score = 0
        for b in range(len(x_mask)):
            score += torch.norm(
                Pdist[b, x_mask[b], y_mask[b]] * Ypred[b, x_mask[b], y_mask[b]]
            )
        return score / len(x_mask)


class SoftAlignmentLoss:
    def __call__(self, Ytrue, Ypred, x_mask, y_mask):
        """ Computes soft alignment loss as proposed in Mensch et al.

        The soft alignment loss is given by

        d(ypred, ytrue) = frobieus_norm(Ytrue - Ypred)

        Where L is given by a lower triangular matrix filled with 1.

        Parameters
        ----------
        Ytrue : torch.Tensor
            Ground truth alignment matrix of dimension N x M.
            All entries are marked by 0 and 1.
        Ypred : torch.Tensor
            Predicted alignment matrix of dimension N x M.

        Returns
        -------
        loss : torch.Tensor
            Scalar valued loss.

        Notes
        -----
        We aren't extracting the lower triangular matrix here,
        since it is possible to leave out important parts of the alignment.
        """
        score = 0
        for b in range(len(x_mask)):
            score += torch.norm(
                Ytrue[b, x_mask[b], y_mask[b]] - Ypred[b, x_mask[b], y_mask[b]]
            )
        return score / len(x_mask)
