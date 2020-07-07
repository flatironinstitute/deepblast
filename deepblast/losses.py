import torch
from torch.nn.utils.rnn import pad_packed_sequence


class AlignmentAccuracy:
    def __call__(self, true_edges, pred_edges):
        pass


class SoftAlignmentLoss:
    def __call__(self, Ytrue, Ypred, x, y):
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
        """
        _, x_len = pad_packed_sequence(x, batch_first=True)
        _, y_len = pad_packed_sequence(y, batch_first=True)
        score = 0
        for b in range(len(x_len)):
            score += torch.norm(
                Ytrue[b, :x_len[b], :y_len[b]] - Ypred[b, :x_len[b], :y_len[b]]
            )
        return score
