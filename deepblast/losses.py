import torch


class AlignmentAccuracy:
    def __call__(self, true_edges, pred_edges):
        pass


class SoftAlignmentLoss:
    def __call__(self, Ytrue, Ypred):
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
        print(Ytrue.shape, Ypred.shape)
        Ytrue = Ytrue[:, 1:-1, 1:-1].cuda()
        Ypred = Ypred.cuda()
        return torch.norm(Ytrue[:, 1:-1, 1:-1] - Ypred)
