import torch


class AlignmentAccuracy:
    def __call__(self, true_edges, pred_edges):
        pass


class MatrixCrossEntropy:
    def __call__(self, Ytrue, Ypred, x_len, y_len, G):
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
        for b in range(len(x_len)):
            pos = torch.masked_select(
                Ytrue[b, :x_len[b], :y_len[b]] * torch.log(
                    Ypred[b, :x_len[b], :y_len[b]]),
                G[b, :x_len[b], :y_len[b]].bool()
            )

            neg = torch.masked_select(
                (1 - Ytrue[b, :x_len[b], :y_len[b]]) * torch.log(
                    1 - Ypred[b, :x_len[b], :y_len[b]]),
                G[b, :x_len[b], :y_len[b]].bool()
            )
            # pos = Ytrue[b, :x_len[b], :y_len[b]] * torch.log(
            #     Ypred[b, :x_len[b], :y_len[b]])
            # neg = (1 - Ytrue[b, :x_len[b], :y_len[b]]) * torch.log(
            #     1 - Ypred[b, :x_len[b], :y_len[b]])

            score += -torch.mean(pos + neg)

        return score / len(x_len)


class SoftPathLoss:
    def __call__(self, P, Y, x_len, y_len, G):
        """ Computes a soft path loss

        The soft path loss is given by

        d(ypred, ytrue) = frobieus_norm(ypred x path(ytrue))

        where path(ytrue) gives a pairwise distance matrix yielding
        the distance between a point and nearest point in the path
        for every element in the matrix.


        Parameters
        ----------
        P : torch.Tensor
            Pairwise path distances.
        Y : torch.Tensor
            Predicted alignment matrix of dimension N x M.
        """
        score = 0
        for b in range(len(x_len)):
            score += torch.norm(
                torch.masked_select(
                    P[b, :x_len[b], :y_len[b]] * Y[b, :x_len[b], :y_len[b]],
                    G[b, :x_len[b], :y_len[b]].bool()
                )
            )
        return score / len(x_len)


class SoftAlignmentLoss:
    def __call__(self, Yt, Yp, x_len, y_len, G):
        """ Computes soft alignment loss as proposed in Mensch et al.

        The soft alignment loss is given by

        d(ypred, ytrue) = frobieus_norm(Ytrue - Ypred)

        Where L is given by a lower triangular matrix filled with 1.

        Parameters
        ----------
        Yt : torch.Tensor
            Ground truth alignment matrix of dimension N x M.
            All entries are marked by 0 and 1.
        Yp : torch.Tensor
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
        for b in range(len(x_len)):
            score += torch.norm(
                torch.masked_select(
                    Yt[b, :x_len[b], :y_len[b]] - Yp[b, :x_len[b], :y_len[b]],
                    G[b, :x_len[b], :y_len[b]].bool()
                )
            )
        return score / len(x_len)
