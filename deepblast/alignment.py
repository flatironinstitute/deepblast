import torch
import torch.nn as nn
from deepblast.language_model import BiLM, pretrained_language_models
from deepblast.nw import NeedlemanWunschDecoder
from deepblast.embedding import StackedRNN, EmbedLinear


class NeedlemanWunschAligner(nn.Module):

    def __init__(self, n_alpha, n_input, n_units, n_embed,
                 n_layers=2, lm=None):
        """ NeedlemanWunsch Alignment model

        Parameters
        ----------
        n_alpha : int
           Size of the alphabet (default 22)
        n_input : int
           Input dimensions.
        n_units : int
           Number of hidden units in RNN.
        n_embed : int
           Embedding dimension
        n_layers : int
           Number of RNN layers.
        lm : BiLM
           Pretrained language model (optional)
        padding_idx : int
           Location of padding index in embedding (default -1)
        transform : function
           Activation function (default relu)
        sparse : False?
        """
        super(NeedlemanWunschAligner, self).__init__()
        if lm is None:
            path = pretrained_language_models['bilstm']
            self.lm = BiLM()
            self.lm.load_state_dict(torch.load(path))
            self.lm.eval()
        if n_layers > 1:
            self.embedding = StackedRNN(
                n_alpha, n_input, n_units, n_embed, n_layers, lm=lm)
        else:
            self.embedding = EmbedLinear(n_alpha, n_input, n_embed, lm=lm)

        self.gap_score = nn.Linear(n_embed * 2, 1)
        self.nw = NeedlemanWunschDecoder(operator='softmax')

    def forward(self, x, y):
        """ Generate alignment matrix.

        Parameters
        ----------
        x : torch.Tensor
            Tokens for sequence x (dim B x N)
        y : torch.Tensor
            Tokens for sequence y (dim B x M)

        Returns
        -------
        aln : torch.Tensor
            Alignment Matrix (dim B x N x M)
        """
        with torch.enable_grad():
            zx = self.embedding(x)    # dim B x N x D
            zy = self.embedding(y)    # dim B x M x D
            # Obtain theta through an inner product across latent dimensions
            theta = torch.einsum('bid,bjd->bij', zx, zy)
            xmean = zx.mean(axis=1)   # dim B x D
            ymean = zy.mean(axis=1)   # dim B x D
            merged = torch.cat((xmean, ymean), axis=1)  # dim B x 2D

            A = self.gap_score(merged)
            # TODO enable batching on needleman-wunsch
            B, N, M = theta.shape
            aln = torch.zeros((B, M, N))
            for b in range(B):
                aln[b] = self.nw.decode(theta[b], A[b]).T
            return aln

    def traceback(self, x, y):
        zx = self.embedding(x)    # dim B x N x D
        zy = self.embedding(y)    # dim B x M x D
        # Obtain theta through an inner product across latent dimensions
        theta = torch.einsum('bij,bjk->bij', zx, zy)
        xmean = zx.mean(axis=1)   # dim B x D
        ymean = zy.mean(axis=1)   # dim B x D
        merged = torch.concat((xmean, ymean), axis=1)  # dim B x 2D
        A = self.gap_score(merged)
        aln = self.nw(theta, A)
        aln.backward()
        decoded = self.nw.traceback(theta.grad)
        return decoded
