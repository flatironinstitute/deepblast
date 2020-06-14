import torch
import torch.nn as nn
import torch.nn.functional as F
from deepblast.language_model import BiLM, pretrained_language_models
from deepblast.nw import NeedlemanWunschDecoder
from deepblast.embedding import StackedRNN


class NeedlemanWunschAligner(nn.Module):

    def __init__(self, nin, nembed, nunits, nout, nlayers=2, lm=None):
        """
        Parameters
        ----------
        nin : int
           Input dimensions (default 22)
        nembed : int
           Number of embedding dimensions
        nunits : int
           Number of hidden units in RNN.
        nout : int
           Output dimensions (default 22)
        nlayers : int
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
        self.embedding = StackedRNN(nin, nembed, nunits, nout, nlayers, lm)
        self.gap_score = nn.Linear(nembed * 2, 1)
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
        zx = self.embedding(x)    # dim B x N x D
        zy = self.embedding(y)    # dim B x M x D
        # Obtain theta through an inner product across latent dimensions
        theta = torch.einsum('bid,bjd->bij', zx, zy)
        xmean = zx.mean(axis=1)   # dim B x D
        ymean = zy.mean(axis=1)   # dim B x D
        merged = torch.cat((xmean, ymean), axis=1) # dim B x 2D
        A = self.gap_score(merged)
        aln = self.nw(theta, A)
        return aln

    def traceback(self, x, y):
        zx = self.embedding(x)    # dim B x N x D
        zy = self.embedding(y)    # dim B x M x D
        # Obtain theta through an inner product across latent dimensions
        theta = torch.einsum('bij,bjk->bij', zx, zy)
        xmean = zx.mean(axis=1)   # dim B x D
        ymean = zy.mean(axis=1)   # dim B x D
        merged = torch.concat((xmean, ymean), axis=1) # dim B x 2D
        A = self.gap_score(merged)
        aln = self.nw(theta, A)
        aln.backward()
        decoded = self.nw.traceback(theta.grad)
        return decoded
