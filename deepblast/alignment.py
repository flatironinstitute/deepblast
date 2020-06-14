import torch
import torch.nn as nn
import torch.nn.functional as F
from deepblast.language_model import BiLM, pretrained_language_models
from deepblast.nw import NeedlemanWunschDecoder
from deepblast.embedding import StackedRNN


class AlignmentModel(nn.Module):

    def __init__(self, **embedding_args):
        if 'lm' not in embedding_args:
            path = pretrained_language_models['bilstm']
            self.lm = BiLM(**lm_args)
            self.lm.load_state_dict(torch.load(path))
            self.lm.eval()
            embedding_args.update('lm', lm)
        self.embedding = StackedRNN(**embedding_args)
        nembed = embedding_args.pop('nembed')
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
        theta = torch.einsum('bij,bjk->bij', zx, zy)
        xmean = zx.mean(axis=1)   # dim B x D
        ymean = zy.mean(axis=1)   # dim B x D
        merged = torch.concat((xmean, ymean), axis=1) # dim B x 2D
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
