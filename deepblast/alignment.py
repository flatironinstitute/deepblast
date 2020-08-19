import torch
import torch.nn as nn
from deepblast.language_model import BiLM, pretrained_language_models
from deepblast.nw_cuda import NeedlemanWunschDecoder as NWDecoderCUDA
from deepblast.embedding import StackedRNN, EmbedLinear, MultiLinear
from deepblast.dataset.utils import unpack_sequences
import torch.nn.functional as F
import math


class NeedlemanWunschAligner(nn.Module):

    def __init__(self, n_alpha, n_input, n_units, n_embed,
                 n_layers=2, n_heads=16, lm=None, device='gpu', local=True):
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
        n_heads : int
           Number of heads in multilinear layer.
        lm : BiLM
           Pretrained language model (optional)
        padding_idx : int
           Location of padding index in embedding (default -1)
        transform : function
           Activation function (default relu)
        sparse : False?
        local : bool
           Specifies if local alignment should be performed on the traceback
        """
        super(NeedlemanWunschAligner, self).__init__()
        if lm is None:
            path = pretrained_language_models['bilstm']
            self.lm = BiLM()
            self.lm.load_state_dict(torch.load(path))
            self.lm.eval()
        if n_layers > 1:
            self.match_embedding = StackedRNN(
                n_alpha, n_input, n_units, n_embed, n_layers, lm=lm)
            self.gap_embedding = StackedRNN(
                n_alpha, n_input, n_units, n_embed, n_layers, lm=lm)
        else:
            self.match_embedding = EmbedLinear(
                n_alpha, n_input, n_embed, lm=lm)
            self.gap_embedding = EmbedLinear(
                n_alpha, n_input, n_embed, lm=lm)
        self.match_mixture = MultiLinear(n_input, n_input, n_heads)
        self.gap_mixture = MultiLinear(n_input, n_input, n_heads)
        # TODO: make cpu compatible version
        # if device == 'cpu':
        #     self.nw = NWDecoderCPU(operator='softmax')
        # else:
        self.nw = NWDecoderCUDA(operator='softmax')
        self.local = local


    def forward(self, x, order):
        """ Generate alignment matrix.

        Parameters
        ----------
        x : PackedSequence
            Packed sequence object of proteins to align.

        Returns
        -------
        aln : torch.Tensor
            Alignment Matrix (dim B x N x M)
        """
        with torch.enable_grad():
            zx_, _, zy_, _ = unpack_sequences(self.match_embedding(x), order)
            gx_, _, gy_, _ = unpack_sequences(self.gap_embedding(x), order)
            zx, zy = match_mixture(zx_), match_mixture(zy_)
            gx, gy = gap_mixture(zx_), gap_mixture(zy_)

            # Obtain theta through an inner product across latent dimensions
            theta = torch.einsum('bid,bjd->bij', zx, zy)
            A = torch.einsum('bid,bjd->bij', gx, gy)
            aln = self.nw.decode(theta, A)
            return aln, theta, A

    def traceback(self, x, order):
        # dim B x N x D
        with torch.enable_grad():
            zx_, _, zy_, _ = unpack_sequences(self.match_embedding(x), order)
            gx_, xlen, gy_, ylen = unpack_sequences(self.gap_embedding(x), order)
            zx, zy = match_mixture(zx_), match_mixture(zy_)
            gx, gy = gap_mixture(zx_), gap_mixture(zy_)

            match = torch.einsum('bid,bjd->bij', zx, zy)
            gap = torch.einsum('bid,bjd->bij', gx, gy)
            B, _, _ = match.shape

            for b in range(B):
                M = match[b, :xlen[b], :ylen[b]].unsqueeze(0)
                G = gap[b, :xlen[b], :ylen[b]].unsqueeze(0)
                val = math.log(1 - (1/50))  # based on average insertion length
                if self.local:
                    G[0, 0, :] = val
                    G[0, :, 0] = val
                aln = self.nw.decode(M, G)
                decoded = self.nw.traceback(aln.squeeze())
                del G, aln
                yield decoded
