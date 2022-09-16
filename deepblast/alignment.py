import torch
import torch.nn as nn
# from deepblast.language_model import BiLM, pretrained_language_models
#from deepblast.nw_cuda import NeedlemanWunschDecoder as NWDecoderCUDA
from deepblast.nw import NeedlemanWunschDecoder as NWDecoderCUDA
from deepblast.embedding import StackedRNN, EmbedLinear
from deepblast.dataset.utils import unpack_sequences
import torch.nn.functional as F


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
        lm : ESM2
           Pretrained language model (optional)
        padding_idx : int
           Location of padding index in embedding (default -1)
        transform : function
           Activation function (default relu)
        sparse : False?
        """
        super(NeedlemanWunschAligner, self).__init__()
        assert lm is not None
        self.lm = lm
        n_embed = self.lm.hidden_size
        self.match_embedding = nn.Linear(n_embed, n_embed)
        self.gap_embedding = nn.Linear(n_embed, n_embed)

        # TODO: make cpu compatible version
        # if device == 'cpu':
        #     self.nw = NWDecoderCPU(operator='softmax')
        # else:
        self.nw = NWDecoderCUDA(operator='softmax')

    def blosum_factor(self, x):
        """ Computes factors for blosum parameters using a single sequence

        Parameters
        ----------
        x : torch.Tensor
           Representation of a single protein sequence from ESM
           with dimensions B x (N + 2) x D
        """
        hx = self.lm.encode(x)
        zx = self.match_embedding(hx)
        gx = self.gap_embedding(hx)
        return zx, gx

    def forward(self, x, order):
        """ Generate alignment matrix.

        Parameters
        ----------
        x : PackedSequence
            Packed sequence object of proteins to align.
        order : np.array
            The origin order of the sequences

        Returns
        -------
        aln : torch.Tensor
            Alignment Matrix (dim B x N x M)
        """
        with torch.enable_grad():
            hx, _, hy, _ = unpack_sequences(x, order)
            zx, gx = self.blosum_factor(hx)
            zy, gy = self.blosum_factor(hy)

            # Obtain theta through an inner product across latent dimensions
            theta = F.softplus(torch.einsum('bid,bjd->bij', zx, zy))
            A = F.logsigmoid(torch.einsum('bid,bjd->bij', gx, gy))
            aln = self.nw.decode(theta, A)
            return aln, theta, A

    def score(self, x, order):
        with torch.no_grad():
            hx, _, hy, _ = unpack_sequences(x, order)
            zx, gx = self.blosum_factor(hx)
            zy, gy = self.blosum_factor(hy)
            # Obtain theta through an inner product across latent dimensions
            theta = F.softplus(torch.einsum('bid,bjd->bij', zx, zy))
            A = F.logsigmoid(torch.einsum('bid,bjd->bij', gx, gy))
            ascore = self.nw(theta, A)
            return ascore

    def traceback(self, x, order):
        """ Generate alignment matrix.

        Parameters
        ----------
        x : PackedSequence
            Packed sequence object of proteins to align.
        order : torch.Tensor
            The origin order of the sequences

        Returns
        -------
        decoded : list of tuple
            State string representing alignment coordinates
        aln : torch.Tensor
            Alignment Matrix (dim B x N x M)
        """
        # dim B x N x D
        with torch.enable_grad():

            hx, xlen, hy, ylen = unpack_sequences(x, order)
            zx, gx = self.blosum_factor(hx)
            zy, gy = self.blosum_factor(hy)

            match = F.softplus(torch.einsum('bid,bjd->bij', zx, zy))
            gap = F.logsigmoid(torch.einsum('bid,bjd->bij', gx, gy))
            B, _, _ = match.shape
            for b in range(B):
                aln = self.nw.decode(
                    match[b, :xlen[b], :ylen[b]].unsqueeze(0),
                    gap[b, :xlen[b], :ylen[b]].unsqueeze(0)
                ).squeeze()
                # Ignore the start/end tokens
                decoded = self.nw.traceback(aln[1:-1, 1:-1])
                yield decoded, aln
