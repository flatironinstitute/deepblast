import torch
import torch.nn as nn
from deepblast.language_model import BiLM, pretrained_language_models
from deepblast.nw_cuda import NeedlemanWunschDecoder as NWDecoderCUDA
from deepblast.embedding import StackedRNN, EmbedLinear
from deepblast.dataset.utils import unpack_sequences
import torch.nn.functional as F


class NeedlemanWunschAligner(nn.Module):

    def __init__(self, n_alpha, n_input, n_units, n_embed,
                 n_layers=2, lm=None, device='gpu'):
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
            self.match_embedding = StackedRNN(
                n_alpha, n_input, n_units, n_embed, n_layers, lm=lm)
            self.gap_embedding = StackedRNN(
                n_alpha, n_input, n_units, n_embed, n_layers, lm=lm)
        else:
            self.match_embedding = EmbedLinear(
                n_alpha, n_input, n_embed, lm=lm)
            self.gap_embedding = EmbedLinear(
                n_alpha, n_input, n_embed, lm=lm)

        # TODO: make cpu compatible version
        # if device == 'cpu':
        #     self.nw = NWDecoderCPU(operator='softmax')
        # else:
        self.nw = NWDecoderCUDA(operator='softmax')

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
            zx, _, zy, _ = unpack_sequences(self.match_embedding(x), order)
            gx, _, gy, _ = unpack_sequences(self.gap_embedding(x), order)

            # Obtain theta through an inner product across latent dimensions
            theta = F.softplus(torch.einsum('bid,bjd->bij', zx, zy))
            A = F.logsigmoid(torch.einsum('bid,bjd->bij', gx, gy))
            aln = self.nw.decode(theta, A)
            return aln, theta, A

    def score(self, x, order):
        with torch.no_grad():
            zx, _, zy, _ = unpack_sequences(self.match_embedding(x), order)
            gx, _, gy, _ = unpack_sequences(self.gap_embedding(x), order)

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
        order : np.array
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
            zx, _, zy, _ = unpack_sequences(self.match_embedding(x), order)
            gx, xlen, gy, ylen = unpack_sequences(self.gap_embedding(x), order)
            match = F.softplus(torch.einsum('bid,bjd->bij', zx, zy))
            gap = F.logsigmoid(torch.einsum('bid,bjd->bij', gx, gy))
            B, _, _ = match.shape
            for b in range(B):
                aln = self.nw.decode(
                    match[b, :xlen[b], :ylen[b]].unsqueeze(0),
                    gap[b, :xlen[b], :ylen[b]].unsqueeze(0)
                )
                decoded = self.nw.traceback(aln.squeeze())
                yield decoded, aln
