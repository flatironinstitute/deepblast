import torch
import torch.nn as nn
from deepblast.language_model import BiLM, pretrained_language_models
from deepblast.nw_cuda import NeedlemanWunschDecoder as NWDecoderCUDA
from deepblast.viterbi_cuda import ViterbiDecoder as ViterbiDecoderCUDA
from deepblast.embedding import StackedRNN, EmbedLinear
from deepblast.dataset.utils import unpack_sequences
from deepblast.constants import pos_mxy
import torch.nn.functional as F


class BaseAligner(nn.Module):
    def __init__(self, n_alpha, n_input, n_units, n_embed,
                 n_layers=2, lm=None, device='gpu'):
        """ Base Alignment model

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
        super(BaseAligner, self).__init__()
        if lm is None:
            path = pretrained_language_models['bilstm']
            self.lm = BiLM()
            self.lm.load_state_dict(torch.load(path))
            self.lm.eval()

    def contract(self, x, order):
        """ Tensor contraction operation. """
        zx, _, zy, _ = unpack_sequences(self.match_embedding(x), order)
        gx, _, gy, _ = unpack_sequences(self.gap_embedding(x), order)

        # Obtain theta through an inner product across latent dimensions
        theta = F.softplus(torch.einsum('bid,bjd->bij', zx, zy))
        A = F.logsigmoid(torch.einsum('bid,bjd->bij', gx, gy))
        return theta, A


class NeedlemanWunschAligner(BaseAligner):

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
        super(NeedlemanWunschAligner, self).__init__(
            n_alpha, n_input, n_units, n_embed,
            n_layers=2, lm=None, device='gpu'
        )
        # TODO: make cpu compatible version
        # if device == 'cpu':
        #     self.nw = NWDecoderCPU(operator='softmax')
        # else:
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
            theta, A = self.contract(x, order)
            aln = self.nw.decode(theta, A)
            return aln, theta, A

    def score(self, x, order):
        with torch.no_grad():
            theta, A = self.contract(x, order)
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
            match, gap = self.contract(x, order)
            B, _, _ = match.shape
            for b in range(B):
                aln = self.nw.decode(
                    match[b, :xlen[b], :ylen[b]].unsqueeze(0),
                    gap[b, :xlen[b], :ylen[b]].unsqueeze(0)
                )
                decoded = self.nw.traceback(aln.squeeze())
                yield decoded, aln


class HMM3Aligner(BaseAligner):
    """ 3 state HMM alignment algorithm for affine gap alignment. """
    def __init__(self, n_alpha, n_input, n_units, n_embed,
                 n_layers=2, lm=None, device='gpu'):
        super(HMM3Aligner, self).__init__(
            n_alpha, n_input, n_units, n_embed,
            n_layers=2, lm=None, device='gpu')
        S = 3  # number of states
        if n_layers > 1:
            self.match_embedding = StackedRNN(
                n_alpha, n_input, n_units, n_embed * S, n_layers, lm=lm)
            self.gap_embedding = StackedRNN(
                n_alpha, n_input, n_units, n_embed * S, n_layers, lm=lm)
        else:
            self.match_embedding = EmbedLinear(
                n_alpha, n_input, n_embed * S, lm=lm)
            self.gap_embedding = EmbedLinear(
                n_alpha, n_input, n_embed * S, lm=lm)
        # TODO: need to fix the embeddings to accomodate for a
        # much larger state-space
        self.hmm = ViterbiDecoderCUDA(pos_mxy)
        self.S = S

    def contract(self, x, order):
        """ Tensor contraction operation. """
        zx, _, zy, _ = unpack_sequences(self.match_embedding(x), order)
        gx, _, gy, _ = unpack_sequences(self.gap_embedding(x), order)
        # Obtain theta through an inner product across latent dimensions
        B, n, d3 = zx.shape
        B, m, d3 = zy.shape
        d = d3 // self.S
        zx, zy = zx.view(B, n, d, self.S), zy.view(B, m, d, self.S)
        gx, gy = gx.view(B, n, d, self.S), gy.view(B, m, d, self.S)
        theta = torch.einsum('bids,bjds->bijs', zx, zy)
        A = torch.einsum('bidu,bjdv->bijuv', gx, gy)
        return theta, A

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
            theta, A = self.contract(x, order)
            aln = self.hmm.decode(theta, A)
            return aln, theta, A

    def score(self, x, order):
        with torch.no_grad():
            theta, A = self.contract(x, order)
            ascore = self.hmm(theta, A)
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
            match, gap = self.contract(x, order)
            B, _, _ = match.shape
            for b in range(B):
                aln, _ = self.nw.decode(
                    match[b, :xlen[b], :ylen[b]].unsqueeze(0),
                    gap[b, :xlen[b], :ylen[b]].unsqueeze(0)
                )
                decoded = self.hmm.traceback(aln.squeeze())
                yield decoded, aln
