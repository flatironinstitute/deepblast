import torch
import torch.nn as nn
from deepblast.nw_cuda import NeedlemanWunschDecoder as NWDecoderCUDA
from deepblast.nw import NeedlemanWunschDecoder as NWDecoderNumba
from deepblast.sw_cuda import SmithWatermanDecoder as SWDecoderCUDA
from deepblast.sw import SmithWatermanDecoder as SWDecoderNumba
from deepblast.embedding import StackedRNN, StackedCNN
from deepblast.dataset.utils import unpack_sequences

import torch.nn.functional as F


class NeuralAligner(nn.Module):

    def __init__(self, n_alpha, n_input, n_units, n_embed,
                 n_layers=2, dropout=0, lm=None, layer_type='cnn',
                 alignment_mode='needleman-wunsch',
                 device='gpu'):
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
        layer_type : str
           Intermediate layer type

        Notes
        -----
        This only works on GPU at the moment.
        """
        super(NeuralAligner, self).__init__()
        self.lm = lm

        if n_layers > 1:
            if layer_type == 'rnn':
                self.match_embedding = StackedRNN(
                    n_input, n_units, n_embed, n_layers,
                    dropout=dropout)
                self.gap_embedding = StackedRNN(
                    n_input, n_units, n_embed, n_layers,
                    dropout=dropout)
            elif layer_type == 'cnn':
                self.match_embedding = StackedCNN(
                    n_input, n_units, n_embed, n_layers)
                self.gap_embedding = StackedCNN(
                    n_input, n_units, n_embed, n_layers)
            else:
                raise ValueError(f'Layer {layer_type} not supported.')
        else:
            self.match_embedding = nn.Linear(n_embed, n_embed)
            self.gap_embedding = nn.Linear(n_embed, n_embed)

        if alignment_mode == 'needleman-wunsch':
            if device == 'gpu':
                self.ddp = NWDecoderCUDA(operator='softmax')
            else:
                self.ddp = NWDecoderNumba(operator='softmax')
        elif alignment_mode == 'smith-waterman':
            if device == 'gpu':
                self.ddp = SWDecoderCUDA(operator='softmax')
            else:
                self.ddp = SWDecoderNumba(operator='softmax')
        else:
            raise NotImplementedError(
                f'Alignment_mode {alignment_mode} not implemented.')

    def blosum_factor(self, x):
        """ Computes factors for blosum parameters using a single sequence

        Parameters
        ----------
        x : torch.Tensor
           Representation of a single protein sequence
           with dimensions B x (N + 2) x D
        """
        with torch.no_grad():
            embedding = self.lm(input_ids=x,
                                attention_mask=None)
            hx = embedding[0]

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
        mask : tuple of torch.Tensor
            Attention masks for pairs of proteins.

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
            aln = self.ddp.decode(theta, A)
            return aln, theta, A

    def score(self, x, order, mask):
        with torch.no_grad():
            hx, _, hy, _ = unpack_sequences(x, order)
            zx, gx = self.blosum_factor(hx)
            zy, gy = self.blosum_factor(hy)

            # Obtain theta through an inner product across latent dimensions
            theta = F.softplus(torch.einsum('bid,bjd->bij', zx, zy))
            A = F.logsigmoid(torch.einsum('bid,bjd->bij', gx, gy))
            ascore = self.ddp(theta, A)
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
            hx, xlen, hy, ylen = unpack_sequences(x, order)
            zx, gx = self.blosum_factor(hx)
            zy, gy = self.blosum_factor(hy)
            match = F.softplus(torch.einsum('bid,bjd->bij', zx, zy))
            gap = F.logsigmoid(torch.einsum('bid,bjd->bij', gx, gy))
            B, _, _ = match.shape
            for b in range(B):
                aln = self.ddp.decode(
                    match[b, :xlen[b], :ylen[b]].unsqueeze(0),
                    gap[b, :xlen[b], :ylen[b]].unsqueeze(0)
                )
                decoded = self.ddp.traceback(aln.squeeze())
                yield decoded, aln
