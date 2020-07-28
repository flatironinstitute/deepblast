import torch
import torch.nn as nn
from deepblast.language_model import BiLM, pretrained_language_models
from deepblast.nw_cuda import NeedlemanWunschDecoder as NWDecoderCUDA
from deepblast.embedding import StackedRNN, EmbedLinear
from torch.nn.utils.rnn import pad_packed_sequence
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
            self.match_embedding = EmbedLinear(n_alpha, n_input, n_embed, lm=lm)
            self.gap_embedding = EmbedLinear(n_alpha, n_input, n_embed, lm=lm)

        # TODO: make cpu compatible version
        # if device == 'cpu':
        #     self.nw = NWDecoderCPU(operator='softmax')
        # else:
        self.nw = NWDecoderCUDA(operator='softmax')

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
            zx, _ = pad_packed_sequence(
                self.match_embedding(x), batch_first=True)  # dim B x N x D
            zy, _ = pad_packed_sequence(
                self.match_embedding(y), batch_first=True)  # dim B x M x D
            gx, _ = pad_packed_sequence(
                self.gap_embedding(x), batch_first=True)  # dim B x N x D
            gy, _ = pad_packed_sequence(
                self.gap_embedding(y), batch_first=True)  # dim B x M x D
            # Obtain theta through an inner product across latent dimensions
            theta = F.softplus(torch.einsum('bid,bjd->bij', zx, zy))
            A = F.logsigmoid(torch.einsum('bid,bjd->bij', gx, gy))
            aln = self.nw.decode(theta, A)
            return aln, theta, A

    def traceback(self, x, y):
        _, x_len = pad_packed_sequence(x)
        _, y_len = pad_packed_sequence(y)
        with torch.enable_grad():
            zx, _ = pad_packed_sequence(
                self.match_embedding(x), batch_first=True)  # dim B x N x D
            zy, _ = pad_packed_sequence(
                self.match_embedding(y), batch_first=True)  # dim B x M x D
            gx, _ = pad_packed_sequence(
                self.gap_embedding(x), batch_first=True)  # dim B x N x D
            gy, _ = pad_packed_sequence(
                self.gap_embedding(y), batch_first=True)  # dim B x M x D
            theta = F.softplus(torch.einsum('bid,bjd->bij', zx, zy))
            A = F.logsigmoid(torch.einsum('bid,bjd->bij', gx, gy))
            B, _, _ = theta.shape
            for b in range(B):
                aln = self.nw.decode(
                    theta[b, :x_len[b], :y_len[b]].unsqueeze(0),
                    A[b, :x_len[b], :y_len[b]].unsqueeze(0)
                )
                decoded = self.nw.traceback(aln.squeeze())
                yield decoded, aln
