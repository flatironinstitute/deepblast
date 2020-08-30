import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence


def init_weights(m):
    # https://stackoverflow.com/a/49433937/1167475
    if type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class BatchNorm(nn.Module):
    """ Batch normalization for RNN outputs. """
    def __init__(self, num_features):
        super(BatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(num_features=num_features)
    def forward(self, x):
        return self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)


class MultiLinear(nn.Module):
    """ Multiple linear layers concatenated together"""
    def __init__(self, n_input, n_output, n_heads=16):
        super(MultiLinear, self).__init__()
        self.multi_output = nn.ModuleList(
            [
                nn.Sequential(
                    # BatchNorm(n_input),
                    nn.Linear(n_input, n_output)
                )
                for i in range(n_heads)
            ]
        )
        self.multi_output.apply(init_weights)

    def forward(self, x):
        outputs = torch.stack(
            [head(x) for head in self.multi_output], dim=-1)
        return outputs


class MultiheadProduct(nn.Module):
    def __init__(self, n_input, n_output, n_heads=16):
        super(MultiheadProduct, self).__init__()
        self.multilinear = MultiLinear(n_input, n_output, n_heads)
        self.linear = nn.Linear(n_heads, 1)
        nn.init.xavier_uniform(self.linear.weight)

    def forward(self, x, y):
        zx = self.multilinear(x)
        zy = self.multilinear(y)
        dists = torch.einsum('bidh,bjdh->bijh', zx, zy)
        output = self.linear(dists)
        return output.squeeze()


class LMEmbed(nn.Module):
    def __init__(self, nin, nout, lm, padding_idx=-1, transform=nn.ReLU(),
                 sparse=False):
        super(LMEmbed, self).__init__()

        if padding_idx == -1:
            padding_idx = nin - 1

        self.lm = lm
        self.embed = nn.Embedding(
            nin, nout, padding_idx=padding_idx, sparse=sparse)
        self.proj = nn.Linear(lm.hidden_size(), nout)
        self.transform = transform
        self.nout = nout

    def forward(self, x):
        packed = type(x) is PackedSequence
        h_lm = self.lm.encode(x)

        # embed and unpack if packed
        if packed:
            h = self.embed(x.data)
            h_lm = h_lm.data
        else:
            h = self.embed(x)

        # project
        h_lm = self.proj(h_lm)
        h = self.transform(h + h_lm)

        # repack if needed
        if packed:
            h = PackedSequence(h, x.batch_sizes)

        return h


class EmbedLinear(nn.Module):
    def __init__(self, nin, nhidden, nout, padding_idx=-1,
                 sparse=False, lm=None, transform=nn.ReLU()):
        super(EmbedLinear, self).__init__()

        if padding_idx == -1:
            padding_idx = nin - 1

        if lm is not None:
            self.embed = LMEmbed(
                nin, nhidden, lm, padding_idx=padding_idx, sparse=sparse,
                transform=transform)
            self.proj = nn.Linear(self.embed.nout, nout)
            self.lm = True
        else:
            self.embed = nn.Embedding(
                nin, nout, padding_idx=padding_idx, sparse=sparse)
            self.proj = nn.Linear(nout, nout)
            self.lm = False

        init_weights(self.proj)
        self.nout = nout

    def forward(self, x):

        if self.lm:
            h = self.embed(x)
            if type(h) is PackedSequence:
                h = h.data
                z = self.proj(h)
                z = PackedSequence(z, x.batch_sizes)
            else:
                h = h.view(-1, h.size(2))
                z = self.proj(h)
                z = z.view(x.size(0), x.size(1), -1)
        else:
            if type(x) is PackedSequence:
                z = self.embed(x.data)
                z = PackedSequence(z, x.batch_sizes)
            else:
                z = self.embed(x)

        return z


class StackedRNN(nn.Module):
    def __init__(self, nin, nembed, nunits, nout, nlayers=2,
                 padding_idx=-1, dropout=0, rnn_type='lstm',
                 sparse=False, lm=None, transform=nn.ReLU()):
        super(StackedRNN, self).__init__()

        if padding_idx == -1:
            padding_idx = nin - 1

        if lm is not None:
            self.embed = LMEmbed(
                nin, nembed, lm, padding_idx=padding_idx, sparse=sparse,
                transform=transform)
            nembed = self.embed.nout
            self.lm = True
        else:
            self.embed = nn.Embedding(
                nin, nembed, padding_idx=padding_idx, sparse=sparse)
            self.lm = False

        if rnn_type == 'lstm':

            RNN = nn.LSTM
        elif rnn_type == 'gru':
            RNN = nn.GRU

        self.dropout = nn.Dropout(p=dropout)
        if nlayers == 1:
            dropout = 0

        self.rnn = RNN(nembed, nunits, nlayers, batch_first=True,
                       bidirectional=True, dropout=dropout)
        self.proj = nn.Linear(2 * nunits, nout)
        self.nout = nout

    def forward(self, x):

        if self.lm:
            h = self.embed(x)
        else:
            if type(x) is PackedSequence:
                h = self.embed(x.data)
                h = PackedSequence(h, x.batch_sizes)
            else:
                h = self.embed(x)

        h, _ = self.rnn(h)

        if type(h) is PackedSequence:
            h = h.data
            h = self.dropout(h)
            z = self.proj(h)
            z = PackedSequence(z, x.batch_sizes)
        else:
            h = h.view(-1, h.size(2))
            h = self.dropout(h)
            z = self.proj(h)
            z = z.view(x.size(0), x.size(1), -1)

        return z
