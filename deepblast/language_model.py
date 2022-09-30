import os
import re
import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    PackedSequence, pack_padded_sequence, pad_packed_sequence)
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod
import esm

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_model(path):
    return os.path.join(_ROOT, 'pretrained_models', path)


pretrained_language_models = {
    'bilstm': get_model('lstm2x.pt')
}


class LanguageModel(nn.Module, metaclass=ABCMeta):
    """ Abstract class for defining new models.

    In order to incorporate new language models, both the
    `encode` method and `tokenize` methods need to be overriden.
    """

    @abstractmethod
    def encode(self, x):
        """ Generates sequence representations from tokenized protein

        Parameters
        ----------
        x : torch.Tensor
            List of protein sequences

        Returns
        -------
        torch.Tensor
            List of protein tensor representations
        """
        pass

    @abstractmethod
    def tokenize(self, x):
        """ Convert protein sequence to tokens

        Parameters
        ----------
        x : list of str
           Protein sequences

        Returns
        -------
        torch.Tensor
            Protein token sequences
        """
        pass

    @property
    @abstractmethod
    def hidden_size(self):
        """ Returns the embedding dimension """
        pass

class BiLM(nn.Module):
    """ Two layer LSTM implemented in Bepler et al 2019

    TODO : Do we want to inherit the new language model structure?
    """
    def __init__(self, nin=22, nout=21, embedding_dim=21, hidden_dim=1024,
                 num_layers=2, tied=True, mask_idx=None, dropout=0):
        super(BiLM, self).__init__()

        if mask_idx is None:
            mask_idx = nin - 1
        self.mask_idx = mask_idx
        self.embed = nn.Embedding(nin, embedding_dim, padding_idx=mask_idx)
        self.dropout = nn.Dropout(p=dropout)

        self.tied = tied
        if tied:
            layers = []
            nin = embedding_dim
            for _ in range(num_layers):
                layers.append(nn.LSTM(nin, hidden_dim, 1, batch_first=True))
                nin = hidden_dim
            self.rnn = nn.ModuleList(layers)
        else:
            layers = []
            nin = embedding_dim
            for _ in range(num_layers):
                layers.append(nn.LSTM(nin, hidden_dim, 1, batch_first=True))
                nin = hidden_dim
            self.lrnn = nn.ModuleList(layers)

            layers = []
            nin = embedding_dim
            for _ in range(num_layers):
                layers.append(nn.LSTM(nin, hidden_dim, 1, batch_first=True))
                nin = hidden_dim
            self.rrnn = nn.ModuleList(layers)

        self.linear = nn.Linear(hidden_dim, nout)

    def hidden_size(self):
        h = 0
        if self.tied:
            for layer in self.rnn:
                h += 2 * layer.hidden_size
        else:
            for layer in self.lrnn:
                h += layer.hidden_size
            for layer in self.rrnn:
                h += layer.hidden_size
        return h

    def reverse(self, h):
        packed = type(h) is PackedSequence
        if packed:
            h, batch_sizes = pad_packed_sequence(h, batch_first=True)
            h_rvs = h.clone().zero_()
            for i in range(h.size(0)):
                n = batch_sizes[i]
                idx = [j for j in range(n - 1, -1, -1)]
                idx = torch.LongTensor(idx).to(h.device)
                h_rvs[i, :n] = h[i].index_select(0, idx)
            # repack h_rvs
            h_rvs = pack_padded_sequence(h_rvs, batch_sizes, batch_first=True)
        else:
            idx = [i for i in range(h.size(1) - 1, -1, -1)]
            idx = torch.LongTensor(idx).to(h.device)
            h_rvs = h.index_select(1, idx)
        return h_rvs

    def transform(self, z_fwd, z_rvs, last_only=False):
        # sequences are flanked by the start/stop token as:
        # [stop, x, stop]

        # z_fwd should be [stop,x]
        # z_rvs should be [x,stop] reversed

        # first, do the forward direction
        if self.tied:
            layers = self.rnn
        else:
            layers = self.lrnn

        h_fwd = []
        h = z_fwd
        for rnn in layers:
            h, _ = rnn(h)
            if type(h) is PackedSequence:
                h = PackedSequence(self.dropout(h.data), h.batch_sizes)
            else:
                h = self.dropout(h)
            if not last_only:
                h_fwd.append(h)
        if last_only:
            h_fwd = h

        # now, do the reverse direction
        if self.tied:
            layers = self.rnn
        else:
            layers = self.rrnn

        # we'll need to reverse the direction of these
        # hidden states back to match forward direction

        h_rvs = []
        h = z_rvs
        for rnn in layers:
            h, _ = rnn(h)
            if type(h) is PackedSequence:
                h = PackedSequence(self.dropout(h.data), h.batch_sizes)
            else:
                h = self.dropout(h)
            if not last_only:
                h_rvs.append(self.reverse(h))
        if last_only:
            h_rvs = self.reverse(h)

        return h_fwd, h_rvs

    def embed_and_split(self, x, pad=False):
        packed = type(x) is PackedSequence
        if packed:
            x, batch_sizes = pad_packed_sequence(x, batch_first=True)

        if pad:
            # pad x with the start/stop token
            x = x + 1
            # append start/stop tokens to x
            x_ = x.data.new(x.size(0), x.size(1) + 2).zero_()
            if packed:
                for i in range(len(batch_sizes)):
                    n = batch_sizes[i]
                    x_[i, 1:n + 1] = x[i, :n]
                batch_sizes = [s + 2 for s in batch_sizes]
            else:
                x_[:, 1:-1] = x
            x = x_

        # sequences x are flanked by the start/stop token as:
        # [stop, x, stop]

        # now, encode x as distributed vectors
        z = self.embed(x)

        # to pass to transform, we discard the last element for z_fwd
        # and the first element for z_rvs
        z_fwd = z[:, :-1]
        z_rvs = z[:, 1:]
        if packed:
            lengths = [s - 1 for s in batch_sizes]
            z_fwd = pack_padded_sequence(z_fwd, lengths, batch_first=True)
            z_rvs = pack_padded_sequence(z_rvs, lengths, batch_first=True)
        # reverse z_rvs
        z_rvs = self.reverse(z_rvs)

        return z_fwd, z_rvs

    def encode(self, x):
        z_fwd, z_rvs = self.embed_and_split(x, pad=True)
        h_fwd_layers, h_rvs_layers = self.transform(z_fwd, z_rvs)

        # concatenate hidden layers together
        packed = type(z_fwd) is PackedSequence
        concat = []
        for h_fwd, h_rvs in zip(h_fwd_layers, h_rvs_layers):
            if packed:
                h_fwd, batch_s = pad_packed_sequence(h_fwd, batch_first=True)
                h_rvs, batch_s = pad_packed_sequence(h_rvs, batch_first=True)
            # discard last element of h_fwd and first element of h_rvs
            h_fwd = h_fwd[:, :-1]
            h_rvs = h_rvs[:, 1:]

            # accumulate for concatenation
            concat.append(h_fwd)
            concat.append(h_rvs)

        h = torch.cat(concat, 2)
        if packed:
            batch_s = [s - 1 for s in batch_s]
            h = pack_padded_sequence(h, batch_s, batch_first=True)

        return h

    def forward(self, x):
        # x's are already flanked by the star/stop token as:
        # [stop, x, stop]
        z_fwd, z_rvs = self.embed_and_split(x, pad=False)
        h_fwd, h_rvs = self.transform(z_fwd, z_rvs, last_only=True)

        packed = type(z_fwd) is PackedSequence
        if packed:
            h_flat = h_fwd.data
            logp_fwd = self.linear(h_flat)
            logp_fwd = PackedSequence(logp_fwd, h_fwd.batch_sizes)
            h_flat = h_rvs.data
            logp_rvs = self.linear(h_flat)
            logp_rvs = PackedSequence(logp_rvs, h_rvs.batch_sizes)
            logp_fwd, batch_s = pad_packed_sequence(logp_fwd, batch_first=True)
            logp_rvs, batch_s = pad_packed_sequence(logp_rvs, batch_first=True)
        else:
            b = h_fwd.size(0)
            n = h_fwd.size(1)
            h_flat = h_fwd.contiguous().view(-1, h_fwd.size(2))
            logp_fwd = self.linear(h_flat)
            logp_fwd = logp_fwd.view(b, n, -1)

            h_flat = h_rvs.contiguous().view(-1, h_rvs.size(2))
            logp_rvs = self.linear(h_flat)
            logp_rvs = logp_rvs.view(b, n, -1)

        # prepend forward logp with zero
        # postpend reverse logp with zero

        b = h_fwd.size(0)
        zero = h_fwd.data.new(b, 1, logp_fwd.size(2)).zero_()
        logp_fwd = torch.cat([zero, logp_fwd], 1)
        logp_rvs = torch.cat([logp_rvs, zero], 1)

        logp = F.log_softmax(logp_fwd + logp_rvs, dim=2)

        if packed:
            batch_s = [s + 1 for s in batch_s]
            logp = pack_padded_sequence(logp, batch_s, batch_first=True)

        return logp


class ESM2(LanguageModel):
    def __init__(self, model_type='esm2_t6_8M_UR50D'):
        """
        Parameters
        ----------
        model_type : str
           Specify the model to be loaded.  The following options are available
               - esm2_t33_650M_UR50D
               - esm2_t36_3B_UR50D
               - esm2_t33_650M_UR50D
               - esm2_t30_150M_UR50D
               - esm2_t12_35M_UR50D
               - esm2_t6_8M_UR50D
        """
        super(ESM2, self).__init__()
        self.model_type = model_type
        self.avail_model_types = {
            'esm2_t33_650M_UR50D' : 5120,
            'esm2_t36_3B_UR50D' : 2560,
            'esm2_t33_650M_UR50D' : 1280,
            'esm2_t30_150M_UR50D' : 640,
            'esm2_t12_35M_UR50D' : 480,
            'esm2_t6_8M_UR50D' : 320
        }
        assert model_type in self.avail_model_types.keys()
        self.model, self.alphabet = eval(f'esm.pretrained.{model_type}()')
        # TODO : assumes GPU is available
        self.model = self.model.eval().cuda()
        pattern = re.compile(r't(\d+)')
        self.layers = int(pattern.findall(model_type)[0])

        self.batch_converter = self.alphabet.get_batch_converter()

        # convert one hot to letters (for decoding)
        data = [('test', 'ARNDCQEGHILKMFPSTWYVXOUBZ')]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)

        bt = list(batch_tokens.squeeze()[1:-1].cpu().detach().numpy())
        self.lookup = {d:b for (d, b) in zip(bt, list(data[0][1]))}

    @property
    def hidden_size(self):
        return self.avail_model_types[self.model_type]

    def encode(self, x):
        results = self.model(x, repr_layers=[self.layers],
                             return_contacts=False)
        # drop beginning token and end tokens
        tokens = results["representations"][self.layers] # [:, 1:-1]
        return tokens

    def forward(self, x):
        return self.encode(x)

    def tokenize(self, x):
        if isinstance(x, str):
            data = [('_', x.upper())]
        else:
            ids = ['_'] * len(x)
            data = list(zip(ids, x.upper()))
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        return batch_tokens

    def untokenize(self, x):
        # optional : we only need this to visualize alignment text while training
        return ''.join([self.lookup[int(i)] for i in x[1:-1]])
