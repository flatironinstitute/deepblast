import torch
from deepblast.language_model import BiLM, pretrained_language_models
from deepblast.dataset.alphabet import UniprotTokenizer
from deepblast.embedding import StackedRNN
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import unittest


class TestBiLM(unittest.TestCase):

    def setUp(self):
        path = pretrained_language_models['bilstm']
        self.embedding = BiLM()
        self.embedding.load_state_dict(torch.load(path))
        self.embedding.eval()
        self.tokenizer = UniprotTokenizer()

    def test_bilm(self):
        toks = torch.Tensor(self.tokenizer(b'ABC')).long().unsqueeze(0)
        res = self.embedding(toks)
        self.assertEqual(res.shape, (1, 5, 21))

    def test_bilm_batch(self):
        n_alpha = 22
        n_input = 512
        n_units = 512
        n_embed = 512
        n_layers = 2
        match_embedding = StackedRNN(
            n_alpha, n_input, n_units, n_embed, n_layers, lm=self.embedding)
        x = torch.Tensor(
            [[13, 5, 13], [1, 18, 16], [4, 4, 4], [7, 6, 15], [2, 10, 6],
             [4, 4, 4], [3, 6, 5], [1, 6, 2], [3, 15, 11], [6, 13, 14],
             [16, 11, 18], [11, 15, 3], [1, 1, 5], [2, 7, 15], [14, 10, 3],
             [1, 10, 13], [2, 19, 2], [15, 1, 19], [8, 8, 8], [10, 11, 18],
             [0, 4, 11], [19, 16, 10], [8, 8, 8], [5, 0, 0], [1, 0, 0],
             [10, 0, 0], [8, 0, 0]]).long().t()
        x_len = torch.Tensor([27, 23, 23]).long()
        y = torch.Tensor(
            [[18, 13,  8], [6, 16, 3], [4, 4, 4], [6, 15, 12], [18, 6, 7],
             [4, 4, 4], [7, 5, 7], [2, 2, 16], [15, 11, 15], [13, 14, 8],
             [15, 18, 3], [8, 3, 17], [4, 5, 14], [3, 15, 13], [3, 3, 16],
             [10, 13, 12], [6, 2, 19], [12, 19, 7], [8, 8, 0], [1, 18, 10],
             [3, 11, 9], [11, 10, 10], [8, 8, 8], [0, 0, 12], [0, 0, 5],
             [0, 0, 10], [0, 0, 5], [0, 0, 8]]).long().t()
        y_len = torch.Tensor([23, 23, 28]).long()
        # print(x.shape, y.shape)
        x_codes = pack_padded_sequence(
            x, x_len,
            enforce_sorted=False, batch_first=True)
        y_codes = pack_padded_sequence(
            y, y_len,
            enforce_sorted=False, batch_first=True)
        zx, _ = pad_packed_sequence(
            match_embedding(x_codes), batch_first=True)  # dim B x N x D
        zy, _ = pad_packed_sequence(
            match_embedding(y_codes), batch_first=True)  # dim B x M x D
        match = F.softplus(torch.einsum('bid,bjd->bij', zx, zy))


if __name__ == '__main__':
    unittest.main()
