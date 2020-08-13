import torch
from deepblast.language_model import BiLM, pretrained_language_models
from deepblast.dataset.alphabet import UniprotTokenizer
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import torch.testing as tt
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
        self.assertEqual(res.shape, (1, 3, 21))

    @unittest.skip('something is misbehaving here.')
    def test_bilm_batch(self):
        toks = torch.Tensor([[0, 20, 4, 3], [0, 20, 4, 0]]).long()
        lens = torch.Tensor([4, 3]).long()
        idx = pack_padded_sequence(toks, lens, batch_first=True)
        res = self.embedding(idx.data)
        x, xlen = pad_packed_sequence(res)
        tt.assert_allclose(xlen, lens)
        tt.assert_allclose(x, toks)


if __name__ == '__main__':
    unittest.main()
