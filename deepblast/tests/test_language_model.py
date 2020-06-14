import torch
from deepblast.language_model import BiLM, pretrained_language_models
from deepblast.dataset.alphabet import UniprotTokenizer
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


if __name__ == '__main__':
    unittest.main()
