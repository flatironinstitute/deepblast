import torch
from deepblast.language_model import BiLM, pretrained_language_models
from deepblast.dataset.alphabet import UniprotTokenizer
import unittest


class TestBiLM(unittest.TestCase):

    def test_bilm(self):
        path = pretrained_language_models['bilstm']
        embedding = BiLM()
        embedding.load_state_dict(torch.load(path))
        embedding.eval()

        tokenizer = UniprotTokenizer()
        toks = torch.Tensor(tokenizer(b'ABC')).long().unsqueeze(0)
        res = embedding(toks)
        self.assertEqual(res.shape, (1, 5, 21))


if __name__ == '__main__':
    unittest.main()
