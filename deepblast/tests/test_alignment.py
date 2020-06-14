import torch
from deepblast.alignment import NeedlemanWunschAligner
from deepblast.language_model import BiLM, pretrained_language_models
from deepblast.dataset.alphabet import UniprotTokenizer
import unittest


class TestAlignmentModel(unittest.TestCase):

    def setUp(self):
        path = pretrained_language_models['bilstm']
        self.embedding = BiLM()
        self.embedding.load_state_dict(torch.load(path))
        self.embedding.eval()
        self.tokenizer = UniprotTokenizer()
        nin, nembed, nunits, nout = 22, 1024, 1024, 1024
        self.aligner = NeedlemanWunschAligner(nin, nembed, nunits, nout)

    def test_alignment(self):
        x = torch.Tensor(
            self.tokenizer(b'ARNDCQEGHILKMFPSTWYVXOUBZ')
        ).unsqueeze(0).long()
        y = torch.Tensor(
            self.tokenizer(b'ARNDCQEGHILKARNDCQMFPSTWYVXOUBZ')
        ).unsqueeze(0).long()
        aln = self.aligner(x, y)
        self.assertEqual(aln.shape, (21, 25))

    def test_decode(self):
        pass


if __name__ == '__main__':
    unittest.main()
