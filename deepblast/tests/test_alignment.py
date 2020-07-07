import torch
from torch.nn.utils.rnn import pack_padded_sequence
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
        nalpha, ninput, nunits, nembed = 22, 1024, 1024, 1024
        self.aligner = NeedlemanWunschAligner(nalpha, ninput, nunits, nembed)


    @unittest.skip("Can only run with GPU")
    def test_alignment(self):
        self.embedding = self.embedding.cuda()
        self.aligner = self.aligner.cuda()
        x = torch.Tensor(
            self.tokenizer(b'ARNDCQEGHILKMFPSTWYVXOUBZ')
        ).unsqueeze(0).long().cuda()
        y = torch.Tensor(
            self.tokenizer(b'ARNDCQEGHILKARNDCQMFPSTWYVXOUBZ')
        ).unsqueeze(0).long().cuda()
        N, M = x.shape[1], y.shape[1]
        x_len = torch.Tensor([N])
        y_len = torch.Tensor([M])
        x = pack_padded_sequence(x, x_len, batch_first=True)
        y = pack_padded_sequence(y, y_len, batch_first=True)

        aln = self.aligner(x, y)
        self.assertEqual(aln.shape, (1, N, M))


if __name__ == '__main__':
    unittest.main()
