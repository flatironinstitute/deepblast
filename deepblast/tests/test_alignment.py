import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from deepblast.alignment import NeedlemanWunschAligner
from deepblast.language_model import BiLM, pretrained_language_models
from deepblast.dataset.alphabet import UniprotTokenizer
from deepblast.dataset.dataset import collate_f
import numpy as np
import numpy.testing as npt
import unittest


class TestAlignmentModel(unittest.TestCase):

    def setUp(self):
        path = pretrained_language_models['bilstm']
        self.embedding = BiLM()
        self.embedding.load_state_dict(torch.load(path))
        self.embedding.eval()
        self.tokenizer = UniprotTokenizer(pad_ends=False)
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

        aln, theta, A = self.aligner(x, y)
        self.assertEqual(aln.shape, (1, N, M))

    @unittest.skip("Can only run with GPU")
    def test_batch_alignment(self):
        self.embedding = self.embedding.cuda()
        self.aligner = self.aligner.cuda()
        l = len('ARNDCQEGHILKMFPSTWYVXOUBZ')
        x = torch.zeros((2, l))
        y = torch.zeros((2, l))
        x1 = self.tokenizer(b'ARNDCQEGHILKMFPSTWYVXOUBZ')
        x2 = self.tokenizer(b'ARNDCQEGHILKMFPSTWY')
        y1 = self.tokenizer(b'ARNDCQEGHILKMFPSTWYVXOUBZ')
        y2 = self.tokenizer(b'ARNDCQEGHILKMFPSTWYV')
        x[0, :len(x1)] = torch.Tensor(x1)
        x[1, :len(x2)] = torch.Tensor(x2)
        y[0, :len(y1)] = torch.Tensor(y1)
        y[1, :len(y2)] = torch.Tensor(y2)
        x = x.long().cuda()
        y = y.long().cuda()
        x_len = torch.Tensor([len(x1), len(x2)])
        y_len = torch.Tensor([len(y1), len(y2)])
        x = pack_padded_sequence(x, x_len, batch_first=True)
        y = pack_padded_sequence(y, y_len, batch_first=True)
        aln, theta, A = self.aligner(x, y)
        self.assertEqual(aln.shape, (2, l, l))
        self.assertEqual(theta.shape, (2, l, l))

    def test_collate_alignment(self):
        N = 4
        M = 5
        x1 = torch.Tensor(self.tokenizer(b'NDCQ'))
        x2 = torch.Tensor(self.tokenizer(b'NDC'))
        y1 = torch.Tensor(self.tokenizer(b'ND'))
        y2 = torch.Tensor(self.tokenizer(b'NDCQE'))
        s1 = torch.Tensor([1, 1, 1, 0])
        s2 = torch.Tensor([1, 1, 2, 2, 2])
        A1 = torch.ones((len(x1), len(y1)))
        A2 = torch.ones((len(x2), len(y2)))
        P1 = torch.ones((len(x1), len(y1)))
        P2 = torch.ones((len(x2), len(y2)))
        batch = [(x1, y1, s1, A1, P1), (x2, y2, s2, A2, P2)]
        gene_codes, other_codes, states, dm, p = collate_f(batch)

        self.embedding = self.embedding.cuda()
        self.aligner = self.aligner.cuda()
        aln, theta, A = self.aligner(
            gene_codes.cuda(), other_codes.cuda())
        self.assertEqual(aln.shape, (2, N, M))
        self.assertEqual(theta.shape, (2, N, M))

        _, glen = pad_packed_sequence(gene_codes, batch_first=True)
        _, olen = pad_packed_sequence(other_codes, batch_first=True)
        npt.assert_allclose(glen.detach().numpy(), np.array([4, 3]))
        npt.assert_allclose(olen.detach().numpy(), np.array([2, 5]))


if __name__ == '__main__':
    unittest.main()
