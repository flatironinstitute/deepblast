import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from deepblast.alignment import NeedlemanWunschAligner
from deepblast.language_model import BiLM, pretrained_language_models
from deepblast.dataset.alphabet import UniprotTokenizer
from deepblast.dataset.utils import collate_f
from deepblast.dataset.utils import unpack_sequences, pack_sequences

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

    @unittest.skipUnless(torch.cuda.is_available(), "No GPU detected")
    def test_batch_alignment(self):
        self.embedding = self.embedding.cuda()
        self.aligner = self.aligner.cuda()
        length = len('ARNDCQEGHILKMFPSTWYVXOUBZ')
        x = torch.zeros((2, length))
        y = torch.zeros((2, length))
        x1 = self.tokenizer(b'ARNDCQEGHILKMFPSTWYVXOUBZ')
        x2 = self.tokenizer(b'ARNDCQEGHILKMFPSTWY')
        y1 = self.tokenizer(b'ARNDCQEGHILKMFPSTWYVXOUBZ')
        y2 = self.tokenizer(b'ARNDCQEGHILKMFPSTWYV')
        x = [torch.Tensor(x1).cuda().long(), torch.Tensor(x2).cuda().long()]
        y = [torch.Tensor(y1).cuda().long(), torch.Tensor(y2).cuda().long()]
        seq, order = pack_sequences(x, y)
        aln, theta, A = self.aligner(seq, order)
        self.assertEqual(aln.shape, (2, length, length))
        self.assertEqual(theta.shape, (2, length, length))

    @unittest.skipUnless(torch.cuda.is_available(), "No GPU detected")
    def test_collate_alignment(self):
        N, M = 4, 5
        x1 = torch.Tensor(self.tokenizer(b'NDCQ')).long()
        x2 = torch.Tensor(self.tokenizer(b'NDC')).long()
        y1 = torch.Tensor(self.tokenizer(b'ND')).long()
        y2 = torch.Tensor(self.tokenizer(b'NDCQE')).long()
        s1 = torch.Tensor([1, 1, 1, 0]).long()
        s2 = torch.Tensor([1, 1, 2, 2, 2]).long()
        A1 = torch.ones((len(x1), len(y1))).long()
        A2 = torch.ones((len(x2), len(y2))).long()
        P1 = torch.ones((len(x1), len(y1))).long()
        P2 = torch.ones((len(x2), len(y2))).long()
        batch = [(x1, y1, s1, A1, P1), (x2, y2, s2, A2, P2)]
        gene_codes, other_codes, states, dm, p = collate_f(batch)
        self.embedding = self.embedding.cuda()
        self.aligner = self.aligner.cuda()
        seq, order = pack_sequences(gene_codes, other_codes)
        seq = seq.cuda()
        aln, theta, A = self.aligner(seq, order)
        # TODO: wtf is going on here??
        self.assertEqual(aln.shape, (2, M, M))
        self.assertEqual(theta.shape, (2, M, M))


if __name__ == '__main__':
    unittest.main()
