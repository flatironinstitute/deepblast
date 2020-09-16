import unittest
from deepblast.utils import get_data_path
from deepblast.dataset import MaliAlignmentDataset, TMAlignDataset
from deepblast.dataset.alphabet import UniprotTokenizer
import pandas as pd


class TestTMAlignDataset(unittest.TestCase):
    def setUp(self):
        self.data_path = get_data_path('test_tm_align.tab')
        self.tokenizer = UniprotTokenizer(pad_ends=False)

    def test_constructor(self):
        x = TMAlignDataset(self.data_path, tm_threshold=0, max_len=10000)
        self.assertEqual(len(x), 10)

    def test_getitem(self):
        x = TMAlignDataset(self.data_path, tm_threshold=0,
                           pad_ends=False, clip_ends=True)
        res = x[0]
        self.assertEqual(len(res), 6)
        gene, pos, states, alignment_matrix, _, _ = res
        # test the lengths
        self.assertEqual(len(gene), 21)
        self.assertEqual(len(pos), 21)
        self.assertEqual(len(states), 21)
        # wtf is going on here??
        self.assertEqual(alignment_matrix.shape, (21, 21))

    def test_gappy_getitem(self):
        TMAlignDataset(self.data_path, tm_threshold=0,
                       pad_ends=False, clip_ends=False)
        # TODO: we need to make sure that the ends can be appropriately handled

class TestMaliDataset(unittest.TestCase):

    def setUp(self):
        self.data_path = get_data_path('example.txt')
        self.pairs = pd.read_table(self.data_path, header=None)

    def test_constructor(self):
        x = MaliAlignmentDataset(self.pairs)
        self.assertEqual(len(x), 3)

    def test_getitem(self):
        x = MaliAlignmentDataset(self.pairs)
        res = x[0]
        self.assertEqual(len(res), 4)
        gene, pos, states, alignment_matrix = res
        # test the lengths
        self.assertEqual(len(gene), 81)
        self.assertEqual(len(pos), 81)
        self.assertEqual(len(states), 100)
        self.assertEqual(alignment_matrix.shape, (81, 82))


if __name__ == '__main__':
    unittest.main()
