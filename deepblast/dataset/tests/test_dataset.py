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
                           pad_ends=True)
        res = x[0]
        self.assertEqual(len(res), 4)
        gene, pos, states, alignment_matrix = res
        # test the lengths
        self.assertEqual(len(gene), 105)
        self.assertEqual(len(pos), 23)
        self.assertEqual(len(states), 105)
        self.assertEqual(alignment_matrix.shape, (105, 23))


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
        self.assertEqual(len(gene) - 2, 81)
        self.assertEqual(len(pos) - 2, 81)
        self.assertEqual(len(states), 100)
        self.assertEqual(alignment_matrix.shape, (81, 82))


if __name__ == '__main__':
    unittest.main()
