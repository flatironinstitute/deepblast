import unittest
from deepblast.utils import get_data_path
from deepblast.dataset import AlignmentDataset
import pandas as pd


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.data_path = get_data_path('example.txt')
        self.pairs = pd.read_table(self.data_path, header=None)

    def test_constructor(self):
        x = AlignmentDataset(self.pairs)
        self.assertEqual(len(x), 3)

    def test_getitem(self):
        x = AlignmentDataset(self.pairs)
        res = x[0]
        self.assertEqual(len(res), 4)
        gene, pos, states, alignment_matrix = res
        # test the lengths
        self.assertEqual(len(gene) - 2, 81)
        self.assertEqual(len(pos) - 2, 81)
        self.assertEqual(len(states), 100)
        self.assertEqual(alignment_matrix.shape, (83, 83))


if __name__ == '__main__':
    unittest.main()
