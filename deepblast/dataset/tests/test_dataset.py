import unittest
from deepblast.utils import get_data_path
from deepblast.dataset import MaliAlignmentDataset, TMAlignDataset
from deepblast.dataset.dataset import tmstate_f, states2matrix
import pandas as pd
import numpy as np


class TestDataUtils(unittest.TestCase):
    def test_states2matrix_only_matches(self):
        s = "111:::111"
        s = list(map(tmstate_f, s))
        self.assertEqual(s, [0, 0, 0, 1, 1, 1, 0, 0, 0])
        M = states2matrix(s, 9, 3, sparse=True)
        res_coords = list(zip(list(M.row), list(M.col)))
        exp_coords = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 1),
                      (5, 2), (6, 2), (7, 2), (8, 2)]
        self.assertListEqual(res_coords, exp_coords)

    def test_states2matrix_shifted(self):
        s = "111:::222"
        s = list(map(tmstate_f, s))
        self.assertEqual(s, [0, 0, 0, 1, 1, 1, 2, 2, 2])
        M = states2matrix(s, 6, 6, sparse=True, )
        res_coords = list(zip(list(M.row), list(M.col)))
        exp_coords = [(0, 0), (1, 0), (2, 0), (3, 0),
                      (4, 1), (5, 2), (5, 3), (5, 4), (5, 5)]
        self.assertListEqual(res_coords, exp_coords)


class TestTMAlignDataset(unittest.TestCase):
    def setUp(self):
        self.data_path = get_data_path('test_tm_align.tab')
        self.pairs = pd.read_table(self.data_path, header=None)
        self.pairs.columns = [
            'chain1_name', 'chain2_name', 'tmscore1', 'tmscore2', 'rmsd',
            'chain1', 'chain2', 'alignment'
        ]

    def test_constructor(self):
        x = TMAlignDataset(self.pairs)
        self.assertEqual(len(x), 3)

    def test_getitem(self):
        x = TMAlignDataset(self.pairs, tm_threshold=0, pad_ends=True)
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
        self.assertEqual(alignment_matrix.shape, (83, 83))


if __name__ == '__main__':
    unittest.main()
