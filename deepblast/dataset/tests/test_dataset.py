import unittest
from deepblast.utils import get_data_path
from deepblast.dataset import MaliAlignmentDataset, TMAlignDataset
from deepblast.dataset.dataset import (
    tmstate_f, states2matrix, states2alignment)
import pandas as pd


class TestDataUtils(unittest.TestCase):

    def test_states2matrix_zinc(self):
        s = ':1111::::1:'
        x = 'RGCFH '
        y = 'YGSVHASERH'
        s = list(map(tmstate_f, s))
        states2matrix(s, sparse=True)

    def test_states2matrix_only_matches(self):
        s = ":11::11:"
        s = list(map(tmstate_f, s))
        self.assertEqual(s, [1, 0, 0, 1, 1, 0, 0, 1])
        M = states2matrix(s, sparse=True)
        res_coords = list(zip(list(M.row), list(M.col)))
        exp_coords = [(0, 0), (1, 0), (2, 0),
                      (3, 1), (4, 2),
                      (5, 2), (6, 2), (7, 3)]
        self.assertListEqual(res_coords, exp_coords)

    def test_states2matrix_shifted(self):
        s = ":11::22:"
        s = list(map(tmstate_f, s))
        self.assertEqual(s, [1, 0, 0, 1, 1, 2, 2, 1])
        M = states2matrix(s, sparse=True)
        res_coords = list(zip(list(M.row), list(M.col)))
        exp_coords = [(0, 0), (1, 0), (2, 0),
                      (3, 1), (4, 2),
                      (4, 3), (4, 4), (5, 5)]
        self.assertListEqual(res_coords, exp_coords)

    def test_states2matrix_swap_x(self):
        s = "::2211::"
        s = list(map(tmstate_f, s))
        self.assertEqual(s, [1, 1, 2, 2, 0, 0, 1, 1])
        M = states2matrix(s, sparse=True)
        res_coords = list(zip(list(M.row), list(M.col)))
        exp_coords = [(0, 0), (1, 1),
                      (1, 2), (1, 3), (2, 3),
                      (3, 3), (4, 4), (5, 5)]
        self.assertListEqual(res_coords, exp_coords)

    def test_states2matrix_swap_y(self):
        s = "::1122::"
        s = list(map(tmstate_f, s))
        self.assertEqual(s, [1, 1, 0, 0, 2, 2, 1, 1])
        M = states2matrix(s, sparse=True)
        res_coords = list(zip(list(M.row), list(M.col)))
        exp_coords = [(0, 0), (1, 1), (2, 1), (3, 1),
                      (3, 2), (3, 3), (4, 4), (5, 5)]
        self.assertListEqual(res_coords, exp_coords)

    def test_states2alignment_1(self):
        s = "111:::222"
        s = list(map(tmstate_f, s))
        X = "123456"
        Y = "abcdef"
        exp_x = "123456---"
        exp_y = "---abcdef"
        res_x, res_y = states2alignment(s, X, Y)
        self.assertEqual(res_x, exp_x)
        self.assertEqual(res_y, exp_y)

    def test_states2alignment_2(self):
        s = "111:::111"
        s = list(map(tmstate_f, s))
        X = "123456789"
        Y = "abc"
        exp_x = "123456789"
        exp_y = "---abc---"
        res_x, res_y = states2alignment(s, X, Y)
        self.assertEqual(res_x, exp_x)
        self.assertEqual(res_y, exp_y)

    def test_states2alignment(self):
        x = 'XFECNGCGMVFSSQSSAEKHIRKKHX'
        y = 'XHRCDHPGVNCGKSTRKQGELVEHKSTHX'
        s = [1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 0, 1, 1]
        true_alignment = states2alignment(s, y, x)


class TestTMAlignDataset(unittest.TestCase):
    def setUp(self):
        self.data_path = get_data_path('test_tm_align.tab')

    def test_constructor(self):
        x = TMAlignDataset(self.data_path)
        self.assertEqual(len(x), 3)

    def test_getitem(self):
        x = TMAlignDataset(self.data_path, tm_threshold=0,
                           pad_ends=True, clip_ends=False)
        res = x[0]
        self.assertEqual(len(res), 4)
        gene, pos, states, alignment_matrix = res
        # test the lengths
        self.assertEqual(len(gene), 105)
        self.assertEqual(len(pos), 23)
        self.assertEqual(len(states), 105)
        self.assertEqual(alignment_matrix.shape, (105, 23))

    ## No clipping for now
    # def test_clip_getitem(self):
    #     x = TMAlignDataset(self.data_path, tm_threshold=0,
    #                        pad_ends=True, clip_ends=True)
    #     res = x[0]
    #     self.assertEqual(len(res), 4)
    #     gene, pos, states, alignment_matrix = res
    #     # test the lengths
    #     self.assertEqual(len(gene), 23)
    #     self.assertEqual(len(pos), 23)
    #     self.assertEqual(len(states), 23)
    #     self.assertEqual(alignment_matrix.shape, (23, 23))


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
