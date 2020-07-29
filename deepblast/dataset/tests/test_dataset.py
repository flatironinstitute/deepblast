import unittest
from deepblast.utils import get_data_path
from deepblast.dataset import MaliAlignmentDataset, TMAlignDataset
from deepblast.dataset.dataset import (
    tmstate_f, states2matrix, states2alignment,
    path_distance_matrix, clip_boundaries, collate_f)
from deepblast.dataset.alphabet import UniprotTokenizer
import pandas as pd
from math import sqrt
import numpy as np
import numpy.testing as npt
import torch


class TestDataUtils(unittest.TestCase):

    def test_path_distance_matrix(self):
        pi = [(0, 0), (1, 1), (1, 2), (1, 3), (2, 3), (3, 3)]
        res = path_distance_matrix(pi)
        exp = np.array([
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [sqrt(2), 1, 1, 0],
            [sqrt(5), 2, 1, 0]
        ])
        npt.assert_allclose(res, exp)

    def test_states2matrix_zinc(self):
        s = ':1111::::1:'
        # x = 'RGCFH '
        # y = 'YGSVHASERH'
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

    def test_states2alignment_3(self):
        x = ('XSDHGDVSLPPEDRVRALSQLGSAVEVNEDIPPRRYFRSGVEIIRMA'
             'SIYSEEGNIEHAFILYNKYITLFIEKLPKHRDYKSAVIPEKKDTVK'
             'KLKEIAFPKAEELKAELLKRYTKEYTEYNEEKKKEAEELARNMAIQ'
             'QELX')
        y = ('XIDVLRAKAAKERAERRLQSQQDDIDFKRAELALKRAMNRLSVAEMKX')
        s = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 0, 1, 1, 2, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 1]
        states2alignment(s, x, y)

    def test_states2alignment_4(self):
        x = ('XGSSGSSGFDENWGADEELLLIDACETLGLGNWADIADYVGNARTKEECRDHYLKTYIEX')
        y = ('XGEIRVGNRYQADITDLLKEGEEDGRDQSRLETQVWEAHNPLTDKQIDQFLVVARSVGTF'
             'ARALDSLHMSAAAASRDITLFHAMDTLHKNIYDISKAISALVPQGGPVLCRDEMEEWSAS'
             'EANLFEEALEKYGKDFTDIQQDFLPWKSLTSIIEYYYMWKTTX')
        s = [1, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
             1, 1, 2, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1]
        states2alignment(s, x, y)

    def test_states2alignment_5(self):
        x = ('XGSSGSSGFDENWGADEELLLIDACETLGLGNWADIADYVGNARTKEECRDHYLKTYIEX')
        y = ('XGEIRVGNRYQADITDLLKEGEEDGRDQSRLETQVWEAHNPLTDKQIDQFLVVARSVGTF'
             'ARALDSLHMSAAAASRDITLFHAMDTLHKNIYDISKAISALVPQGGPVLCRDEMEEWSAS'
             'EANLFEEALEKYGKDFTDIQQDFLPWKSLTSIIEYYYMWKTTX')
        s = [1, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
             1, 1, 2, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1]
        states2alignment(s, x, y)

    def test_states2alignment_6(self):
        x = 'XGSSGSSGFDENWGADEELLLIDACETLGLGNWADIADYVGNARTKEECRDHYLKTYIEX'
        y = ('XGEIRVGNRYQADITDLLKEGEEDGRDQSRLETQVWEAHNPLTDKQIDQFLVVARSVGT'
             'FARALDSLHMSAAAASRDITLFHAMDTLHKNIYDISKAISALVPQGGPVLCRDEMEEWS'
             'ASEANLFEEALEKYGKDFTDIQQDFLPWKSLTSIIEYYYMWKTTX')
        s = [1, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
             1, 1, 2, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1]
        states2alignment(s, x, y)

    def test_states2alignment_7(self):
        x = ('XGSSGSSGFDENWGADEELLLIDACETLGLGNWADIADYVGNARTKEECRDHYLKTYIEX')
        y = ('XGEIRVGNRYQADITDLLKEGEEDGRDQSRLETQVWEAHNPLTDKQIDQFLVVARSVGTF'
             'ARALDSLHMSAAAASRDITLFHAMDTLHKNIYDISKAISALVPQGGPVLCRDEMEEWSAS'
             'EANLFEEALEKYGKDFTDIQQDFLPWKSLTSIIEYYYMWKTTX')
        s = [1, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
             1, 1, 2, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1]
        states2alignment(s, x, y)

    def test_states2alignment_8(self):
        x = 'HECDDCSKQFSRNNHLAKHLRAH'
        y = 'YRCHKVCPYTFVGKSDLDLHQFITAH'
        print(len(x), len(y))
        s = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]
        states2alignment(s, x, y)

    def test_states2alignment_9(self):
        x = 'HCH'
        y = 'HCAH'
        print(len(x), len(y))
        s = [1, 1, 0, 1]
        states2alignment(s, x, y)

    def test_states2alignment_10(self):
        gen = 'YACSGGCGQNFRTMSEFNEHMIRLVH'
        oth = 'LICPKHTRDCGKVFKRNSSLRVHEKTH'
        pred = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 0, 2, 1, 1, 0, 1, 2, 0, 1, 1, 1, 1, 1]
        states2alignment(pred, gen, oth)

    def test_states2alignment_11(self):
        gen = 'LNCKEIKKYCEMSFRNPDDIRKHRGAIH'
        oth = 'YTCSSCNESLRTAWCLNKHLRQH'
        pred = [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 1]
        states2alignment(pred, gen, oth)

    def test_clip_ends_none(self):
        from deepblast.constants import m
        s_ = [m, m, m, m]
        x_ = 'GSSG'
        y_ = 'GEIR'
        rx, ry, rs = clip_boundaries(x_, y_, s_)
        self.assertEqual(x_, rx)
        self.assertEqual(y_, ry)
        self.assertEqual(s_, rs)

    def test_clip_ends(self):
        from deepblast.constants import x, m, y
        s = [x, m, m, m, y]
        x = 'GSSG'
        y = 'GEIR'
        rx, ry, rs = clip_boundaries(x, y, s)
        ex, ey, es = 'SSG', 'GEI', [m, m, m]
        self.assertEqual(ex, rx)
        self.assertEqual(ey, ry)
        self.assertEqual(es, rs)

    def test_clip_ends_2(self):
        gen = 'YACNHCGATAIRNPNWKNHQREH'
        oth = 'FHCKSQRVMSDCGSNGSKPFVTNYYVRHQCRKH'
        st = np.array([1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1,
                       1, 2, 2, 2, 2, 1, 1, 1, 0, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 2, 1])
        rx, ry, rs = clip_boundaries(gen, oth, st)
        self.assertTrue(1)


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
