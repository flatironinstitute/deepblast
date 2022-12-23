import unittest
from deepblast.dataset.utils import (
    tmstate_f, states2matrix, states2alignment,
    path_distance_matrix, clip_boundaries,
    pack_sequences, unpack_sequences,
    gap_mask, remove_orphans, revstate_f, decode, _trim_gap, is_subset)
from math import sqrt
import numpy as np
import numpy.testing as npt
import torch
import torch.testing as tt
from transformers import T5EncoderModel, T5Tokenizer


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
        s = np.array(list(map(tmstate_f, s)))
        states2matrix(s, sparse=True)

    def test_states2matrix_only_matches(self):
        s = ":11::11:"
        s = np.array(list(map(tmstate_f, s)))
        npt.assert_allclose(s, np.array([1, 0, 0, 1, 1, 0, 0, 1]))
        M = states2matrix(s, sparse=True)
        res_coords = list(zip(list(M.row), list(M.col)))
        exp_coords = [(0, 0), (1, 0), (2, 0),
                      (3, 1), (4, 2),
                      (5, 2), (6, 2), (7, 3)]
        self.assertListEqual(res_coords, exp_coords)

    def test_states2matrix_shifted(self):
        s = ":11::22:"
        s = np.array(list(map(tmstate_f, s)))
        npt.assert_allclose(s, np.array([1, 0, 0, 1, 1, 2, 2, 1]))
        M = states2matrix(s, sparse=True)
        res_coords = list(zip(list(M.row), list(M.col)))
        exp_coords = [(0, 0), (1, 0), (2, 0),
                      (3, 1), (4, 2),
                      (4, 3), (4, 4), (5, 5)]
        self.assertListEqual(res_coords, exp_coords)

    def test_states2matrix_swap_x(self):
        s = "::2211::"
        s = np.array(list(map(tmstate_f, s)))
        npt.assert_allclose(s, np.array([1, 1, 2, 2, 0, 0, 1, 1]))
        M = states2matrix(s, sparse=True)
        res_coords = list(zip(list(M.row), list(M.col)))
        exp_coords = [(0, 0), (1, 1),
                      (1, 2), (1, 3), (2, 3),
                      (3, 3), (4, 4), (5, 5)]
        self.assertListEqual(res_coords, exp_coords)

    def test_states2matrix_swap_y(self):
        s = "::1122::"
        s = np.array(list(map(tmstate_f, s)))
        npt.assert_allclose(s, np.array([1, 1, 0, 0, 2, 2, 1, 1]))
        M = states2matrix(s, sparse=True)
        res_coords = list(zip(list(M.row), list(M.col)))
        exp_coords = [(0, 0), (1, 1), (2, 1), (3, 1),
                      (3, 2), (3, 3), (4, 4), (5, 5)]
        self.assertListEqual(res_coords, exp_coords)

    def test_states2alignment_1(self):
        s = "111:::222"
        s = np.array(list(map(tmstate_f, s)))
        X = "123456"
        Y = "abcdef"
        exp_x = "123456---"
        exp_y = "---abcdef"
        res_x, res_y = states2alignment(s, X, Y)
        self.assertEqual(res_x, exp_x)
        self.assertEqual(res_y, exp_y)

    def test_states2alignment_2(self):
        s = "111:::111"
        s = np.array(list(map(tmstate_f, s)))
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
        s = np.array(
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 0, 1, 1, 2, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 1]
        )
        states2alignment(s, x, y)

    def test_states2alignment_4(self):
        x = ('XGSSGSSGFDENWGADEELLLIDACETLGLGNWADIADYVGNARTKEECRDHYLKTYIEX')
        y = ('XGEIRVGNRYQADITDLLKEGEEDGRDQSRLETQVWEAHNPLTDKQIDQFLVVARSVGTF'
             'ARALDSLHMSAAAASRDITLFHAMDTLHKNIYDISKAISALVPQGGPVLCRDEMEEWSAS'
             'EANLFEEALEKYGKDFTDIQQDFLPWKSLTSIIEYYYMWKTTX')
        s = np.array(
            [1, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
             1, 1, 2, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1]
        )
        states2alignment(s, x, y)

    def test_states2alignment_5(self):
        x = ('XGSSGSSGFDENWGADEELLLIDACETLGLGNWADIADYVGNARTKEECRDHYLKTYIEX')
        y = ('XGEIRVGNRYQADITDLLKEGEEDGRDQSRLETQVWEAHNPLTDKQIDQFLVVARSVGTF'
             'ARALDSLHMSAAAASRDITLFHAMDTLHKNIYDISKAISALVPQGGPVLCRDEMEEWSAS'
             'EANLFEEALEKYGKDFTDIQQDFLPWKSLTSIIEYYYMWKTTX')
        s = np.array(
            [1, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
             1, 1, 2, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1]
        )
        states2alignment(s, x, y)

    def test_states2alignment_6(self):
        x = 'XGSSGSSGFDENWGADEELLLIDACETLGLGNWADIADYVGNARTKEECRDHYLKTYIEX'
        y = ('XGEIRVGNRYQADITDLLKEGEEDGRDQSRLETQVWEAHNPLTDKQIDQFLVVARSVGT'
             'FARALDSLHMSAAAASRDITLFHAMDTLHKNIYDISKAISALVPQGGPVLCRDEMEEWS'
             'ASEANLFEEALEKYGKDFTDIQQDFLPWKSLTSIIEYYYMWKTTX')
        s = np.array(
            [1, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
             1, 1, 2, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1]
        )
        states2alignment(s, x, y)

    def test_states2alignment_7(self):
        x = ('XGSSGSSGFDENWGADEELLLIDACETLGLGNWADIADYVGNARTKEECRDHYLKTYIEX')
        y = ('XGEIRVGNRYQADITDLLKEGEEDGRDQSRLETQVWEAHNPLTDKQIDQFLVVARSVGTF'
             'ARALDSLHMSAAAASRDITLFHAMDTLHKNIYDISKAISALVPQGGPVLCRDEMEEWSAS'
             'EANLFEEALEKYGKDFTDIQQDFLPWKSLTSIIEYYYMWKTTX')
        s = np.array(
            [1, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
             1, 1, 2, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1]
        )
        states2alignment(s, x, y)

    def test_states2alignment_8(self):
        x = 'HECDDCSKQFSRNNHLAKHLRAH'
        y = 'YRCHKVCPYTFVGKSDLDLHQFITAH'
        s = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1])
        states2alignment(s, y, x)

    def test_states2alignment_9(self):
        x = 'HCH'
        y = 'HCAH'
        s = np.array([1, 1, 0, 1])
        states2alignment(s, y, x)

    def test_states2alignment_10(self):
        gen = 'YACSGGCGQNFRTMSEFNEHMIRLVH'
        oth = 'LICPKHTRDCGKVFKRNSSLRVHEH'
        pred = np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 0, 2, 1, 1, 0, 1, 2, 0, 1, 1, 1, 1]
        )
        states2alignment(pred, gen, oth)

    def test_states2alignment_11(self):
        gen = 'LNCKEIKKYCEMSFRNPDDIRKHRGAIH'
        oth = 'YTCSSCNESLRTAWCLNKHLR'
        pred = np.array(
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0])
        states2alignment(pred, gen, oth)

    def test_clip_ends_none(self):
        from deepblast.constants import m
        s_ = [m, m, m, m]
        x_ = 'GSSG'
        y_ = 'GEIR'
        a_ = '::::'
        rx, ry, rs, _ = clip_boundaries(x_, y_, s_, a_)
        self.assertEqual(x_, rx)
        self.assertEqual(y_, ry)
        self.assertEqual(s_, rs)

    def test_clip_ends(self):
        from deepblast.constants import x, m, y
        s = [x, m, m, m, y]
        x = 'GSSG'
        y = 'GEIR'
        a = '1:::2'
        rx, ry, rs, _ = clip_boundaries(x, y, s, a)
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
        a = ''.join(list(map(revstate_f, list(st))))
        rx, ry, rs, _ = clip_boundaries(gen, oth, st, a)
        self.assertTrue(1)  # just make sure it runs

    def test_pack_sequences(self):
        X = [torch.Tensor([6, 4, 5]),
             torch.Tensor([1, 4, 5, 7])]
        Y = [torch.Tensor([21, 10, 12, 2, 4, 5]),
             torch.Tensor([1, 4, 11, 13, 14])]
        res, order = pack_sequences(X, Y)
        npt.assert_allclose(order, np.array([2, 3, 1, 0]))

    def test_unpack_sequences(self):
        X = [torch.Tensor([6, 4, 5]),
             torch.Tensor([1, 4, 5, 7])]
        Y = [torch.Tensor([21, 10, 12, 2, 4, 5]),
             torch.Tensor([1, 4, 11, 13, 14])]
        z, order = pack_sequences(X, Y)
        resX, xlen, resY, ylen = unpack_sequences(z, order)
        tt.assert_allclose(xlen, torch.Tensor([3, 4]).long())
        tt.assert_allclose(ylen, torch.Tensor([6, 5]).long())
        expX = torch.Tensor([[6, 4, 5, 0, 0, 0],
                             [1, 4, 5, 7, 0, 0]])
        expY = torch.Tensor([[21, 10, 12, 2, 4, 5],
                             [1, 4, 11, 13, 14, 0]])
        tt.assert_allclose(expX, resX)
        tt.assert_allclose(expY, resY)

    def test_is_subset(self):
        x = np.array([0, 1, 1, 0, 0])
        y = np.array([1, 1])
        res = is_subset(y, x)
        self.assertTrue(res)

    def test_trim_gap(self):
        x = np.array([0, 1, 1, 0, 0, 0, 0])
        exp = (0, 4)
        res = _trim_gap(x, 2)
        self.assertEqual(exp[0], res[0])
        self.assertEqual(exp[1], res[1])

    def test_trim_gap2(self):
        x = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1])
        exp = (2, 8)
        i, j = _trim_gap(x, 2)
        self.assertEqual(exp[0], i)
        self.assertEqual(exp[1], j)

    def test_trim_gap3(self):
        # not a real use case, since this will not be evaluated
        x = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        exp = (0, 11)
        _trim_gap(x, 6)


class TestPreprocess(unittest.TestCase):

    def test_gap_mask(self):
        s = ":11::22:"
        res = gap_mask(s)
        exp = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1]
            ]
        )
        npt.assert_equal(res, exp)

        s = ":11:.:22:"
        res = gap_mask(s)
        exp = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1]
            ]
        )
        npt.assert_equal(res, exp)

    def test_gap_mask2(self):
        s = (
            '222222222222222222.11112222222222222222222222222'
            '222222222222222222222222222222222222222222222222'
            '22222222...::::::..:2:22::2:::::::..11.111...::.'
            '::::::::::.::::......:::::::::::222:.::::::::.11'
            '.:::::::::.:22.::::::::::::2:::::::::::::::1::..'
            '.::::::::::::::::::::::22:2:2::::::::::1::::::::'
            '::::22222::::::::::1::::::.'
        )
        # N, M = 197, 283
        gap_mask(s)

    @unittest.skip
    def test_replace_orphans_small(self):
        # WARNING: This test is broken
        s = ":11:11:"
        e = ":111211:"
        r = remove_orphans(s, threshold=3)
        self.assertEqual(r, e)

    @unittest.skip
    def test_replace_orphans(self):
        # WARNING: This test is broken
        s = ":1111111111:11111111111111:"
        e = ":11111111111211111111111111:"
        r = remove_orphans(s, threshold=9)
        self.assertEqual(r, e)

        s = ":2222222222:22222222222222:"
        e = ":22222222221222222222222222:"
        r = remove_orphans(s, threshold=9)
        self.assertEqual(r, e)

        s = ":1111111111:22222222222222:"
        r = remove_orphans(s, threshold=9)
        self.assertEqual(r, s)


class TestPostProcess(unittest.TestCase):
    def test_decode(self):
        tokenizer = T5Tokenizer.from_pretrained(
            "Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
        exp = 'ARNDCQEGHILKMFPSTWYVXOUBZ'
        chars = [' '.join(list(exp))]
        ids = tokenizer.batch_encode_plus(chars, add_special_tokens=False, padding=True)
        input_ids = torch.tensor(ids['input_ids'])
        x_str = decode(list(input_ids.squeeze().cpu().detach().numpy()),
                       tokenizer.get_vocab())
        self.assertEqual(x_str, exp)


if __name__ == '__main__':
    unittest.main()
