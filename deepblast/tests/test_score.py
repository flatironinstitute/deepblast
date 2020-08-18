from deepblast.score import roc_edges, alignment_text
from deepblast.dataset.utils import states2edges, tmstate_f
import pandas as pd
import numpy as np
import unittest


class TestScore(unittest.TestCase):

    def setUp(self):
        pass

    def test_alignment_text(self):
        gene = 'YACSGGCGQNFRTMSEFNEHMIRLVH'
        other = 'LICPKHTRDCGKVFKRNSSLRVHEKTH'
        pred = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         1, 2, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1])
        truth = np.array([1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1])
        stats = np.array([1, 1, 1, 1, 1, 1, 1])
        alignment_text(gene, other, pred, truth, stats)

    def test_roc_edges(self):
        cols = ['tp', 'fp', 'fn', 'perc_id', 'ppv', 'fnr', 'fdr']

        exp_alignment = (
            'FRCPRPAGCE--KLYSTSSHVNKHLLL',
            'YDCE---ICQSFKDFSPYMKLRKHRAT',
            '::::111:::22:::::::::::::::'
        )
        res_alignment = (
            'FRCPRPAGCEKLYSTSSHVNKHLL',
            'YDCEICQSFKDFSPYMKLRKHRAT',
            '::::::::::::::::::::::::'
        )
        # TODO: there are still parts of the alignment
        # that are being clipped erronously
        exp_edges = states2edges(
            list(map(tmstate_f, exp_alignment[2])))
        res_edges = states2edges(
            list(map(tmstate_f, res_alignment[2])))

        res = pd.Series(roc_edges(exp_edges, res_edges), index=cols)

        self.assertGreater(res.perc_id, 0.1)

    def test_roc_edges_2(self):
        cols = ['tp', 'fp', 'fn', 'perc_id', 'ppv', 'fnr', 'fdr']
        exp_alignment = (
            'SVHTLLDEKHETLDSEWEKLVRDAMTSGVSKKQFREFLDYQKWRKSQ',
            ':1111111111111111111111111111::::::::::::::::::',
            'I----------------------------FTYGELQRMQEKERNKGQ'
        )
        res_alignment = (
            'SVHTLLDEKHETLDSEWEKLVRDAMTSGVSKKQFREFLDYQKWRKSQ',
            '1:1111111:111111111111111111:11::::::::::::::::',
            '-I-------F------------------T--YGELQRMQEKERNKGQ'
        )

        exp_edges = states2edges(
            list(map(tmstate_f, exp_alignment[2])))
        res_edges = states2edges(
            list(map(tmstate_f, res_alignment[2])))
        res = pd.Series(roc_edges(exp_edges, res_edges), index=cols)
        self.assertGreater(res.tp, 20)
        self.assertGreater(res.perc_id, 0.5)

    def test_roc_edges_3(self):
        cols = ['tp', 'fp', 'fn', 'perc_id', 'ppv', 'fnr', 'fdr']
        exp_alignment = (
            'F--GD--D--------QN-PYTESVDILEDLVIEFITEMTHKAMSI',
            'ISHLVIMHEEGEVDGKAIPDLTAPVSAVQAAVSNLVRVGKETVQTT',
            ':22::22:22222222::2:::::::::::::::::::::::::::'
        )
        res_alignment = (
            '-FG---D------D--QN-PYTESVDILEDLVIEFITEMTHKAMSI',
            'ISHLVIMHEEGEVDGKAIPDLTAPVSAVQAAVSNLVRVGKETVQTT',
            '2::222:222222:22::2:::::::::::::::::::::::::::'
        )
        exp_edges = states2edges(
            list(map(tmstate_f, exp_alignment[2])))
        res_edges = states2edges(
            list(map(tmstate_f, res_alignment[2])))
        res = pd.Series(roc_edges(exp_edges, res_edges), index=cols)
        self.assertGreater(res.tp, 20)
        self.assertGreater(res.perc_id, 0.5)


if __name__ == '__main__':
    unittest.main()
