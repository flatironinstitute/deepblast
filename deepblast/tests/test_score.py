import unittest
from deepblast.score import alignment_score, alignment_score_kernel


class TestScore(unittest.TestCase):

    def setUp(self):
        pass

    def test_alignment_score(self):
        s = ":11::21:"
        p = ":11::12:"
        alignment_score(s, p)

    def test_alignment_score_no_gaps(self):
        s = ":11::21:"
        p = ":11::12:"
        score = alignment_score(s, p, no_gaps=True)
        self.assertEqual(score[0], 4)
        self.assertEqual(score[1], 0)
        self.assertEqual(score[2], 0)

    def test_alignment_score_kernel(self):
        s = ":11::21:"
        p = ":11::12:"
        k = [1, 2]
        alignment_score_kernel(s, p, k, no_gaps=True)


if __name__ == '__main__':
    unittest.main()
