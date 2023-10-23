import numpy as np
import torch
from torch.autograd import gradcheck
from torch.autograd.gradcheck import gradgradcheck
from deepblast.sw_cuda import SmithWatermanDecoder
from deepblast.utils import get_data_path
from deepblast.dataset.utils import states2alignment
from sklearn.metrics.pairwise import pairwise_distances
import unittest


def make_data():
    rng = np.random.RandomState(0)
    m, n, k = 2, 1, 3
    M = rng.randn(k, 3)
    X = rng.randn(m, 3)
    Y = rng.randn(n, 3)
    X = np.concatenate((X, M), axis=0)
    Y = np.concatenate((M, Y), axis=0)
    eps = 0.1
    return 1 / (pairwise_distances(X, Y) + eps)


class TestSmithWatermanDecoder(unittest.TestCase):
    def setUp(self):
        if torch.cuda.is_available():
            # smoke tests
            cuda_device = torch.device('cuda')

            torch.manual_seed(2)
            B, S, N, M = 3, 3, 5, 5
            self.theta = torch.rand(B,
                                    N,
                                    M,
                                    requires_grad=True,
                                    dtype=torch.float32,
                                    device=cuda_device)
            self.Ztheta = torch.rand(B,
                                     N,
                                     M,
                                     requires_grad=True,
                                     dtype=torch.float32,
                                     device=cuda_device)
            self.A = -1. * torch.ones_like(
                self.theta, dtype=torch.float32, device=cuda_device)
            self.B, self.S, self.N, self.M = B, S, N, M
            # TODO: Compare against hardmax and sparsemax
            self.operator = 'softmax'

    @unittest.skipUnless(torch.cuda.is_available(), 'No GPU was detected')
    def test_grad_smithwaterman_function(self):
        needle = SmithWatermanDecoder(self.operator)
        theta, A = self.theta, self.A
        theta.requires_grad_()
        gradcheck(needle, (theta, A), eps=1e-1, atol=1e-1, rtol=1e-1)

    @unittest.skipUnless(torch.cuda.is_available(), 'No GPU was detected')
    def test_decoding(self):
        theta = torch.tensor(make_data().astype(np.float32),
                             device=self.theta.device).unsqueeze(0)
        theta.requires_grad_()
        A = 0.1 * torch.ones_like(
            theta, dtype=torch.float32, device=self.theta.device)
        needle = SmithWatermanDecoder(self.operator)
        v = needle(theta, A)
        v.backward()
        decoded = needle.traceback(theta.grad.squeeze())
        decoded = [(x[0], x[1]) for x in decoded]
        states = [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 2), (4, 3)]
        self.assertListEqual(states, decoded)

    @unittest.skipUnless(torch.cuda.is_available(), 'No GPU was detected')
    def test_decoding2(self):
        X = 'HECDRKTCDESFSTKGNLRVHKLGH'
        Y = 'LKCSGCGKNFKSQYAYKRHEQTH'

        needle = SmithWatermanDecoder(self.operator)
        dm = torch.Tensor(np.loadtxt(get_data_path('dm.txt')))
        decoded = needle.traceback(dm)
        pred_x, pred_y, pred_states = list(zip(*decoded))
        states2alignment(np.array(pred_states), X, Y)


if __name__ == "__main__":
    unittest.main()
