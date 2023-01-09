import numpy as np
import torch
from torch.autograd import gradcheck
from torch.autograd.gradcheck import gradgradcheck
from deepblast.nw import NeedlemanWunschDecoder
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


class TestNeedlemanWunschDecoder(unittest.TestCase):
    def setUp(self):
        # smoke tests
        torch.manual_seed(2)
        B, S, N, M = 1, 3, 4, 4
        self.theta = torch.rand(B,
                                N,
                                M,
                                requires_grad=True,
                                dtype=torch.float32)
        self.Ztheta = torch.rand(B,
                                 N,
                                 M,
                                 requires_grad=True,
                                 dtype=torch.float32)
        self.Et = torch.Tensor([1., 1.])
        self.A = torch.ones_like(self.theta) * -1
        self.B, self.S, self.N, self.M = B, S, N, M
        # TODO: Compare against hardmax and sparsemax
        self.operator = 'softmax'

    def test_decoding(self):
        theta = torch.from_numpy(make_data()).unsqueeze(0)
        theta.requires_grad_()
        A = torch.ones_like(theta) * 0.1
        A.requires_grad_()
        needle = NeedlemanWunschDecoder(self.operator)
        v = needle(theta, A)
        v.backward()
        decoded = needle.traceback(theta.grad.squeeze())
        states = [(0, 0, 0), (1, 0, 0), (2, 0, 1), (3, 1, 1), (4, 2, 2),
                  (4, 3, 1)]
        self.assertListEqual(states, decoded)

    def test_grad_needlemanwunsch_function(self):
        needle = NeedlemanWunschDecoder(self.operator)
        theta, A = self.theta.double(), self.A.double()
        theta.requires_grad_()
        gradcheck(needle, (theta, A), eps=1e-2)

    def test_hessian_needlemanwunsch_function(self):
        needle = NeedlemanWunschDecoder(self.operator)
        inputs = (self.theta, self.A)
        gradgradcheck(needle, inputs, eps=1e-2)

    def test_grad_needlemanwunsch_function_Arand(self):
        needle = NeedlemanWunschDecoder(self.operator)
        theta = self.theta.double()
        A = torch.rand_like(theta)
        theta.requires_grad_()
        gradcheck(needle, (theta, A), eps=1e-2)

    def test_hessian_needlemanwunsch_function_Arand(self):
        needle = NeedlemanWunschDecoder(self.operator)
        theta = self.theta.double()
        A = torch.rand_like(theta)
        inputs = (theta, A)
        gradgradcheck(needle, inputs, eps=1e-2)


if __name__ == "__main__":
    unittest.main()
