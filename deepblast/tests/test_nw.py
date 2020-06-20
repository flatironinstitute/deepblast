import numpy as np
import torch
from torch.autograd import gradcheck
from torch.autograd.gradcheck import gradgradcheck
from deepblast.nw import NeedlemanWunschDecoder, NeedlemanWunschFunction
from deepblast.ops import operators
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


class TestSoftmax(unittest.TestCase):
    def test_softmax(self):
        x = torch.Tensor([0.1, 1.0, 0.0001]).float()
        P = torch.Tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ]).float()
        op = operators['softmax']
        m, arg = op.max(x)
        h = op.hessian_product(P, x)
        exp_m = torch.Tensor([1.5735]).float()
        exp_arg = torch.Tensor([0.2291, 0.5635, 0.2073]).float()
        exp_h = torch.Tensor(
            [[-0.1520, -0.1240, -0.4860],
             [-0.6081, -0.3101, -0.9720],
             [-1.0641, -0.4961, -1.4581]]).float()
        assert torch.allclose(m, exp_m)
        assert torch.allclose(arg, exp_arg, atol=1e-3, rtol=1e-3)
        assert torch.allclose(h, exp_h, atol=1e-3, rtol=1e-3)

class TestNeedlemanWunschTimer(unittest.TestCase):
    def test_autograd(self):
        N = 100
        M = 100
        theta = torch.randn(N, M, requires_grad=True)
        A = torch.Tensor([1.])
        y = NeedlemanWunschFunction.apply(theta, A, 'softmax')
        y.sum().backward()

class TestNeedlemanWunschDecoder(unittest.TestCase):
    def setUp(self):
        # smoke tests
        torch.manual_seed(2)
        S, N, M = 3, 5, 5
        self.theta = torch.rand(
            N, M, requires_grad=True, dtype=torch.float32)
        self.Ztheta = torch.rand(
            N, M, requires_grad=True, dtype=torch.float32)
        self.Et = 1.
        self.A = torch.Tensor([-1])
        self.S, self.N, self.M = S, N, M
        # TODO: Compare against hardmax and sparsemax
        self.operator = 'softmax'

    def test_decoding(self):
        theta = torch.from_numpy(make_data())
        theta.requires_grad_()
        A = torch.Tensor([0.1])
        needle = NeedlemanWunschDecoder(self.operator)
        v = needle(theta, A)
        v.backward()
        decoded = needle.traceback(theta.grad)
        states = [(0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (4, 3)]
        self.assertListEqual(states, decoded)

    def test_grad_needlemanwunsch_function(self):
        needle = NeedlemanWunschDecoder(self.operator)
        theta, A = self.theta.double(), self.A.double()
        theta.requires_grad_()
        gradcheck(needle, (theta, A), eps=1e-2)

    def test_hessian_needlemanwunsch_function(self):
        needle = NeedlemanWunschDecoder(self.operator)
        inputs = (self.theta, self.A)
        gradgradcheck(needle, inputs, eps=1e-1)


if __name__ == "__main__":
    unittest.main()
