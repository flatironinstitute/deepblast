import numpy as np
import torch
from torch.autograd import gradcheck
from torch.autograd.gradcheck import gradgradcheck
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from deepblast.viterbi import (
    _forward_pass, _backward_pass,
    _adjoint_forward_pass, _adjoint_backward_pass,
    ViterbiFunction, ViterbiFunctionBackward,
    ViterbiDecoder
)
import unittest


class TestViterbiUtils(unittest.TestCase):

    def setUp(self):
        # smoke tests
        torch.manual_seed(0)
        N, M, S = 4, 5, 3
        self.theta = torch.randn(N, M, S)
        self.Ztheta = torch.randn(N, M, S)
        self.G = torch.randn(N, M, S, S)
        self.ZG = torch.randn(N, M, S, S)

        self.N = N
        self.M = M
        self.operator = 'softmax'

    def test_forward_backward_ones(self):
        N, M, S = 2, 2, 3
        Et = torch.Tensor([1.])
        self.theta_ones = torch.ones(N, M, S)
        self.gap_ones = torch.ones(N, M, S, S)
        V, Q = _forward_pass(
            self.theta_ones, self.gap_ones, self.operator)
        print('V\n', V)
        print('Q\n', Q)
        resE = _backward_pass(Et, Q)
        print('E\n', resE)

    def test_forward_pass(self):
        res = _forward_pass(
            self.theta, self.G, self.operator)
        self.assertEqual(len(res), 2)
        resV, resQ = res
        self.assertEqual(resQ.shape, (self.N + 2, self.M + 2, 3, 3))

    def test_backward_pass(self):
        Et = torch.Tensor([1.])
        _, Q = _forward_pass(
            self.theta, self.G, self.operator)
        resE = _backward_pass(Et, Q)
        print(resE)
        self.assertEqual(resE.shape, (self.N + 2, self.M + 2, 3))

    def test_adjoint_forward_pass(self):
        V, Q = _forward_pass(
            self.theta, self.G, self.operator)
        E = _backward_pass(Q)
        res = _adjoint_forward_pass(Q, E, self.Ztheta, self.ZG,
                                    self.operator)
        self.assertEqual(len(res), 2)
        resVd, resQd = res
        self.assertEqual(resVd.shape, (self.N + 1, self.M + 1, 3))
        self.assertEqual(resQd.shape, (self.N + 2, self.M + 2, 3, 3))

    def test_adjoint_backward_pass(self):
        V, Q = _forward_pass(
            self.theta, self.G, self.operator)
        E = _backward_pass(Q)
        Vd, Qd = _adjoint_forward_pass(Q, E, self.Ztheta, self.ZG,
                                       self.operator)
        resEd = _adjoint_backward_pass(Q, Qd, E)
        self.assertEqual(resEd.shape, (self.N + 2, self.M + 2, 3))


class TestViterbiDecoder(unittest.TestCase):

    def setUp(self):
        # smoke tests
        torch.manual_seed(2)
        B, S, N, M = 1, 3, 4, 4
        self.theta = torch.rand(N, M, S,
                                requires_grad=True,
                                dtype=torch.float32).squeeze()
        self.G = torch.rand(N, M, S, S,
                            requires_grad=True,
                            dtype=torch.float32).squeeze()
        self.Ztheta = torch.rand(N, M, S,
                                 requires_grad=True,
                                 dtype=torch.float32).squeeze()
        self.Et = torch.Tensor([1.])
        self.S, self.N, self.M = S, N, M
        # TODO: Compare against hardmax and sparsemax
        self.operator = 'softmax'

    def test_grad_needlemanwunsch_function(self):
        needle = ViterbiDecoder(self.operator)
        theta, G = self.theta.double(), self.G.double()
        theta.requires_grad_()
        gradcheck(needle, (theta, G), eps=1e-2)

    def test_hessian_needlemanwunsch_function(self):
        needle = ViterbiDecoder(self.operator)
        theta, A = self.theta, self.A
        theta.requires_grad_()
        inputs = (theta, A)
        gradgradcheck(needle, inputs, eps=1e-2)


if __name__ == "__main__":
    unittest.main()
