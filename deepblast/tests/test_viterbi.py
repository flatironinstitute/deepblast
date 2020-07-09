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
        N, M = 4, 5
        self.theta = torch.randn(N, M)
        self.Ztheta = torch.randn(N, M)
        self.A = torch.Tensor([0.2, 0.1, 0.3, 0.4])
        self.N = N
        self.M = M
        self.operator = 'softmax'

    def test_forward_pass(self):
        res = _forward_pass(
            self.theta, self.A, self.operator)
        self.assertEqual(len(res), 2)
        resV, resQ = res
        self.assertEqual(resQ.shape, (self.N + 2, self.M + 2, 3, 3))

    def test_backward_pass(self):
        Et = 1
        _, Q = _forward_pass(
            self.theta, self.A, self.operator)
        resE = _backward_pass(Et, Q)
        self.assertEqual(resE.shape, (self.N + 2, self.M + 2, 3))

    def test_adjoint_forward_pass(self):
        V, Q = _forward_pass(
            self.theta, self.A, self.operator)
        E = _backward_pass(Q)
        res = _adjoint_forward_pass(Q, E, self.Ztheta, self.ZA,
                                    self.operator)
        self.assertEqual(len(res), 2)
        resVd, resQd = res
        self.assertEqual(resVd.shape, (self.N + 1, self.M + 1, 3))
        self.assertEqual(resQd.shape, (self.N + 2, self.M + 2, 3, 3))

    def test_adjoint_backward_pass(self):
        V, Q = _forward_pass(
            self.theta, self.A, self.operator)
        E = _backward_pass(Q)
        Vd, Qd = _adjoint_forward_pass(Q, E, self.Ztheta, self.ZA,
                                       self.operator)
        resEd = _adjoint_backward_pass(Q, Qd, E)
        self.assertEqual(resEd.shape, (self.N + 2, self.M + 2, 3))


class TestViterbiDecoder(unittest.TestCase):

    def setUp(self):
        # smoke tests
        torch.manual_seed(2)
        B, S, N, M = 1, 3, 2, 3
        self.theta = torch.rand(N,
                                M,
                                S,
                                requires_grad=True,
                                dtype=torch.float32).squeeze()
        self.Ztheta = torch.rand(N,
                                 M,
                                 S,
                                 requires_grad=True,
                                 dtype=torch.float32).squeeze()
        self.Et = torch.Tensor([1.])
        self.A = torch.Tensor([0.2, 0.3, 0.4, 0.5])
        self.S, self.N, self.M = S, N, M
        # TODO: Compare against hardmax and sparsemax
        self.operator = 'softmax'

    def test_grad_needlemanwunsch_function(self):
        needle = ViterbiDecoder(self.operator)
        theta, A = self.theta.double(), self.A.double()
        theta.requires_grad_()
        gradcheck(needle, (theta, A), eps=1e-2)

    def test_hessian_needlemanwunsch_function(self):
        needle = ViterbiDecoder(self.operator)
        theta, A = self.theta, self.A
        theta.requires_grad_()
        inputs = (theta, A)
        gradgradcheck(needle, inputs, eps=1e-2)


if __name__ == "__main__":
    unittest.main()
