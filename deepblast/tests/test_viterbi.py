import numpy as np
import pytest
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
        self.psi = torch.randn(N)
        self.phi = torch.randn(M)
        self.Ztheta = torch.randn(N, M)
        self.Zpsi = torch.randn(N)
        self.Zphi = torch.randn(M)
        self.A = torch.Tensor([0.2, 0.1])
        self.N = N
        self.M = M
        self.operator = 'softmax'

    def test_forward_pass(self):
        res = _forward_pass(
            self.theta, self.psi, self.phi, self.A, self.operator)
        self.assertEqual(len(res), 2)
        resV, resQ = res
        self.assertEqual(resV.shape, (self.N + 1, self.M + 1, 3))
        self.assertEqual(resQ.shape, (self.N + 2, self.M + 2, 3, 3))

    def test_backward_pass(self):
        _, Q = _forward_pass(
            self.theta, self.psi, self.phi, self.A, self.operator)
        resE = _backward_pass(Q)
        self.assertEqual(resE.shape, (self.N + 2, self.M + 2, 3))

    def test_adjoint_forward_pass(self):
        V, Q = _forward_pass(
            self.theta, self.psi, self.phi, self.A, self.operator)
        E = _backward_pass(Q)
        res = _adjoint_forward_pass(Q, E, self.Ztheta, self.Zpsi, self.Zphi,
                                    self.operator)
        self.assertEqual(len(res), 2)
        resVd, resQd = res
        self.assertEqual(resVd.shape, (self.N + 1, self.M + 1, 3))
        self.assertEqual(resQd.shape, (self.N + 2, self.M + 2, 3, 3))

    def test_adjoint_backward_pass(self):
        V, Q = _forward_pass(
            self.theta, self.psi, self.phi, self.A, self.operator)
        E = _backward_pass(Q)
        Vd, Qd = _adjoint_forward_pass(Q, E, self.Ztheta, self.Zpsi, self.Zphi,
                                       self.operator)
        resEd = _adjoint_backward_pass(Q, Qd, E)
        self.assertEqual(resEd.shape, (self.N + 2, self.M + 2, 3))


class TestViterbiDecoder(unittest.TestCase):

    def setUp(self):
        pass

    def test_forward(self):
        pass

    def test_decode(self):
        pass


if __name__ == "__main__":
    unittest.main()
