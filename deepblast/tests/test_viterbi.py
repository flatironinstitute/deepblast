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
from deepblast.utils import make_data, make_alignment_data
import unittest


class TestViterbiUtils(unittest.TestCase):

    def setUp(self):
        # smoke tests
        torch.manual_seed(0)
        S, N, M = 3, 4, 5
        self.theta = torch.randn(N, M, requires_grad=True, dtype=torch.float64)
        self.psi = torch.randn(N, requires_grad=True, dtype=torch.float64)
        self.phi = torch.randn(M, requires_grad=True, dtype=torch.float64)
        self.Ztheta = torch.randn(N, M, requires_grad=True, dtype=torch.float64)
        self.Zpsi = torch.randn(N, requires_grad=True, dtype=torch.float64)
        self.Zphi = torch.randn(M, requires_grad=True, dtype=torch.float64)

        self.Et = torch.ones(S)
        de= torch.randn(2, requires_grad=True, dtype=torch.float64)
        d, e = de[0], de[1]
        self.A = torch.log(
            torch.Tensor([[(1 - 2 * d), d, d],
                          [(1 - e), e, 0],
                          [(1 - e), 0, e]]))
        self.ZA = torch.Tensor([[1 / (1 - 2 * d), 1 / d, 1 / d],
                                [1 / (1 - e), 1 / e, 0],
                                [1 / (1 - e), 0, 1 / e]])
        self.S, self.N, self.M = S, N, M
        # TODO: Compare against hardmax and sparsemax
        self.operator = 'softmax'

    def test_forward_pass(self):
        res = _forward_pass(
            self.theta, self.psi, self.phi, self.A, self.operator)
        self.assertEqual(len(res), self.S)
        resVt, resQt, resQ = res
        self.assertFalse(torch.isnan(resVt))
        self.assertEqual(resQ.shape, (self.N + 2, self.M + 2, self.S, self.S))
        self.assertEqual(resQt.shape[0], self.S)

    def test_backward_pass(self):
        _, Qt, Q = _forward_pass(
            self.theta, self.psi, self.phi, self.A, self.operator)
        resE = _backward_pass(self.Et, Qt, Q)
        self.assertEqual(resE.shape, (self.N + 2, self.M + 2, self.S))

    def test_adjoint_forward_pass(self):
        V, Q = _forward_pass(
            self.theta, self.psi, self.phi, self.A, self.operator)
        E = _backward_pass(Q)
        res = _adjoint_forward_pass(Q, E, self.Ztheta, self.Zpsi,
                                    self.Zphi, self.ZA,
                                    self.operator)
        self.assertEqual(len(res), 2)
        resVd, resQd = res
        self.assertEqual(resVd.shape, (self.N + 1, self.M + 1, self.S))
        self.assertEqual(resQd.shape, (self.N + 2, self.M + 2, self.S, self.S))

    def test_adjoint_backward_pass(self):
        V, Q = _forward_pass(
            self.theta, self.psi, self.phi, self.A, self.operator)
        E = _backward_pass(Q)
        Vd, Qd = _adjoint_forward_pass(Q, E, self.Ztheta, self.Zpsi,
                                       self.Zphi, self.ZA,
                                       self.operator)
        resEd = _adjoint_backward_pass(Q, Qd, E)
        self.assertEqual(resEd.shape, (self.N + 2, self.M + 2, self.S))


class TestViterbiDecoder(TestViterbiUtils):

    def test_grad_viterbi_function(self):
        viterbi = ViterbiDecoder(self.operator)
        inputs = (self.theta, self.psi, self.phi, self.A)
        for i in inputs:
            print(i.shape)
        gradcheck(viterbi, inputs)

    def test_hessian_viterbi_function_backward(self):
        viterbi = ViterbiDecoder(self.operator)
        inputs = (self.theta, self.psi, self.phi, self.A)
        gradgradcheck(viterbi, inputs)


if __name__ == "__main__":
    unittest.main()
