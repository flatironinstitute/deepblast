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
    # Careful, we know that these tests are broken
    def setUp(self):
        # smoke tests
        torch.manual_seed(0)
        S, N, M = 3, 2, 1
        self.theta = torch.rand(
            N, M, S, requires_grad=True, dtype=torch.float32)
        self.Ztheta = torch.rand(
            N, M, S, requires_grad=True, dtype=torch.float32)

        self.Et = 1.
        eps = 1e-12
        d, e = 0.2, 0.1
        self.A = torch.log(
            torch.Tensor([[(1 - 2 * d), d, d],
                          [(1 - e), e, eps],
                          [(1 - e), eps, e]]))
        self.ZA = torch.Tensor([[1 / (1 - 2 * d), 1 / d, 1 / d],
                                [1 / (1 - e), 1 / e, eps],
                                [1 / (1 - e), eps, 1 / e]])
        self.S, self.N, self.M = S, N, M
        # TODO: Compare against hardmax and sparsemax
        self.operator = 'softmax'

    def test_forward_pass(self):
        res = _forward_pass(
            self.theta, self.A, self.operator)
        self.assertEqual(len(res), self.S)
        resVt, resQt, resQ = res
        self.assertEqual(resQ.shape, (self.N + 2, self.M + 2, self.S, self.S))
        self.assertEqual(resQt.shape[0], self.S)
        self.assertFalse(torch.isnan(resVt))
        self.assertFalse(torch.isnan(resQt).any())
        self.assertFalse(torch.isnan(resQ).any())

    def test_backward_pass(self):
        _, Qt, Q = _forward_pass(
            self.theta, self.A, self.operator)
        resE = _backward_pass(self.Et, Qt, Q)
        print(resE)
        self.assertEqual(resE.shape, (self.N + 2, self.M + 2, self.S))
        self.assertFalse(torch.isnan(resE).any())

    def test_adjoint_forward_pass(self):
        Vt, Qt, Q = _forward_pass(
            self.theta, self.A, self.operator)
        E = _backward_pass(self.Et, Qt, Q)
        res = _adjoint_forward_pass(Qt, Q, self.Ztheta,
                                    self.ZA, self.operator)
        self.assertEqual(len(res), 3)
        resVtd, resQtd, resQd = res
        self.assertEqual(resVtd.shape[0], self.S)
        self.assertEqual(resQtd.shape, (self.S, self.S))
        self.assertEqual(resQd.shape, (self.N + 2, self.M + 2, self.S, self.S))

        self.assertFalse(torch.isnan(resVtd).any())
        self.assertFalse(torch.isnan(resQtd).any())
        self.assertFalse(torch.isnan(resQd).any())

    def test_adjoint_backward_pass(self):
        Vt, Qt, Q = _forward_pass(
            self.theta, self.A, self.operator)
        E = _backward_pass(self.Et, Qt, Q)
        _, _, Qd = _adjoint_forward_pass(Qt, Q, self.Ztheta,
                                         self.ZA, self.operator)
        resEd = _adjoint_backward_pass(Q, Qd, E)
        self.assertEqual(resEd.shape, (self.N + 2, self.M + 2, self.S))
        self.assertFalse(torch.isnan(resEd).any())


class TestViterbiDecoder(unittest.TestCase):
    # Careful, we know that these tests are broken
    def setUp(self):
        # smoke tests
        torch.manual_seed(0)
        S, N, M = 3, 2, 2
        self.theta = torch.rand(
            N, M, S, requires_grad=True, dtype=torch.float32)
        self.Ztheta = torch.rand(
            N, M, S, requires_grad=True, dtype=torch.float32)

        self.Et = 1.
        eps = 1e-12
        d, e = 0.2, 0.1
        self.A = torch.log(
            torch.Tensor([[(1 - 2 * d), d, d],
                          [(1 - e), e, eps],
                          [(1 - e), eps, e]]))
        self.ZA = torch.Tensor([[1 / (1 - 2 * d), 1 / d, 1 / d],
                                [1 / (1 - e), 1 / e, eps],
                                [1 / (1 - e), eps, 1 / e]])
        self.S, self.N, self.M = S, N, M
        # TODO: Compare against hardmax and sparsemax
        self.operator = 'softmax'

    def test_grad_viterbi_function(self):
        viterbi = ViterbiDecoder(self.operator)
        inputs = (self.theta.double(), self.A.double())
        output = viterbi(*inputs)
        gradcheck(viterbi, inputs)

    def test_hessian_viterbi_function(self):
        viterbi = ViterbiDecoder(self.operator)
        inputs = (self.theta, self.A)
        gradgradcheck(viterbi, inputs)


if __name__ == "__main__":
    unittest.main()
