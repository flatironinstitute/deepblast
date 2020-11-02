import numpy as np
import torch
from torch.autograd import gradcheck
from torch.autograd.gradcheck import gradgradcheck
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import torch.testing as tt
from deepblast.viterbi import (
    _forward_pass, _backward_pass,
    ForwardFunction, ForwardDecoder,
    ViterbiDecoder
)
from deepblast.constants import x, m, y, pos_mxys, pos_test
import unittest


class TestViterbiUtils(unittest.TestCase):

    def setUp(self):
        # smoke tests
        torch.manual_seed(0)
        N, M = 4, 5
        S = 4
        self.theta = torch.rand(N, M, S,
                                requires_grad=True,
                                dtype=torch.float32).squeeze()
        self.theta[:, :, x] = 0
        self.theta[:, :, y] = 0
        self.Ztheta = torch.randn(N + 2, M + 2, S)
        self.A = torch.randn(S, S)
        self.ZA = torch.randn(S, S)
        self.N = N
        self.M = M
        self.S = S
        self.operator = 'softmax'

    def test_forward_pass(self):
        res = _forward_pass(
            self.theta, self.A, self.operator)
        self.assertEqual(len(res), 2)
        resV, resQ = res
        self.assertEqual(resQ.shape, (self.N + 2, self.M + 2, 3, 3))
        self.assertAlmostEqual(resV.detach().cpu().item(), 9.1165285)

    def test_forward_pass_soft(self):
        N, M = 2, 2
        theta = torch.ones(N, M, self.S)
        theta[:, :, x] = 0
        theta[:, :, y] = 0
        A = torch.ones(self.S, self.S)
        res = _forward_pass(theta, A, 'softmax')
        self.assertEqual(len(res), 2)
        resV, resQ = res
        self.assertAlmostEqual(resV.detach().cpu().item(), np.log(3*np.exp(4)))

    def test_forward_pass_hard(self):
        N, M = 2, 2
        theta = torch.ones(N, M, self.S)
        theta[:, :, x] = 0
        theta[:, :, y] = 0
        A = torch.ones(self.S, self.S)
        res = _forward_pass(theta, A, 'hardmax')
        vt, q = res
        self.assertEqual(vt, 4)

    def test_backward_pass(self):
        Et = 1
        _, Q = _forward_pass(
            self.theta, self.A, self.operator)
        resE = _backward_pass(Et, Q)
        self.assertEqual(resE.shape, (self.N + 2, self.M + 2, 3))

    def test_backward_pass_hard(self):
        N, M = 2, 2
        theta = torch.ones(N, M, self.S)
        theta[:, :, x] = 0
        theta[:, :, y] = 0
        A = torch.ones(self.S, self.S)
        vt, Q = _forward_pass(theta, A, 'hardmax')
        Et = 1
        resE = _backward_pass(Et, Q)[1:-1, 1:-1]
        print('E\n', resE)

    def test_backward_pass_soft(self):
        N, M = 2, 2
        theta = torch.ones(N, M, self.S)
        #theta[:, :, x] = 0
        #theta[:, :, y] = 0
        A = torch.ones(self.S, self.S)
        vt, Q = _forward_pass(theta, A, 'softmax')
        Et = torch.Tensor([1.])
        resE = _backward_pass(Et, Q)[1:-1, 1:-1]
        expE = torch.Tensor(
            [[[0.0000, 1.0000, 0.0000],
              [0.0000, 0.0000, 0.4683]],
             [[0.4683, 0.0000, 0.0000],
              [0.4683, 0.0634, 0.4683]]])
        # print('resE\n', resE)
        # print('expE\n', expE)

        tt.assert_allclose(resE, expE, atol=1e-3, rtol=1e-3)


class TestForwardDecoder(unittest.TestCase):

    def setUp(self):
        # smoke tests
        torch.manual_seed(2)
        B, S, N, M = 1, 2, 5, 5
        self.theta = torch.ones(N, M, S,
                                requires_grad=True,
                                dtype=torch.float32)
        self.A = torch.ones(N, M, S, S)
        self.Ztheta = torch.rand(N, M, S,
                                 requires_grad=True,
                                 dtype=torch.float32).squeeze()
        self.Et = torch.Tensor([1.])
        self.S, self.N, self.M = S, N, M
        # TODO: Compare against hardmax and sparsemax
        self.operator = 'softmax'

    def test_grad_needlemanwunsch_function_small(self):
        fwd = ForwardDecoder(pos_test, self.operator)
        theta, A = self.theta.double(), self.A.double()
        theta.requires_grad_()
        gradcheck(fwd, (theta, A), eps=1e-2)

    def test_grad_needlemanwunsch_function_larger(self):
        torch.manual_seed(2)
        B, S, N, M = 1, 4, 5, 5
        self.theta = torch.ones(N, M, S,
                                requires_grad=True,
                                dtype=torch.float32)
        self.A = torch.ones(N, M, S, S)
        self.Ztheta = torch.rand(N, M, S,
                                 requires_grad=True,
                                 dtype=torch.float32).squeeze()
        self.Et = torch.Tensor([1.])
        self.S, self.N, self.M = S, N, M
        # TODO: Compare against hardmax and sparsemax
        self.operator = 'softmax'
        fwd = ForwardDecoder(pos_mxys, self.operator)
        theta, A = self.theta.double(), self.A.double()
        theta.requires_grad_()
        gradcheck(fwd, (theta, A), eps=1e-2)


if __name__ == "__main__":
    unittest.main()
