import torch
from torch.autograd import gradcheck, gradgradcheck
import torch.testing as tt
from deepblast.viterbi import (
    _forward_pass, _backward_pass,
    ForwardDecoder
)
from deepblast.constants import x, y, pos_mxys, pos_test, pos_mxy
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
        self.A = torch.randn(N, M, S, S)
        self.ZA = torch.randn(N, M, S, S)
        self.N = N
        self.M = M
        self.S = S
        self.operator = 'softmax'

    def test_forward_pass(self):
        res = _forward_pass(
            self.theta, self.A, pos_mxys, self.operator)
        self.assertEqual(len(res), 2)
        resV, resQ = res
        self.assertEqual(resQ.shape, (self.N + 2, self.M + 2, self.S, self.S))
        self.assertAlmostEqual(resV.detach().cpu().item(), 12.27943229675293)

    def test_forward_pass_soft(self):
        N, M, S = 2, 2, 3
        theta = torch.ones(N, M, S)
        theta[:, :, x] = 0
        theta[:, :, y] = 0
        A = torch.ones(N, M, S, S)
        res = _forward_pass(theta, A, pos_mxy, 'softmax')
        self.assertEqual(len(res), 2)
        resV, resQ = res
        self.assertAlmostEqual(resV.detach().cpu().item(), 5.857235908508301)

    def test_forward_pass_hard(self):
        N, M, S = 2, 2, 3
        theta = torch.ones(N, M, S)
        theta[:, :, x] = 0
        theta[:, :, y] = 0
        A = torch.ones(N, M, S, S)
        res = _forward_pass(theta, A, pos_mxy, 'hardmax')
        vt, q = res
        self.assertEqual(vt, 4)

    def test_backward_pass(self):
        Et = 1
        _, Q = _forward_pass(
            self.theta, self.A, pos_mxys, self.operator)
        resE = _backward_pass(Et, Q, pos_mxys)
        self.assertEqual(resE.shape, (self.N + 2, self.M + 2, self.S))

    def test_backward_pass_hard(self):
        N, M = 2, 2
        theta = torch.ones(N, M, self.S)
        theta[:, :, x] = 0
        theta[:, :, y] = 0
        A = torch.ones(N, M, self.S, self.S)
        vt, Q = _forward_pass(theta, A, pos_mxys, 'hardmax')
        Et = 1
        _backward_pass(Et, Q, pos_mxys)[1:-1, 1:-1]

    def test_backward_pass_soft(self):
        N, M, S = 2, 2, 3
        theta = torch.ones(N, M, S)
        A = torch.ones(N, M, S, S)
        vt, Q = _forward_pass(theta, A, pos_mxy, 'softmax')
        Et = torch.Tensor([1.])
        resE = _backward_pass(Et, Q, pos_mxy)[1:-1, 1:-1]
        expE = torch.Tensor(
            [[[1.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.4683]],
             [[0.0000, 0.4683, 0.0000],
              [0.0634, 0.4683, 0.4683]]])

        tt.assert_allclose(resE, expE, atol=1e-3, rtol=1e-3)


class TestForwardDecoder(unittest.TestCase):

    def setUp(self):
        # smoke tests
        torch.manual_seed(2)
        S, N, M = 2, 5, 5
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

    def test_grad_hmm_function_tiny(self):
        torch.manual_seed(2)
        S, N, M = 2, 1, 1
        self.theta = torch.ones(N, M, S,
                                requires_grad=True,
                                dtype=torch.float32)
        self.A = torch.ones(N, M, S, S)
        self.Et = torch.Tensor([1.])
        self.S, self.N, self.M = S, N, M
        fwd = ForwardDecoder(pos_test, self.operator)
        theta, A = self.theta.double(), self.A.double()
        theta.requires_grad_()
        # A.requires_grad_() # TODO: need to test this scenario
        gradcheck(fwd, (theta, A), eps=1e-2)
        gradgradcheck(fwd, (theta, A), eps=1e-2)

    def test_grad_hmm_function_small(self):
        fwd = ForwardDecoder(pos_test, self.operator)
        theta, A = self.theta.double(), self.A.double()
        theta.requires_grad_()
        gradcheck(fwd, (theta, A), eps=1e-2)
        gradgradcheck(fwd, (theta, A), eps=1e-2)

    def test_grad_hmm_function_larger(self):
        torch.manual_seed(2)
        S, N, M = 4, 5, 5
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
        gradgradcheck(fwd, (theta, A), eps=1e-2)


if __name__ == "__main__":
    unittest.main()
