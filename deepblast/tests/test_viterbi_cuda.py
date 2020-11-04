import torch
from torch.autograd import gradcheck
import torch.testing as tt
from deepblast.viterbi_cuda import (
    ViterbiDecoder, ForwardFunction
)
import deepblast.viterbi_cuda as vc
import deepblast.viterbi as vb
from deepblast.constants import x, y, pos_mxys, pos_test, pos_mxy
import unittest


class TestViterbiUtils(unittest.TestCase):

    def setUp(self):
        if torch.cuda.is_available():
            # smoke tests
            cuda_device = torch.device('cuda')
            float_type = torch.float32

            torch.manual_seed(0)
            N, M = 4, 5
            B, S = 1, 4
            self.theta = torch.rand(B, N, M, S,
                                    requires_grad=True,
                                    device=cuda_device,
                                    dtype=float_type)
            self.theta[:, :, :, x] = 0
            self.theta[:, :, :, y] = 0

            self.Ztheta = torch.randn(B, N + 2, M + 2, S, dtype=float_type, device=cuda_device,
                                      requires_grad=True)
            self.A = torch.randn(B, N, M, S, S, dtype=float_type, device=cuda_device)
            self.ZA = torch.randn(B, N, M, S, S, dtype=float_type, device=cuda_device)
            self.B = B
            self.N = N
            self.M = M
            self.S = S
            self.pos = torch.tensor(pos_mxys, dtype=torch.int8, device=cuda_device).repeat(B, 1, 1)
            self.cuda_device = cuda_device
            self.float_type = float_type

    @unittest.skipUnless(torch.cuda.is_available(), 'No GPU was detected')
    def test_forward_pass(self):
        Vt = ForwardFunction.apply(self.theta, self.A, self.pos)

        Vt_ref, _ = vb._forward_pass(self.theta[0], self.A[0], pos_mxys, 'softmax')

        # Likely huge FP rounding issues between cuda and cpu implementations of log/exp
        assert(abs(Vt[0] - Vt_ref) < 1E-1)

    @unittest.skipUnless(torch.cuda.is_available(), 'No GPU was detected')
    def test_forward_pass_soft(self):
        B, N, M, S = 1, 2, 2, 3
        theta = torch.ones(B, N, M, S, dtype=self.float_type, device=self.cuda_device)
        theta[:, :, :, x] = 0
        theta[:, :, :, y] = 0
        A = torch.ones(B, N, M, S, S, dtype=self.float_type, device=self.cuda_device)
        pos = torch.tensor(pos_mxy, dtype=torch.int8, device=self.cuda_device).repeat(B, 1, 1)
        res = ForwardFunction.apply(theta, A, pos)

        ref = vb.ForwardFunction.apply(theta[0], A[0], pos_mxy, 'softmax')
        print(ref/res[0] - 1.0)

    @unittest.skipUnless(torch.cuda.is_available(), 'No GPU was detected')
    def test_backward_pass_soft(self):
        B, N, M, S = 1, 2, 2, 3
        theta = torch.ones(B, N, M, S, device=self.cuda_device, dtype=self.float_type)
        A = torch.ones(B, N, M, S, S, device=theta.device, dtype=theta.dtype)
        Q = torch.zeros((B, N + 2, M + 2, S, S), dtype=theta.dtype, device=theta.device)
        resE = torch.zeros((B, N + 2, M + 2, S), dtype=theta.dtype, device=theta.device)
        pos = torch.tensor(pos_mxys, dtype=torch.int8, device=self.cuda_device).repeat(B, 1, 1)
        Vt = torch.zeros((B), dtype=theta.dtype, device=theta.device)
        bpg = (B + (vc.tpb - 1)) // vc.tpb

        vc._forward_pass_kernel[vc.tpb, bpg](theta.detach(), A.detach(), pos.detach(), Q, Vt)

        Et = torch.ones(B, device=theta.device, dtype=theta.dtype)

        vc._backward_pass_kernel[vc.tpb, bpg](Et, Q, pos, resE)
        resE = resE[0, 1:-1, 1:-1].cpu()

        expE = torch.Tensor(
            [[[1.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.4683]],
             [[0.0000, 0.4683, 0.0000],
              [0.0634, 0.4683, 0.4683]]])

        tt.assert_allclose(resE, expE, atol=1e-3, rtol=1e-3)


# class TestForwardDecoder(unittest.TestCase):

#     def setUp(self):
#         # smoke tests
#         torch.manual_seed(2)
#         S, N, M = 1, 2, 5, 5
#         self.theta = torch.ones(N, M, S,
#                                 requires_grad=True,
#                                 dtype=torch.float32)
#         self.A = torch.ones(N, M, S, S)
#         self.Ztheta = torch.rand(N, M, S,
#                                  requires_grad=True,
#                                  dtype=torch.float32).squeeze()
#         self.Et = torch.Tensor([1.])
#         self.S, self.N, self.M = S, N, M
#         # TODO: Compare against hardmax and sparsemax
#         self.operator = 'softmax'

#     def test_grad_needlemanwunsch_function_small(self):
#         fwd = ForwardDecoder(pos_test, self.operator)
#         theta, A = self.theta.double(), self.A.double()
#         theta.requires_grad_()
#         gradcheck(fwd, (theta, A), eps=1e-2)

#     def test_grad_needlemanwunsch_function_larger(self):
#         torch.manual_seed(2)
#         S, N, M = 4, 5, 5
#         self.theta = torch.ones(N, M, S,
#                                 requires_grad=True,
#                                 dtype=torch.float32)
#         self.A = torch.ones(N, M, S, S)
#         self.Ztheta = torch.rand(N, M, S,
#                                  requires_grad=True,
#                                  dtype=torch.float32).squeeze()
#         self.Et = torch.Tensor([1.])
#         self.S, self.N, self.M = S, N, M
#         # TODO: Compare against hardmax and sparsemax
#         self.operator = 'softmax'
#         fwd = ForwardDecoder(pos_mxys, self.operator)
#         theta, A = self.theta.double(), self.A.double()
#         theta.requires_grad_()
#         gradcheck(fwd, (theta, A), eps=1e-2)


if __name__ == "__main__":
    unittest.main()
