from numba import cuda
import torch
from torch.autograd import gradcheck
import torch.testing as tt
from deepblast.viterbi_cuda import (
    ForwardFunction, ForwardDecoder,
    ViterbiDecoder, _soft_max_device
)
from deepblast.ops import operators
import deepblast.viterbi_cuda as vc
import deepblast.viterbi as vb
from deepblast.constants import x, y, pos_mxys, pos_mxy
import unittest


@cuda.jit
def softmax_kernel(X, S, A, M):
    # purely for testing purposes
    i = cuda.grid(1)
    M[i] = _soft_max_device(X[i], S, A[i])


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

            self.Ztheta = torch.randn(B, N + 2, M + 2, S, dtype=float_type,
                                      device=cuda_device, requires_grad=True)
            self.A = torch.randn(B, N, M, S, S, dtype=float_type,
                                 device=cuda_device)
            self.ZA = torch.randn(B, N, M, S, S, dtype=float_type,
                                  device=cuda_device)
            self.B = B
            self.N = N
            self.M = M
            self.S = S
            self.pos = torch.tensor(pos_mxys, dtype=torch.int8,
                                    device=cuda_device).repeat(B, 1, 1)
            self.cuda_device = cuda_device
            self.float_type = float_type

    @unittest.skipUnless(torch.cuda.is_available(), 'No GPU was detected')
    def test_logsumexp(self):
        with torch.no_grad():
            S = 8
            A = torch.zeros(1, S, dtype=self.float_type,
                            device=self.cuda_device, requires_grad=False)
            U = torch.randn(1, S, dtype=self.float_type,
                            device=self.cuda_device, requires_grad=False)
            A = A.detach()
            U = U.detach()
            resM = torch.zeros(1, dtype=self.float_type,
                               device=self.cuda_device, requires_grad=False)
            resM = resM.detach()
            bpg = (1 + (32 - 1)) // 32  # blocks per grid
            softmax_kernel[32, bpg](U, S, A, resM)
            op = operators['softmax']
            expM, expA = op.max(U)
            tt.assert_allclose(resM, expM)
            tt.assert_allclose(A, expA)

    @unittest.skipUnless(torch.cuda.is_available(), 'No GPU was detected')
    def test_forward_pass(self):
        Vt = ForwardFunction.apply(self.theta, self.A, self.pos)
        Vt_ref, _ = vb._forward_pass(self.theta[0], self.A[0],
                                     pos_mxys, 'softmax')
        # Likely huge FP rounding issues between cuda and cpu
        # implementations of log/exp
        assert(abs(Vt[0] - Vt_ref) < 1E-1)
        self.assertEqual(int(Vt.item()), int(Vt_ref.item()))

    @unittest.skipUnless(torch.cuda.is_available(), 'No GPU was detected')
    def test_forward_pass_soft(self):
        B, N, M, S = 1, 100, 100, 3
        theta = torch.ones(B, N, M, S, dtype=self.float_type,
                           device=self.cuda_device)
        theta[:, :, :, x] = 0
        theta[:, :, :, y] = 0
        A = torch.ones(B, N, M, S, S, dtype=self.float_type,
                       device=self.cuda_device)
        pos = torch.tensor(pos_mxy, dtype=torch.int8,
                           device=self.cuda_device).repeat(B, 1, 1)
        res = ForwardFunction.apply(theta, A, pos)

        ref = vb.ForwardFunction.apply(theta[0], A[0], pos_mxy, 'softmax')
        print(ref/res[0] - 1.0)
        self.assertEqual(int(res.item()), int(ref.item()))

    @unittest.skipUnless(torch.cuda.is_available(), 'No GPU was detected')
    def test_backward_pass_soft(self):
        B, N, M, S = 1, 100, 100, 3
        device = self.cuda_device
        dtype = self.float_type
        theta = torch.ones(B, N, M, S, device=device, dtype=dtype)
        A = torch.ones(B, N, M, S, S, device=device, dtype=dtype)
        Q = torch.zeros((B, N + 2, M + 2, S, S), dtype=dtype, device=device)
        resE = torch.zeros((B, N + 2, M + 2, S), dtype=dtype, device=device)
        pos = torch.tensor(pos_mxy, dtype=torch.int8, device=device)
        pos = pos.repeat(B, 1, 1)
        Vt = torch.zeros((B), dtype=dtype, device=device)
        bpg = (B + (vc.tpb - 1)) // vc.tpb

        vc._forward_pass_kernel[vc.tpb, bpg](theta.detach(), A.detach(),
                                             pos.detach(), Q, Vt)

        Et = torch.ones(B, device=theta.device, dtype=theta.dtype)

        vc._backward_pass_kernel[vc.tpb, bpg](Et, Q, pos, resE)
        resE = resE[0, 1:-1, 1:-1].cpu()

        expE = vb._backward_pass(Et.cpu(), Q.squeeze().cpu(), pos_mxy)
        expE = expE[1:-1, 1:-1]
        tt.assert_allclose(resE, expE, atol=1e-3, rtol=1e-3)


class TestForwardDecoder(unittest.TestCase):

    def setUp(self):
        # smoke tests
        torch.manual_seed(2)
        # TODO: Compare against hardmax and sparsemax
        self.operator = 'softmax'
        self.cuda_device = torch.device('cuda')
        self.float_type = torch.float32

    def test_grad_needlemanwunsch_function_small(self):
        B, N, M, S = 1, 3, 3, 4
        self.theta = torch.rand(B, N, M, S,
                                requires_grad=True,
                                device=self.cuda_device,
                                dtype=self.float_type)
        self.A = torch.rand(B, N, M, S, S,
                            device=self.cuda_device,
                            dtype=self.float_type)
        self.Ztheta = torch.rand(B, N, M, S,
                                 requires_grad=True,
                                 device=self.cuda_device,
                                 dtype=self.float_type)
        self.Et = torch.Tensor([1.])
        pos = torch.tensor(pos_mxys, dtype=torch.int8,
                           device=self.cuda_device).repeat(B, 1, 1)
        fwd = ForwardDecoder(pos)
        theta, A = self.theta.double(), self.A.double()
        theta.requires_grad_()
        gradcheck(fwd, (theta, A), eps=1e-2, atol=1e-2, rtol=1e-2)


class TestViterbiDecoder(unittest.TestCase):

    def setUp(self):
        # smoke tests
        torch.manual_seed(2)
        # TODO: Compare against hardmax and sparsemax
        self.operator = 'softmax'
        self.cuda_device = torch.device('cuda')
        self.float_type = torch.float32
        B, N, M, S = 1, 3, 3, 4
        self.theta = torch.rand(B, N, M, S,
                                requires_grad=True,
                                device=self.cuda_device,
                                dtype=self.float_type)
        self.A = torch.rand(B, N, M, S, S,
                            device=self.cuda_device,
                            dtype=self.float_type)
        self.Ztheta = torch.rand(B, N, M, S,
                                 requires_grad=True,
                                 device=self.cuda_device,
                                 dtype=self.float_type)
        self.Et = torch.Tensor([1.])
        self.B = B
        self.N = N
        self.M = M
        self.S = S

    def test_forward(self):
        pos = torch.tensor(pos_mxys, dtype=torch.int8,
                           device=self.cuda_device)
        vit = ViterbiDecoder(pos)
        theta, A = self.theta.double(), self.A.double()
        theta.requires_grad_()
        vit(theta, A)

    def test_decode(self):
        pos = torch.tensor(pos_mxys, dtype=torch.int8,
                           device=self.cuda_device)
        vit = ViterbiDecoder(pos)
        theta, A = self.theta.double(), self.A.double()
        theta.requires_grad_()
        A.requires_grad_()
        vit.decode(theta, A)

    def test_traceback(self):
        pos = torch.tensor(pos_mxys, dtype=torch.int8,
                           device=self.cuda_device)
        vit = ViterbiDecoder(pos)
        theta, A = self.theta.double(), self.A.double()
        theta.requires_grad_()
        A.requires_grad_()
        grad = vit.decode(theta, A)
        theta_grad, A_grad = grad
        states = vit.traceback(theta_grad[0])


if __name__ == "__main__":

    unittest.main()