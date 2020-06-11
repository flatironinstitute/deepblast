import numpy as np
import pytest
import torch
from torch.autograd import gradcheck
from torch.autograd.gradcheck import gradgradcheck
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from deepblast.nw import (
    _forward_pass, _backward_pass,
    _adjoint_forward_pass, _adjoint_backward_pass,
    NeedlemanWunschFunction, NeedlemanWunschFunctionBackward,
    NeedlemanWunschDecoder
)
from deepblast.utils import make_data, make_alignment_data
import unittest


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

    def test_grad_needlemanwunsch_function(self):
        needle = NeedlemanWunschDecoder(self.operator)
        inputs = ()
        theta, A = self.theta.double(), self.A.double()
        theta.requires_grad_()
        gradcheck(needle, (theta, A), eps=1e-2)

    def test_hessian_needlemanwunsch_function(self):
        needle = NeedlemanWunschDecoder(self.operator)
        inputs = (self.theta, self.A)
        gradgradcheck(needle, inputs)


if __name__ == "__main__":
    unittest.main()


