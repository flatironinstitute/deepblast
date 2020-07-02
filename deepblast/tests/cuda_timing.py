#!/usr/bin/env python
import torch
from deepblast.nw_cuda import NeedlemanWunschDecoder
import time
import numpy as np

torch.manual_seed(2)
B, S, N, M = 512, 3, 800, 800
theta = torch.rand(B, N, M, requires_grad=True, dtype=torch.float32)
A = torch.ones(B, dtype=torch.float32) * -1.0


def mytime(func, args):
    starttime = time.time()
    res = func(*args)
    return res, time.time() - starttime


needle = NeedlemanWunschDecoder('softmax')
v = needle(theta[0:1, :, :], A[0:1])
v.sum().backward()

ft, bt = [], []
for i in range(5):
    v, forwardtime = mytime(needle, (theta, A))
    res, backtime = mytime(v.sum().backward, ())

    ft.append(forwardtime)
    bt.append(backtime)

print(np.array(ft).mean())
print(np.array(bt).mean())
