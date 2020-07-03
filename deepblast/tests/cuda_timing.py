#!/usr/bin/env python
import torch
from deepblast.nw_cuda import NeedlemanWunschDecoder
import time
import numpy as np

nruns = 5
torch.manual_seed(2)
B, S, N, M = 1024, 3, 800, 800
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
for i in range(nruns):
    v, forwardtime = mytime(needle, (theta, A))
    _, backtime = mytime(v.sum().backward, ())
    del v

    ft.append(forwardtime)
    bt.append(backtime)

print(np.array(ft).mean())
print(np.array(bt).mean())

while True:
    pass
