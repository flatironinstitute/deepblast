import torch
from deepblast.nw_cuda import NeedlemanWunschFunction

B, N, M = (256, 800, 800)
theta = torch.rand(B, N, M, requires_grad=True, dtype=torch.float32)
A = torch.ones(1)
test = NeedlemanWunschFunction.apply(theta, A, 'softmax')
print(test)
