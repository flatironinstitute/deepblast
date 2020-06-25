import torch
from deepblast.nw import NeedlemanWunschFunction
import time
import cProfile
from pstats import Stats

profile = True
N, M = (800, 800)


def run_benchmark(N, M):
    theta = torch.randn(N, M, requires_grad=True)
    start = time.time()
    y = NeedlemanWunschFunction.apply(theta, torch.ones(1), 'softmax')
    y.sum().backward()
    print(time.time() - start)
    return theta.grad


# Run once to load everything and compile numba functions
run_benchmark(N, M)

if profile:
    pr = cProfile.Profile()
    pr.enable()

for i in range(5):
    run_benchmark(N, M)

if profile:
    pr.disable()
    pr.dump_stats('stats')
    Stats(pr).sort_stats('cumtime').print_stats(10)
