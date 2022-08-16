import torch
from deepblast.nw import NeedlemanWunschFunction
from deepblast.nw_cuda import NeedlemanWunschFunction as CUDANWFunction
import time
import cProfile
from pstats import Stats

profile = False
N, M = (800, 800)


def run_cpu_benchmark(N, M):
    theta = torch.randn(N, M, requires_grad=True)
    gap = torch.randn(N, M, requires_grad=True)
    start = time.time()
    y = NeedlemanWunschFunction.apply(theta, gap, 'softmax')
    y.sum().backward()
    t = time.time() - start
    return t


def run_gpu_benchmark(B, N, M):
    cuda_device = torch.device('cuda')
    theta = torch.randn(B, N, M, requires_grad=True,
                        dtype=torch.float32, device=cuda_device)
    gap = torch.randn(B, N, M, requires_grad=True,
                      dtype=torch.float32, device=cuda_device)

    start = time.time()
    y = CUDANWFunction.apply(theta, gap, 'softmax')
    y.sum().backward()
    t = time.time() - start
    return t


# Run once to load everything and compile numba functions
run_cpu_benchmark(N, M)
run_gpu_benchmark(2, N, M)
# Batch size benchmark
if profile:
    pr = cProfile.Profile()
    pr.enable()
print('CPU')
for b in [4, 8, 16, 32, 64, 128, 256]:
    ts = 0
    for i in range(b):
        t = run_cpu_benchmark(N, M)
        ts += t
    print(ts)

print('GPU')
for b in [4, 8, 16, 32, 64, 128, 256]:
    t = run_gpu_benchmark(b, N, M)
    print(t)

print('CPU')

# Length benchmark
b = 64
for N in [64, 128, 256, 512, 1024]:
    ts = 0
    for i in range(b):
        t = run_cpu_benchmark(N, N)
        ts += t
    print(ts)

print('GPU')
for N in [64, 128, 256, 512, 1024]:
    t = run_gpu_benchmark(b, N, N)
    print(t)


if profile:
    pr.disable()
    pr.dump_stats('stats')
    Stats(pr).sort_stats('cumtime').print_stats(10)
