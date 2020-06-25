from deepblast.nw_cuda import _forward_pass_batch
from deepblast.nw import _forward_pass_numba
import numpy as np
import time
from numba import cuda

nbatch = 1024
N, M = (800, 800)
np.random.seed(1)
theta = np.random.uniform(size=(nbatch, N, M)).astype(np.float32)

Q = np.zeros((nbatch, N+2, M+2, 3), dtype=np.float32)
d_Q = cuda.device_array_like(Q)

A = np.ones(1)
d_A = cuda.to_device(A)

threadsperblock = 32
blockspergrid = (nbatch + (threadsperblock - 1)) // threadsperblock

start = time.time()
d_theta = cuda.to_device(theta)
_forward_pass_batch[blockspergrid, threadsperblock](d_theta, d_A, d_Q)
d_Q.copy_to_host(Q)
print("time: ", time.time() - start)

start = time.time()
d_theta = cuda.to_device(theta)
_forward_pass_batch[blockspergrid, threadsperblock](d_theta, d_A, d_Q)
d_Q.copy_to_host(Q)
print("time: ", time.time() - start)


_forward_pass_numba(theta[0], A[0])
numba_Q = []
start = time.time()
for i in range(10):
    numba_Q.append(_forward_pass_numba(theta[i], A[0])[1])

print("time: ", time.time() - start)

norms = []
for i in range(10):
    norms.append(np.linalg.norm(Q[i] - numba_Q[i]))

print(np.mean(np.array(norms)) / np.sqrt(N*M*3))

