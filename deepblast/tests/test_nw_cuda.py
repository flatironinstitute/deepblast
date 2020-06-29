from deepblast.nw_cuda import _forward_pass_batch, _backward_pass_batch
from deepblast.nw import _forward_pass_numba, _backward_pass_numba
import numpy as np
import time
from numba import cuda

nbatch = 1024
N, M = (800, 800)
np.random.seed(1)
theta = np.random.uniform(size=(nbatch, N, M)).astype(np.float32)

Q = np.empty((nbatch, N + 2, M + 2, 3), dtype=np.float32)
d_Q = cuda.device_array_like(Q)

A = np.ones(1)
d_A = cuda.to_device(A)

Vt = np.empty((nbatch), dtype=np.float32)
d_Vt = cuda.device_array_like(Vt)

E = np.empty((nbatch, N + 2, M + 2))
d_E = cuda.device_array_like(E)

Et = np.ones(1)
d_Et = cuda.to_device(Et)

threadsperblock = 32
blockspergrid = (nbatch + (threadsperblock - 1)) // threadsperblock

start = time.time()
d_theta = cuda.to_device(theta)
_forward_pass_batch[blockspergrid, threadsperblock](d_theta, d_A, d_Q, d_Vt)
_backward_pass_batch[blockspergrid, threadsperblock](d_Et, d_Q, d_E)
d_Q.copy_to_host(Q)
d_Vt.copy_to_host(Vt)
d_E.copy_to_host(E)
print("time: ", time.time() - start)

start = time.time()
d_theta = cuda.to_device(theta)
_forward_pass_batch[blockspergrid, threadsperblock](d_theta, d_A, d_Q, d_Vt)
d_Q.copy_to_host(Q)
d_Vt.copy_to_host(Vt)
d_E.copy_to_host(E)
print("time: ", time.time() - start)

Vt_numba, Q_numba = _forward_pass_numba(theta[0], A[0])
E_numba = _backward_pass_numba(Et[0], Q[0])
print(np.linalg.norm(Q_numba - Q[0]) / np.sqrt(Q[0].size))
print(np.linalg.norm(E_numba - E[0]) / np.sqrt(E[0].size))

print(Vt[0] - Vt_numba)
