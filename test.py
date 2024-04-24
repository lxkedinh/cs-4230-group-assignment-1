import numpy as np
import math
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 10
Qn = math.ceil(N / size)

if rank == 0:
    A = np.random.randint(1, 10, (N, N), dtype=int)
else:
    A = None

A = comm.scatter(A, root=0)

print(f'{rank}:\n{A}')