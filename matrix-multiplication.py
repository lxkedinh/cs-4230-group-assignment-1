import numpy as np
from mpi4py import MPI
import math
import sys
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = int(sys.argv[3])
Qn = math.ceil(N / size)

# pad matrices in case where N is not cleanly divisible by p
if N % size != 0:
    padding = size - (N % size)
else:
    padding = 0

if rank == 0:
    time_start = time.process_time()

    # Read matrices and prepare them for scattering slices to other processors
    A = np.load(sys.argv[1])['arr_0']
    A = np.pad(A, [(0, padding), (0, padding)], mode="constant")
    B = np.load(sys.argv[2])['arr_0']
    B = np.pad(B, [(0, padding), (0, padding)], mode="constant")

    A_splits = np.array_split(A, size, axis=0)
    for arr in A_splits:
        print(arr.shape)
    A_splits = np.ravel(A_splits)
    B_splits = np.array_split(B, size, axis=1)
    B_splits = np.ravel(B_splits)
else:
    A_splits = B_splits = A = B = None

Ar = np.empty((Qn, N + padding), dtype=int)
Br = np.empty((N + padding, Qn), dtype=int)

comm.Scatter(A_splits, Ar, root=0)
comm.Scatter(B_splits, Br, root=0)


# each processor sends their B column chunk to all other processors
# (non-blocking)
for r in range(0, size):
    if r == rank:
        continue

    comm.Isend(Br, dest=r)

# initialize C row chunk with the first B column chunk in process 0
if rank == 0:
    Cr = np.matmul(Ar, Br)
else:
    B_process0 = np.empty((N + padding, Qn), dtype=int)
    req = comm.Irecv(B_process0, source=0)
    req.wait()
    Cr = np.matmul(Ar, B_process0)

for r in range(1, size):
    # Get B column chunk from other processes to multiply
    if r != rank:
        B_other_process = np.empty((N + padding, Qn), dtype=int)
        req = comm.Irecv(B_other_process, source=r)
        req.wait()
        C_chunk = np.matmul(Ar, B_other_process)
        Cr = np.concatenate((Cr, C_chunk), axis=1)
    # use processs's own B column chunk to multiply
    else:
        C_chunk = np.matmul(Ar, Br)
        Cr = np.concatenate((Cr, C_chunk), axis=1)

# Send all C chunks back to process 0
if rank != 0:
    comm.Send(Cr, dest=0)
else:
    C = Cr

    for r in range(1, size):
        Cr_other_process = np.empty((Qn, N + padding), dtype=int)
        comm.Recv(Cr_other_process, source=r)
        C = np.vstack((C, Cr_other_process))

    time_end = time.process_time()
    print(f'C:\n{C[0:N, 0:N]}')
    print(f'{time_end - time_start} seconds')