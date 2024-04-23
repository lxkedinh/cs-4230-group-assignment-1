import numpy as np
from mpi4py import MPI
import sys
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = int(sys.argv[3])
Qn = N // size

A = B = None
C = np.empty((N,N), dtype=int)

Ar = np.empty((Qn, N), dtype=int)
Br = np.empty((N, Qn), dtype=int)

if rank == 0:
    start = time.process_time()
    A = np.load(sys.argv[1])['arr_0']
    B = np.load(sys.argv[2])['arr_0']
    print(f'A:\n{A}')
    print(f'B:\n{B}')
    # split B by columns
    B = np.hsplit(B, size)

    # flatten B columns into a contiguous array to use with Scatter
    B = [np.ravel(arr) for arr in B]
    B = np.concatenate(B)

# scatter A row chunks and B column chunks to all processes
comm.Scatter(A, Ar, root=0)
comm.Scatter(B, Br, root=0)

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
    B_process0 = np.empty((N, Qn), dtype=int)
    req = comm.Irecv(B_process0, source=0)
    req.wait()
    Cr = np.matmul(Ar, B_process0)

for r in range(1, size):
    B_other_process = np.empty((N, Qn), dtype=int)

    # Get B column chunk from other processes to multiply
    if r != rank:
        req = comm.Irecv(B_other_process, source=r)
        req.wait()
        C_chunk = np.matmul(Ar, B_other_process)
        Cr = np.concatenate((Cr, C_chunk), axis=1)
    # use processs's own B column chunk to multiply
    else:
        C_chunk = np.matmul(Ar, Br)
        Cr = np.concatenate((Cr, C_chunk), axis=1)

comm.Gather(Cr, C, root=0)

if rank == 0:
    end = time.process_time()
    print(f'C:\n{C}')
    print(f'{end - start} seconds')