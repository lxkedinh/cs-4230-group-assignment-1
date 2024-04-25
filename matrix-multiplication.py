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
is_divisor = N % size == 0

A = B = None

if rank == 0:
    time_start = time.process_time()
    A = np.load(sys.argv[1])['arr_0']
    B = np.load(sys.argv[2])['arr_0']

    # initialize A and B slice for process 0
    Ar = A[0:Qn, :]
    Br = B[: , 0:Qn]

    # send A and B slices to other processes
    for r in range(1, size):
        start = Qn * r
        # handle case when N is not divisible by p, last process gets remainder 
        # chunk that's slightly smaller than Qn
        end = min(Qn * (r + 1), N)

        comm.send(A[start:end, :], dest=r, tag=0)
        comm.send(B[: , start:end], dest=r, tag=1)
else:
    Ar = comm.recv(source=0, tag=0)
    Br = comm.recv(source=0, tag=1)

# each processor sends their B column chunk to all other processors
# (non-blocking)
for r in range(0, size):
    if r == rank:
        continue

    comm.send(Br, dest=r)

# initialize C row chunk with the first B column chunk in process 0
if rank == 0:
    Cr = np.matmul(Ar, Br)
else:
    B_process0 = comm.recv(source=0)
    Cr = np.matmul(Ar, B_process0)

for r in range(1, size):
    # Get B column chunk from other processes to multiply
    if r != rank:
        B_other_process = comm.recv(source=r)
        C_chunk = np.matmul(Ar, B_other_process)
        Cr = np.concatenate((Cr, C_chunk), axis=1)
    # use processs's own B column chunk to multiply
    else:
        C_chunk = np.matmul(Ar, Br)
        Cr = np.concatenate((Cr, C_chunk), axis=1)

# Send all C chunks back to process 0
if rank != 0:
    comm.send(Cr, dest=0)
else:
    C = Cr

    for r in range(1, size):
        Cr_other_process = comm.recv(source=r)
        C = np.vstack((C, Cr_other_process))

    time_end = time.process_time()
    print(f'C:\n{C}')
    print(f'{time_end - time_start} seconds')