import numpy as np
from mpi4py import MPI
import math
import sys
import time

def transform_B_slice(matrix, start, end):
    '''
        Transforms matrix B column slice into a flattened, contiguous array
        so it can be sent with MPI 

        Arguments:
        - matrix: the numpy matrix to transform
        - start: start index for slice
        - end: end index for slice

        Returns:
        - flattened, contiguous array of matrix column slice from start to end
    '''
    m_slice = matrix[: , start:end]
    m_slice = [np.ravel(row) for row in m_slice]
    return np.concatenate(m_slice)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = int(sys.argv[3])
Qn = math.ceil(N / size)
is_divisor = N % size == 0


A = B = None
C = np.empty((N,N), dtype=int)

if rank == 0:
    time_start = time.process_time()
    A = np.load(sys.argv[1])['arr_0']
    B = np.load(sys.argv[2])['arr_0']
    print(f'A:\n{A}')
    print(f'B:\n{B}')

    # initialize A and B slice for process 0
    Ar = A[0:Qn, :]
    Br = B[: , 0:Qn]

    # send A and B slices to other processes
    for r in range(1, size):
        start = Qn * r
        # handle case when N is not divisible by p, last process gets remainder 
        # chunk that's slightly smaller than Qn
        end = min(Qn * (r + 1), N)

        B_slice = transform_B_slice(B, start, end)

        comm.Isend(A[start:end, :], dest=r, tag=0)
        comm.Isend(B_slice, dest=r, tag=1)

elif not is_divisor and rank == size - 1:
    m_size = N - (Qn * (size - 1))
    Ar = np.empty((m_size, N), dtype=int)
    Br = np.empty((N, m_size), dtype=int)

    req = comm.Irecv(Ar, source=0, tag=0)
    req.wait()
    req = comm.Irecv(Br, source=0, tag=1)
    req.wait()

else:
    Ar = np.empty((Qn, N), dtype=int)
    Br = np.empty((N, Qn), dtype=int)

    req = comm.Irecv(Ar, source=0, tag=0)
    req.wait()
    req = comm.Irecv(Br, source=0, tag=1)
    req.wait()

# each processor sends their B column chunk to all other processors
# (non-blocking)
for r in range(0, size):
    if r == rank:
        continue

    B_slice = transform_B_slice(Br, 0, Qn)

    comm.Isend(B_slice, dest=r)

# initialize C row chunk with the first B column chunk in process 0
if rank == 0:
    Cr = np.matmul(Ar, Br)
else:
    if rank == size - 1 and not is_divisor:
        B_process0 = np.empty((N, N - (Qn * (size - 1))), dtype=int)
    else:
        B_process0 = np.empty((N, Qn), dtype=int)
    req = comm.Irecv(B_process0, source=0)
    req.wait()
    Cr = np.matmul(Ar, B_process0)

for r in range(1, size):
    if r == size - 1 and not is_divisor:
        B_other_process = np.empty((N, N - (Qn * (size - 1))), dtype=int)
    else:
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

# Send all C chunks back to process 0
for r in range(1, size):
    comm.Send(Cr, dest=0)

if rank == 0:
    time_end = time.process_time()
    print(f'C:\n{C}')
    print(f'{time_end - time_start} seconds')