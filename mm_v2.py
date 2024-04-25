def main():
    from mpi4py import MPI
    import numpy as np
    import time
    import sys

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Load matrices
    if rank == 0:
        start = time.process_time()
        matrix_file_a = sys.argv[1]
        matrix_file_b = sys.argv[2]
        n = int(sys.argv[3])

        # Load the matrices from files
        A = np.load(matrix_file_a)['arr_0']
        B = np.load(matrix_file_b)['arr_0']
    else:
        A = None
        B = None
        n = None

    # Broadcast n to all processes
    n = comm.bcast(n, root=0)

    # Split matrices A and B
    if rank == 0:
        A_splits = np.array_split(A, size, axis=0)
        B_splits = np.array_split(B, size, axis=1)
    else:
        A_splits = None
        B_splits = None

    # Scatter rows of A and columns of B
    A_local = comm.scatter(A_splits, root=0)
    B_local = comm.scatter(B_splits, root=0)

    # Local computation of matrix multiplication with ring communication
    C_local = np.zeros((A_local.shape[0], n))
    current_B = B_local.copy()

    for step in range(size):
        offset = (rank - step + size) % size
        B_local = current_B

        for i in range(A_local.shape[0]):
            for j in range(n // size):
                for k in range(A_local.shape[1]):
                    C_local[i, j + offset * (n // size)] += A_local[i, k] * B_local[k, j]

        # Send B_local to the next processor and receive new B_local from the previous one
        if step < size - 1:
            current_B = np.empty_like(B_local)
            comm.Sendrecv(sendbuf=B_local, dest=(rank + 1) % size, recvbuf=current_B, source=(rank - 1 + size) % size)

    # Gather all local C matrices
    C_gathered = comm.gather(C_local, root=0)

    # Assemble the final matrix C
    if rank == 0:
        end = time.process_time()
        C = np.hstack(np.split(np.vstack(C_gathered), size, axis=1))
        with open("result_v2.txt", "a") as file:
            file.write(f'Multiplying matrices with n = {n} and p = {size}\n')
            file.write("Resulting Matrix C:\n")
            file.write(f"{C}\n")
            file.write(f'It takes {end - start} seconds\n\n\n')

if __name__ == "__main__":
    main()
