#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <mpi.h>
#include "matrix_multiplication.h"

#define ROOT 0

int *initializeMatrix(int n);
int *add(int *A, int *B, int n);
int *subtract(int *A, int *B, int n);
int *strassenMultiply(int *A, int *B, int n);

int *run_root_mpi(int *A, int *B, int n, int max_rank, int tag, MPI_Comm comm);
int *strassen_parallel_mpi(int *A, int *B, int size, int level, int my_rank, int max_rank, int tag, MPI_Comm comm);
int my_topmost_level_mpi(int my_rank);
void run_helper_mpi(int my_rank, int max_rank, int tag, MPI_Comm comm);
void finalize_nodes();

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);

	int comm_size;
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	int max_rank = comm_size - 1;
	int tag = 123;

	if (my_rank == ROOT)
	{
		clock_t start_time = clock();
		int *A = initializeMatrix(N);
		int *B = initializeMatrix(N);
		int *C = initializeMatrix(N);
		initialization(A, B, C);

		C = run_root_mpi(A, B, N, max_rank, tag, MPI_COMM_WORLD);

		clock_t end_time = clock();
		double cpu_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

		verification(C);
		debug_results(C);
		release_resources(A, B, C);
		execution_report((char *)"Matrix Multiplication", (char *)WORKLOAD, cpu_time, passed_verification);
		finalize_nodes();
	}
	else
	{
		run_helper_mpi(my_rank, max_rank, tag, MPI_COMM_WORLD);
	}
	MPI_Finalize();
	return 0;
}

void run_helper_mpi(int my_rank, int max_rank, int tag, MPI_Comm comm)
{
	int executed = 1;
	while (executed)
	{
		int level = my_topmost_level_mpi(my_rank);
		MPI_Status status_recv[6];
		MPI_Status status;
		int size;
		MPI_Probe(MPI_ANY_SOURCE, 11, comm, &status);
		MPI_Get_count(&status, MPI_INT, &size);
		int parent_rank = status.MPI_SOURCE;

		if (size == 1) { break; }
		int k = sqrt(size);

		int *A11 = initializeMatrix(size);
		int *B12_B22 = initializeMatrix(size);
		int *A11_A12 = initializeMatrix(size);
		int *A21_A22 = initializeMatrix(size);
		int *B11 = initializeMatrix(size);
		int *B22 = initializeMatrix(size);

		MPI_Recv(A11, size, MPI_INT, parent_rank, 11, comm, &status_recv[0]);
		MPI_Recv(B12_B22, size, MPI_INT, parent_rank, 1222, comm, &status_recv[1]);
		MPI_Recv(A11_A12, size, MPI_INT, parent_rank, 1112, comm, &status_recv[2]);
		MPI_Recv(B22, size, MPI_INT, parent_rank, 22, comm, &status_recv[3]);
		MPI_Recv(A21_A22, size, MPI_INT, parent_rank, 2122, comm, &status_recv[4]);
		MPI_Recv(B11, size, MPI_INT, parent_rank, 11, comm, &status_recv[5]);

		int *P1 = strassen_parallel_mpi(A11, B12_B22, k, level, my_rank, max_rank, tag, comm);
		int *P2 = strassen_parallel_mpi(A11_A12, B22, k, level + 1, my_rank, max_rank, tag, comm);
		int *P3 = strassen_parallel_mpi(A21_A22, B11, k, level + 2, my_rank, max_rank, tag, comm);

		MPI_Send(P1, size, MPI_INT, parent_rank, tag + 1 + my_rank, comm);
		MPI_Send(P2, size, MPI_INT, parent_rank, tag + 2 + my_rank, comm);
		MPI_Send(P3, size, MPI_INT, parent_rank, tag + 3 + my_rank, comm);
	}
	return;
}

void finalize_nodes()
{
	int comm_size;
	int my_node;
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_node);
	int exit = 1;
	for (int i = 1; i < comm_size; i++)
	{
		MPI_Send(&exit, 1, MPI_INT, i, 11, MPI_COMM_WORLD);
	}
}

int *strassen_parallel_mpi(int *A, int *B, int size, int level, int my_rank, int max_rank, int tag, MPI_Comm comm)
{
	int helper_rank = my_rank + pow(2, level);
	if (helper_rank >= max_rank || size <= 1)
	{
		return strassenMultiply(A, B, size);
	}
	else
	{
		int k = size / 2;

		MPI_Request request[6];
		MPI_Status status_send[6];
		MPI_Status status[3];

		int *A11 = initializeMatrix(k);
		int *A12 = initializeMatrix(k);
		int *A21 = initializeMatrix(k);
		int *A22 = initializeMatrix(k);
		int *B11 = initializeMatrix(k);
		int *B12 = initializeMatrix(k);
		int *B21 = initializeMatrix(k);
		int *B22 = initializeMatrix(k);

		for (int i = 0; i < k; i++)
		{
			for (int j = 0; j < k; j++)
			{
				A11[index(i, j, k)] = A[index(i, j, size)];
				A12[index(i, j, k)] = A[index(i, k + j, size)];
				A21[index(i, j, k)] = A[index(k + i, j, size)];
				A22[index(i, j, k)] = A[index(k + i, k + j, size)];
				B11[index(i, j, k)] = B[index(i, j, size)];
				B12[index(i, j, k)] = B[index(i, k + j, size)];
				B21[index(i, j, k)] = B[index(k + i, j, size)];
				B22[index(i, j, k)] = B[index(k + i, k + j, size)];
			}
		}

		int *B12_B22 = initializeMatrix(k);
		int *A11_A12 = initializeMatrix(k);
		int *A21_A22 = initializeMatrix(k);

		B12_B22 = subtract(B12, B22, k);
		A11_A12 = add(A11, A12, k);
		A21_A22 = add(A21, A22, k);

		// MPI_Send(A11, k*k, MPI_INT, helper_rank, 11, comm);
		// MPI_Send(B12_B22, k*k, MPI_INT, helper_rank, 1222, comm);
		// MPI_Send(A11_A12, k*k, MPI_INT, helper_rank, 1112, comm);
		// MPI_Send(B22, k*k, MPI_INT, helper_rank, 22, comm);
		// MPI_Send(A21_A22, k*k, MPI_INT, helper_rank, 2122, comm);
		// MPI_Send(B11, k*k, MPI_INT, helper_rank, 11, comm);

		MPI_Isend(A11, k * k, MPI_INT, helper_rank, 11, comm, &request[0]);
		MPI_Isend(subtract(B12, B22, k), k * k, MPI_INT, helper_rank, 1222, comm, &request[1]);
		MPI_Isend(add(A11, A12, k), k * k, MPI_INT, helper_rank, 1112, comm, &request[2]);
		MPI_Isend(B22, k * k, MPI_INT, helper_rank, 22, comm, &request[3]);
		MPI_Isend(add(A21, A22, k), k * k, MPI_INT, helper_rank, 2122, comm, &request[4]);
		MPI_Isend(B11, k * k, MPI_INT, helper_rank, 11, comm, &request[5]);

		int *P1 = initializeMatrix(k);
		int *P2 = initializeMatrix(k);
		int *P3 = initializeMatrix(k);

		int *P4 = strassen_parallel_mpi(A22, subtract(B21, B11, k), k, level + 1, my_rank, max_rank, tag, comm);
		int *P5 = strassen_parallel_mpi(add(A11, A22, k), add(B11, B22, k), k, level + 2, my_rank, max_rank, tag, comm);
		int *P6 = strassen_parallel_mpi(subtract(A12, A22, k), add(B21, B22, k), k, level + 3, my_rank, max_rank, tag, comm);
		int *P7 = strassen_parallel_mpi(subtract(A11, A21, k), add(B11, B12, k), k, level + 4, my_rank, max_rank, tag, comm);

		MPI_Waitall(5, request, status_send);

		MPI_Recv(P1, k * k, MPI_INT, helper_rank, tag + 1 + helper_rank, comm, &status[0]);
		MPI_Recv(P2, k * k, MPI_INT, helper_rank, tag + 2 + helper_rank, comm, &status[1]);
		MPI_Recv(P3, k * k, MPI_INT, helper_rank, tag + 3 + helper_rank, comm, &status[2]);

		int *C = initializeMatrix(size);

		int *C11 = subtract(add(add(P5, P4, k), P6, k), P2, k);
		int *C12 = add(P1, P2, k);
		int *C21 = add(P3, P4, k);
		int *C22 = subtract(subtract(add(P5, P1, k), P3, k), P7, k);

		for (int i = 0; i < k; i++)
		{
			for (int j = 0; j < k; j++)
			{
				C[index(i, j, size)] = C11[index(i, j, k)];
				C[index(i, j + k, size)] = C12[index(i, j, k)];
				C[index(k + i, j, size)] = C21[index(i, j, k)];
				C[index(k + i, k + j, size)] = C22[index(i, j, k)];
			}
		}
		return C;
	}
}

int my_topmost_level_mpi(int my_rank)
{
	int level = 0;
	while (pow(2, level) <= my_rank)
		level++;
	return level;
}

int *run_root_mpi(int *A, int *B, int n, int max_rank, int tag, MPI_Comm comm)
{
	int my_rank;
	MPI_Comm_rank(comm, &my_rank);
	if (my_rank != 0)
	{
		printf("Error: run_root_mpi called from process %d; must be called from process 0 only\n",
			   my_rank);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	return strassen_parallel_mpi(A, B, n, 0, my_rank, max_rank, tag, comm);
}

int *initializeMatrix(int n)
{
	return (int *)malloc(n * n * sizeof(int *));
}

int *add(int *M1, int *M2, int n)
{
	int *temp = initializeMatrix(n);
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			temp[index(i, j, n)] = M1[index(i, j, n)] + M2[index(i, j, n)];

	return temp;
}

int *subtract(int *M1, int *M2, int n)
{
	int *temp = initializeMatrix(n);
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			temp[index(i, j, n)] = M1[index(i, j, n)] - M2[index(i, j, n)];

	return temp;
}

int *strassenMultiply(int *A, int *B, int n)
{
	int *C = initializeMatrix(n);
	int k = n / 2;

	if (n == 1)
	{
		int *C = initializeMatrix(1);
		C[0] = A[0] * B[0];
		return C;
	}
	else
	{
		int *A11 = initializeMatrix(k);
		int *A12 = initializeMatrix(k);
		int *A21 = initializeMatrix(k);
		int *A22 = initializeMatrix(k);
		int *B11 = initializeMatrix(k);
		int *B12 = initializeMatrix(k);
		int *B21 = initializeMatrix(k);
		int *B22 = initializeMatrix(k);

		for (int i = 0; i < k; i++)
		{
			for (int j = 0; j < k; j++)
			{
				A11[index(i, j, k)] = A[index(i, j, n)];
				A12[index(i, j, k)] = A[index(i, k + j, n)];
				A21[index(i, j, k)] = A[index(k + i, j, n)];
				A22[index(i, j, k)] = A[index(k + i, k + j, n)];
				B11[index(i, j, k)] = B[index(i, j, n)];
				B12[index(i, j, k)] = B[index(i, k + j, n)];
				B21[index(i, j, k)] = B[index(k + i, j, n)];
				B22[index(i, j, k)] = B[index(k + i, k + j, n)];
			}
		}

		int *P1 = strassenMultiply(A11, subtract(B12, B22, k), k);
		int *P2 = strassenMultiply(add(A11, A12, k), B22, k);
		int *P3 = strassenMultiply(add(A21, A22, k), B11, k);
		int *P4 = strassenMultiply(A22, subtract(B21, B11, k), k);
		int *P5 = strassenMultiply(add(A11, A22, k), add(B11, B22, k), k);
		int *P6 = strassenMultiply(subtract(A12, A22, k), add(B21, B22, k), k);
		int *P7 = strassenMultiply(subtract(A11, A21, k), add(B11, B12, k), k);

		int *C11 = subtract(add(add(P5, P4, k), P6, k), P2, k);
		int *C12 = add(P1, P2, k);
		int *C21 = add(P3, P4, k);
		int *C22 = subtract(subtract(add(P5, P1, k), P3, k), P7, k);

		for (int i = 0; i < k; i++)
		{
			for (int j = 0; j < k; j++)
			{
				C[index(i, j, n)] = C11[index(i, j, k)];
				C[index(i, j + k, n)] = C12[index(i, j, k)];
				C[index(k + i, j, n)] = C21[index(i, j, k)];
				C[index(k + i, k + j, n)] = C22[index(i, j, k)];
			}
		}

		return C;
	}
}
