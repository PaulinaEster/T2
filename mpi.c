#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <mpi.h>
#include "matrix_multiplication.h"

#define ROOT 0

int **initializeMatrix(int n);
int **add(int **A, int **B, int n);
int **subtract(int **A, int **B, int n);
void allocMatrixes(int **A, int **B, int **C, int n);
void printMatrix(int **M, int n);
int **strassenMultiply(int **A, int **B, int n);

int** run_root_mpi(int **A, int **B, int n, int max_rank, int tag, MPI_Comm comm);
int **strassen_parallel_mpi(int **A, int **B, int size, int level, int my_rank, int max_rank, int tag, MPI_Comm comm);
int my_topmost_level_mpi(int my_rank);
void run_helper_mpi(int my_rank, int max_rank, int tag, MPI_Comm comm);

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
		int **A = initializeMatrix(N+1);
		int **B = initializeMatrix(N+1);
		int **C = initializeMatrix(N+1);
		initialization(A, B, C);

		printf("[%d] initialization A %d\n", my_rank, A[1][0]);

		printf("run_root_mpi\n");
		C = run_root_mpi(A, B, N, max_rank, tag, MPI_COMM_WORLD);

		printf("[%d] verification\n", my_rank);
		verification(C); 
	} else {
		printf("[%d] run_helper_mpi\n", my_rank);
		run_helper_mpi(my_rank, max_rank, tag, MPI_COMM_WORLD);
	}

    MPI_Finalize();
	return 0;
}

void run_helper_mpi(int my_rank, int max_rank, int tag, MPI_Comm comm)
{
	int level = my_topmost_level_mpi(my_rank);
	MPI_Status status_recv[6];
	MPI_Status status;
	int size;
	MPI_Probe(MPI_ANY_SOURCE, 11, comm, &status);
	MPI_Get_count(&status, MPI_INT, &size);
	
	int parent_rank = status.MPI_SOURCE;
	
	int **A11 = initializeMatrix(size);
	int **B12_B22 = initializeMatrix(size);
	int **A11_A12 = initializeMatrix(size); 
	int **A21_A22 = initializeMatrix(size);
	int **B11 = initializeMatrix(size);
	int **B22 = initializeMatrix(size);
	A11[0][0] = 1;

	printf("[%d] RECEIV A11 %d STATUS %d A[0][0] %d \n", my_rank, parent_rank, status_recv[0].MPI_ERROR, A11[0][0]);
	MPI_Recv(A11, size, MPI_INT, parent_rank, 11, comm, &status_recv[0]);
	
	MPI_Recv(B12_B22, size, MPI_INT, parent_rank, 1222, comm, &status_recv[1]);
	MPI_Recv(A11_A12, size, MPI_INT, parent_rank, 1112, comm, &status_recv[2]);
	MPI_Recv(B22, size, MPI_INT, parent_rank, 22, comm, &status_recv[3]);
	MPI_Recv(A21_A22, size, MPI_INT, parent_rank, 2122, comm, &status_recv[4]);
	MPI_Recv(B11, size, MPI_INT, parent_rank, 11, comm, &status_recv[5]);
	
	printf("[%d] RECEIV %d SIZE %d\n", my_rank, parent_rank, size);
	printf("[%d] RECEIV A11 %d STATUS %d A[0][0] %d \n", my_rank, parent_rank, status_recv[0].MPI_ERROR, A11[0][0]);
	
	int k = sqrt(size);
	printf("[%d] K %d\n",my_rank, k);
	int** P1 = strassen_parallel_mpi(A11, B12_B22, k, level, my_rank, max_rank, tag, comm);
	int** P2 = strassen_parallel_mpi(A11_A12, B22, k, level, my_rank, max_rank, tag, comm);
	int** P3 = strassen_parallel_mpi(A21_A22, B11, k, level, my_rank, max_rank, tag, comm);
	
	printf("[%d] SEND -> %d\n", my_rank, parent_rank);
	MPI_Send(P1, size, MPI_INT, parent_rank, tag+1, comm);
	MPI_Send(P2, size, MPI_INT, parent_rank, tag+2, comm);
	MPI_Send(P3, size, MPI_INT, parent_rank, tag+3, comm);
	printf("[%d] FIM SEND  %d\n", my_rank, parent_rank);
	return;
}

int **strassen_parallel_mpi(int **A, int **B, int size, int level, int my_rank, int max_rank, int tag, MPI_Comm comm)
{
	int helper_rank = my_rank + pow(2, level);
	if (helper_rank > max_rank)
	{
		printf("[%d] SERIAL helper_rank %d max_rank %d\n", my_rank, helper_rank, max_rank);
		return strassenMultiply(A, B, size);
	}
	else
	{
		int k = size / 2;
		printf("\n[%d] CHEGUEI 1.1 | %d SIZE %d \n", my_rank, k, size);

		MPI_Request request[6];
		MPI_Status status_send[6];
		MPI_Status status[3];

		int **A11 = initializeMatrix(k);
		int **A12 = initializeMatrix(k);
		int **A21 = initializeMatrix(k);
		int **A22 = initializeMatrix(k);
		int **B11 = initializeMatrix(k);
		int **B12 = initializeMatrix(k);
		int **B21 = initializeMatrix(k);
		int **B22 = initializeMatrix(k);
		printf("[%d] CHEGUEI 1.2 | %d SIZE %d \n", my_rank, k, size);

		for (int i = 0; i < k; i++)
		{
			for (int j = 0; j < k; j++)
			{
				A11[i][j] = A[i][j];
				A12[i][j] = A[i][k + j];
				A21[i][j] = A[k + i][j];
				A22[i][j] = A[k + i][k + j];
				if(my_rank == 1) { printf("i %d %d\n", i, j); }
				B11[i][j] = B[i][j];
				B12[i][j] = B[i][k + j];
				B21[i][j] = B[k + i][j];
				B22[i][j] = B[k + i][k + j];
				if(my_rank == 1) { printf("i %d %d\n", i, j); }
			}
		}
		printf("[%d] DIVISAO %d PARA %d SIZE %d\n\n", my_rank, level, helper_rank, k*k);
        MPI_Send(A11, k*k, MPI_INT, helper_rank, 11, comm);
        MPI_Send(subtract(B12, B22, k), k*k, MPI_INT, helper_rank, 1222, comm);
        MPI_Send(add(A11, A12, k), k*k, MPI_INT, helper_rank, 1112, comm);
        MPI_Send(B22, k*k, MPI_INT, helper_rank, 22, comm);
        MPI_Send(add(A21, A22, k), k*k, MPI_INT, helper_rank, 2122, comm);
        MPI_Send(B11, k*k, MPI_INT, helper_rank, 11, comm);
		printf("[%d] Enviado para %d SIZE %d\n\n", my_rank, helper_rank, k);

		int** P1 = initializeMatrix(k);
		int** P2 = initializeMatrix(k);
		int** P3 = initializeMatrix(k);

		int** P4 = strassen_parallel_mpi(A22, subtract(B21, B11, k), k, level + 1, my_rank, max_rank, tag, comm);
		int** P5 = strassen_parallel_mpi(add(A11, A22, k), add(B11, B22, k), k, level + 1, my_rank, max_rank, tag, comm);
		int** P6 = strassen_parallel_mpi(subtract(A12, A22, k), add(B21, B22, k), k, level + 1, my_rank, max_rank, tag, comm);
		int** P7 = strassen_parallel_mpi(subtract(A11, A21, k), add(B11, B12, k), k, level + 1, my_rank, max_rank, tag, comm);

		// MPI_Waitall(5, request, status_send);
		
		printf("\n[%d] Recebendo de %d \n", my_rank, helper_rank);
		MPI_Recv(P1, k*k, MPI_INT, helper_rank, tag+1, comm, &status[0]);
		MPI_Recv(P2, k*k, MPI_INT, helper_rank, tag+2, comm, &status[1]);
		MPI_Recv(P3, k*k, MPI_INT, helper_rank, tag+3, comm, &status[2]);

		int** C = initializeMatrix(k*k);
		int** C11 = subtract(add(add(P5, P4, k), P6, k), P2, k);
		int** C12 = add(P1, P2, k);
		int** C21 = add(P3, P4, k);
		int** C22 = subtract(subtract(add(P5, P1, k), P3, k), P7, k);

		for(int i = 0; i < k; i++) {
		    for(int j = 0; j < k; j++) {
		        C[i][j] = C11[i][j];
		        C[i][j+k] = C12[i][j];
		        C[k+i][j] = C21[i][j];
		        C[k+i][k+j] = C22[i][j];
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

int** run_root_mpi(int **A, int **B, int n, int max_rank, int tag, MPI_Comm comm)
{
	int my_rank; 
	MPI_Comm_rank(comm, &my_rank);
	if (my_rank != 0)
	{
		printf("Error: run_root_mpi called from process %d; must be called from process 0 only\n",
			   my_rank);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	printf("[%d] inicio do strassen_parallel_mpi\n", my_rank);
	return strassen_parallel_mpi(A, B, n, 0, my_rank, max_rank, tag, comm); 
}

int **initializeMatrix(int n)
{
	int **temp = (int**)malloc((n + 1) * sizeof(int *));
	if (!temp)
    	printf("allocation failure 1 in matrix()");
	temp[0] = (int*) malloc((n*n + 1) * sizeof(int));
	for (int i = 0+1; i <= n; i++)
		temp[i] = temp[i - 1] + n;
	return temp;
}

void allocMatrixes(int **A, int **B, int **C, int n)
{
	//// Matrix A ////
	A = initializeMatrix(n);
	B = initializeMatrix(n);
	C = initializeMatrix(n);

	srand(time(0));

	for (int l = 0; l < n; l++)
	{
		for (int c = 0; c < n; c++)
		{
			A[l][c] = rand() % 31;
			B[l][c] = rand() % 31;
		}
	}
}

void printMatrix(int **M, int n)
{
	for (int l = 0; l < n; l++)
	{
		for (int c = 0; c < n; c++)
		{
			printf("%d ", M[l][c]);
		}
		printf("\n");
	}
	printf("\n");
}

int **add(int **M1, int **M2, int n)
{
	int **temp = initializeMatrix(n);
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			temp[i][j] = M1[i][j] + M2[i][j];

	return temp;
}

int **subtract(int **M1, int **M2, int n)
{
	int **temp = initializeMatrix(n);
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			temp[i][j] = M1[i][j] - M2[i][j];

	return temp;
}

int **strassenMultiply(int **A, int **B, int n)
{
	int **C = initializeMatrix(n);
	int k = n / 2;

	if (n == 1)
	{
		int **C = initializeMatrix(1);
		C[0][0] = A[0][0] * B[0][0];
		return C;
	}
	else
	{
		int **A11 = initializeMatrix(k);
		int **A12 = initializeMatrix(k);
		int **A21 = initializeMatrix(k);
		int **A22 = initializeMatrix(k);
		int **B11 = initializeMatrix(k);
		int **B12 = initializeMatrix(k);
		int **B21 = initializeMatrix(k);
		int **B22 = initializeMatrix(k);

		for (int i = 0; i < k; i++)
		{
			for (int j = 0; j < k; j++)
			{
				A11[i][j] = A[i][j];
				A12[i][j] = A[i][k + j];
				A21[i][j] = A[k + i][j];
				A22[i][j] = A[k + i][k + j];
				B11[i][j] = B[i][j];
				B12[i][j] = B[i][k + j];
				B21[i][j] = B[k + i][j];
				B22[i][j] = B[k + i][k + j];
			}
		}

		int **P1 = strassenMultiply(A11, subtract(B12, B22, k), k);
		int **P2 = strassenMultiply(add(A11, A12, k), B22, k);
		int **P3 = strassenMultiply(add(A21, A22, k), B11, k);
		int **P4 = strassenMultiply(A22, subtract(B21, B11, k), k);
		int **P5 = strassenMultiply(add(A11, A22, k), add(B11, B22, k), k);
		int **P6 = strassenMultiply(subtract(A12, A22, k), add(B21, B22, k), k);
		int **P7 = strassenMultiply(subtract(A11, A21, k), add(B11, B12, k), k);

		int **C11 = subtract(add(add(P5, P4, k), P6, k), P2, k);
		int **C12 = add(P1, P2, k);
		int **C21 = add(P3, P4, k);
		int **C22 = subtract(subtract(add(P5, P1, k), P3, k), P7, k);

		for (int i = 0; i < k; i++)
		{
			for (int j = 0; j < k; j++)
			{
				C[i][j] = C11[i][j];
				C[i][j + k] = C12[i][j];
				C[k + i][j] = C21[i][j];
				C[k + i][k + j] = C22[i][j];
			}
		}

		return C;
	}
}
