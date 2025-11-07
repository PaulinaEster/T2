#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <mpi.h>
#include<math.h>

int **A;
int **B;
int **C;

int **initializeMatrix(int n);
int **add(int **A, int **B, int n);
int **subtract(int **A, int **B, int n);
void freeMatrix(int **M, int n);

int **strassenMultiplyMPI(int **A, int **B, int n, int level, int rank, int max_rank, int tag, MPI_Comm comm);
int **run_root_mpi(int n, int level, int rank, int max_rank, int tag, MPI_Comm comm);

void printMatrix(int **M, int n);

int **initializeMatrix(int n)
{
	//// Matrix A ////
	int **temp = (int **)malloc(n * sizeof(int *));
	for (int i = 0; i < n; i++)
		temp[i] = (int *)malloc(n * sizeof(int));

	return temp;
}

void allocMatrixes(int n)
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

void GenerateMatrixes(int n)
{
	printf("\nSerão geradas 2 matrizes %d x %d de inteiros aleatórios\n", n, n);

	allocMatrixes(n);
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

// Root process code
int **run_root_mpi(int n, int max_rank, int tag, MPI_Comm comm)
{
	int my_rank;
	MPI_Comm_rank(comm, &my_rank);
	if (my_rank != 0)
	{
		printf("Error: run_root_mpi called from process %d; must be called from process 0 only\n",
			   my_rank);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	return strassenMultiplyMPI(A, B, n, 0, my_rank, max_rank, tag, comm);
}

int **strassenMultiplyMPI(int **A, int **B, int n, int level, int rank, int max_rank, int tag, MPI_Comm comm)
{
	int **C = initializeMatrix(n);
	int k = n / 2;

	// Caso base
	if (n == 1)
	{
		printf("[%d] n == 1\n", rank);
		C[0][0] = A[0][0] * B[0][0];
		return C;
	}

	// Divide A e B em 4 submatrizes
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

	// Define processo auxiliar (helper)
	int helper_rank = rank + (int)pow(2, level);

	if (helper_rank > max_rank)
	{
		// Nenhum processo livre -> resolve localmente (recursão pura)
		int **P1 = strassenMultiplyMPI(A11, subtract(B12, B22, k), k, level + 1, rank, max_rank, tag, comm);
		int **P2 = strassenMultiplyMPI(add(A11, A12, k), B22, k, level + 1, rank, max_rank, tag, comm);
		int **P3 = strassenMultiplyMPI(add(A21, A22, k), B11, k, level + 1, rank, max_rank, tag, comm);
		int **P4 = strassenMultiplyMPI(A22, subtract(B21, B11, k), k, level + 1, rank, max_rank, tag, comm);
		int **P5 = strassenMultiplyMPI(add(A11, A22, k), add(B11, B22, k), k, level + 1, rank, max_rank, tag, comm);
		int **P6 = strassenMultiplyMPI(subtract(A12, A22, k), add(B21, B22, k), k, level + 1, rank, max_rank, tag, comm);
		int **P7 = strassenMultiplyMPI(subtract(A11, A21, k), add(B11, B12, k), k, level + 1, rank, max_rank, tag, comm);

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
	else
	{
		MPI_Request req;
		MPI_Status status;

		// Prepara dados para envio (metade do trabalho)
		// Aqui mandamos as submatrizes necessárias para o helper
		// Exemplo: helper resolve P1..P3 e o processo atual resolve P4..P7
		int ***send_data = (int***)malloc(6 * sizeof(int **)); // ajustável
		send_data[0] = A11;
		send_data[1] = A12;
		send_data[2] = A21;
		send_data[3] = A22;
		send_data[4] = B11;
		send_data[5] = B22;

		// Envia tamanho
		MPI_Isend(&k, 1, MPI_INT, helper_rank, tag, comm, &req);

		// Envia matrizes
		for (int m = 0; m < 6; m++)
		{
			printf("[%d] Enviando para %d \n", rank, helper_rank);
			MPI_Isend(send_data[m], k * k, MPI_INT, helper_rank, tag + m + 1, comm, &req);
		}

		// Resolve localmente parte dos P’s
		int **P4 = strassenMultiplyMPI(A22, subtract(B21, B11, k), k, level + 1, rank, max_rank, tag, comm);
		int **P5 = strassenMultiplyMPI(add(A11, A22, k), add(B11, B22, k), k, level + 1, rank, max_rank, tag, comm);
		int **P6 = strassenMultiplyMPI(subtract(A12, A22, k), add(B21, B22, k), k, level + 1, rank, max_rank, tag, comm);
		int **P7 = strassenMultiplyMPI(subtract(A11, A21, k), add(B11, B12, k), k, level + 1, rank, max_rank, tag, comm);

		// Libera request (o envio completa de forma assíncrona)
		MPI_Request_free(&req);

		// Recebe resultados do helper (P1..P3)
		int **P1 = initializeMatrix(k);
		int **P2 = initializeMatrix(k);
		int **P3 = initializeMatrix(k);

		MPI_Recv(P1, k * k, MPI_INT, helper_rank, tag + 100, comm, &status);
		printf("[%d] Recebe de %d status %d\n", rank, helper_rank, status.MPI_SOURCE);
		MPI_Recv(P2, k * k, MPI_INT, helper_rank, tag + 101, comm, &status);
		printf("[%d] Recebe de %d status %d\n", rank, helper_rank, status.MPI_SOURCE);

		MPI_Recv(P3, k * k, MPI_INT, helper_rank, tag + 102, comm, &status);
		printf("[%d] Recebe de %d status %d\n", rank, helper_rank, status.MPI_SOURCE);


		// Combina tudo
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

int my_topmost_level_mpi(int my_rank)
{
    int level = 0;
    while (pow(2, level) <= my_rank)
        level++;
    return level;
}

void run_helper_mpi(int my_rank, int max_rank, int tag, MPI_Comm comm)
{
    int level = my_topmost_level_mpi(my_rank);
    // probe for a message and determine its size and sender
    MPI_Status status;
    int size;
    MPI_Probe(MPI_ANY_SOURCE, tag, comm, &status);
    MPI_Get_count(&status, MPI_INT, &size);
    int parent_rank = status.MPI_SOURCE;
    // allocate int a[size], temp[size]

	int **A11 = initializeMatrix(k);
	int **A12 = initializeMatrix(k);
	int **A21 = initializeMatrix(k);
	int **A22 = initializeMatrix(k);
	int **B11 = initializeMatrix(k);
	int **B12 = initializeMatrix(k);
	int ***receiv_data = (int***)malloc(6 * sizeof(int **)); // ajustável
		receiv_data[0] = A11;
		receiv_data[1] = A12;
		receiv_data[2] = A21;
		receiv_data[3] = A22;
		receiv_data[4] = B11;
		receiv_data[5] = B12;
	for (int m = 0; m < 6; m++)
	{
		MPI_Recv(receiv_data[m], size*size, MPI_INT, parent_rank, tag, comm, &status);
	}
   	int** P1 = strassenMultiply(A11, subtract(B12, B22, k), k);
	int** P2 = strassenMultiply(add(A11, A12, k), B22, k);
	int** P3 = strassenMultiply(add(A21, A22, k), B11, k);
    // Send sorted array to parent process
    MPI_Send(a, size, MPI_INT, parent_rank, tag, comm);
    return;
}

int main(int argc, char *argv[])
{
	// All processes
	MPI_Init(&argc, &argv);
	// Check processes and their ranks
	// number of processes == communicator size
	int comm_size;
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	int max_rank = comm_size - 1;
	int tag = 123;
	int n = atoi(argv[1]);
	if (my_rank == 0)
	{ // Only root process sets test data
		puts("-MPI Recursive Mergesort-\t");
		// Check arguments
		if (argc != 2) /* argc must be 2 for proper execution! */
		{
			printf("Usage: %s array-size\n", argv[0]);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		// Get argument
		int size = atoi(argv[1]); // Array size
		printf("Matrix size = %d\nProcesses = %d\n", size, comm_size);

		GenerateMatrixes(size);
		if (A == NULL || B == NULL || C == NULL)
		{
			printf("Error: Could not allocate array of size %d\n", size);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		// Sort with root process
		// double start = get_time();
		C = run_root_mpi(size, max_rank, tag, MPI_COMM_WORLD);
		// double end = get_time();
	} else {
        run_helper_mpi(my_rank, max_rank, tag, MPI_COMM_WORLD);
	}
	MPI_Finalize();

	return 0;
}