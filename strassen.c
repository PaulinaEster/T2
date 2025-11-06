#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

int **A;
int **B;
int **C;

int **initializeMatrix(int n);
int **add(int **A, int **B, int n);
int **subtract(int **A, int **B, int n);
void freeMatrix(int **M, int n);

void printMatrix(int **M, int n);

int **initializeMatrix(int n)
{
	//// Matrix A ////
	int **temp = (int **) malloc(n * sizeof(int *));
	for(int i = 0; i < n; i++)
		temp[i] = (int *) malloc(n * sizeof(int));

	return temp;
}

void allocMatrixes(int n)
{
	//// Matrix A ////
	A = initializeMatrix(n);
	B = initializeMatrix(n);
	C = initializeMatrix(n);

	srand(time(0));

	for(int l = 0; l < n; l++)
	{
		for(int c = 0; c < n; c++)
		{
			A[l][c] = rand()%31;
			B[l][c] = rand()%31;
		}
	}
}

int GenerateMatrixes(int n)
{
	printf("\nSerão geradas 2 matrizes %d x %d de inteiros aleatórios\n", n, n);

	allocMatrixes(n);
	sleep(3);

	system("clear");
	if(n <= 10) {
		printf("Matrizes geradas:\n\n");
		printf("Matriz A\n");
		printMatrix(A, n);
		printf("Matriz B\n");
		printMatrix(B, n);
	}

	return n;
}


void printMatrix(int **M, int n)
{
	for(int l = 0; l < n; l++) {
		for(int c = 0; c < n; c++) {
			printf("%d ", M[l][c]);
		}
		printf("\n");
	}
	printf("\n");	
	
}

int** add(int **M1, int **M2, int n)
{
    int **temp = initializeMatrix(n);
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            temp[i][j] = M1[i][j] + M2[i][j];

    return temp;
}

int **subtract(int **M1, int **M2, int n)
{
    int **temp = initializeMatrix(n);
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            temp[i][j] = M1[i][j] - M2[i][j];

    return temp;
}

int **strassenMultiply(int **A, int **B, int n)
{
	int** C = initializeMatrix(n);

  	if (n == 1) {
    	C[0][0] = A[0][0] * B[0][0];
    	return C;
	}
	else
	{
		int k = n/2;
		int **A11 = initializeMatrix(k);
		int **A12 = initializeMatrix(k);
		int **A21 = initializeMatrix(k);
		int **A22 = initializeMatrix(k);
		int **B11 = initializeMatrix(k);
		int **B12 = initializeMatrix(k);
		int **B21 = initializeMatrix(k);
		int **B22 = initializeMatrix(k);

		for(int i=0; i<k; i++) {
		    for(int j=0; j<k; j++) {
		        A11[i][j] = A[i][j];
		        A12[i][j] = A[i][k+j];
		        A21[i][j] = A[k+i][j];
		        A22[i][j] = A[k+i][k+j];
		        B11[i][j] = B[i][j];
		        B12[i][j] = B[i][k+j];
		        B21[i][j] = B[k+i][j];
		        B22[i][j] = B[k+i][k+j];
	   		}
		}

		int** P1 = strassenMultiply(A11, subtract(B12, B22, k), k);
		int** P2 = strassenMultiply(add(A11, A12, k), B22, k);
		int** P3 = strassenMultiply(add(A21, A22, k), B11, k);


		int** P4 = strassenMultiply(A22, subtract(B21, B11, k), k);
		int** P5 = strassenMultiply(add(A11, A22, k), add(B11, B22, k), k);
		int** P6 = strassenMultiply(subtract(A12, A22, k), add(B21, B22, k), k);
		int** P7 = strassenMultiply(subtract(A11, A21, k), add(B11, B12, k), k);

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

int main(int argc, char const *argv[])
{
	int n = atoi(argv[1]);

	n = GenerateMatrixes(n);

	printf("Multiplicando...\nResultado:\n\n");
	C = strassenMultiply(A, B, n);

	return 0;
}