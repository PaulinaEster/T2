#include "strassen_mpi.h"
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

void initializeRandomMatrix(int** matrix, int n, int seed);
void verifyResult(int** A, int** B, int** C, int n);
int** sequentialStandardMultiply(int** A, int** B, int n);
void workerProcess(int rank, int num_procs);


int main(int argc, char* argv[]) {
    int rank, num_procs;
    int n = 4;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc > 1) {
        n = atoi(argv[1]);
        if (!isPowerOfTwo(n) || n < 2) {
            if (rank == 0) {
                printf("Error: Matrix size must be a power of 2 and >= 2\n");
                printf("Usage: %s [matrix_size]\n", argv[0]);
            }
            MPI_Finalize();
            return 0;
        }
    }

    if (rank == 0) {
        printf("=== MPI Strassen Matrix Multiplication ===\n");
        printf("Matrix size: %dx%d\n", n, n);
        printf("Number of processes: %d\n", num_procs);
        printf("Tree height limit: %d\n", MAX_TREE_HEIGHT);
        printf("Sequential threshold: %d\n", MIN_SIZE_THRESHOLD);
        printf("==========================================\n\n");

        int** A = initializeMatrix(n);
        int** B = initializeMatrix(n);

        initializeRandomMatrix(A, n, 123);
        initializeRandomMatrix(B, n, 456);

        if (n <= 8) {
            printMatrix(A, n, "A");
            printMatrix(B, n, "B");
        }

        clock_t start_time = clock();
        double mpi_start_time = MPI_Wtime();

        printf("Starting MPI Strassen multiplication...\n");
        int** C = strassenMultiplyMPI(A, B, n, rank, num_procs, 0);
        
        double mpi_end_time = MPI_Wtime();
        clock_t end_time = clock();
        
        double cpu_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
        double wall_time = mpi_end_time - mpi_start_time;
        
        printf("MPI Strassen multiplication completed!\n");
        printf("CPU Time: %.6f seconds\n", cpu_time);
        printf("Wall Time: %.6f seconds\n", wall_time);

        if (n <= 8) {
            printMatrix(C, n, "Result C");
        }

        if (n <= 128) {
            printf("\nVerifying result with standard multiplication...\n");
            clock_t verify_start = clock();
            int** C_verify = sequentialStandardMultiply(A, B, n);
            clock_t verify_end = clock();

            double verify_time = ((double)(verify_end - verify_start)) / CLOCKS_PER_SEC;
            printf("Standard multiplication time: %.6f seconds\n", verify_time);

            int correct = 1;
            for (int i = 0; i < n && correct; i++) {
                for (int j = 0; j < n && correct; j++) {
                    if (C[i][j] != C_verify[i][j]) {
                        correct = 0;
                        printf("Mismatch at [%d][%d]: Strassen=%d, Standard=%d\n", 
                               i, j, C[i][j], C_verify[i][j]);
                    }
                }
            }

            if (correct) {
                printf("Verification PASSED - Results match!\n");
                printf("Speedup: %.2fx\n", verify_time / wall_time);
            } else {
                printf("Verification FAILED - Results do not match!\n");
            }
            
            freeMatrix(C_verify, n);
        }

        freeMatrix(A, n);
        freeMatrix(B, n);
        freeMatrix(C, n);
    } else {
        // Only enter worker mode if there's a possibility of receiving work
        // Workers are only used if matrix size > MIN_SIZE_THRESHOLD
        if (n > MIN_SIZE_THRESHOLD) {
            workerProcess(rank, num_procs);
        }
        // Otherwise, worker process just terminates without doing anything
    }
    
    MPI_Finalize();
    return 0;
}


void workerProcess(int rank, int num_procs) {
    MPI_Status status;
    
    // Receive combined matrix data and use MPI_Get_count to discover size
    MPI_Probe(MPI_ANY_SOURCE, TAG_MATRIZ + 1, MPI_COMM_WORLD, &status);
    int total_elements;
    MPI_Get_count(&status, MPI_INT, &total_elements);
    
    // Get the parent rank from the status
    int parent_rank = status.MPI_SOURCE;
    
    // total_elements = 2 * k * k, so k = sqrt(total_elements / 2)
    int k = (int)sqrt(total_elements / 2);
    
    // Receive combined buffer containing both matrices
    int* combined = (int*)malloc(total_elements * sizeof(int));
    MPI_Recv(combined, total_elements, MPI_INT, parent_rank, TAG_MATRIZ + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    // Deserialize both matrices from the combined buffer
    int** A = initializeMatrix(k);
    int** B = initializeMatrix(k);
    deserializeMatrix(combined, A, k);              // First half
    deserializeMatrix(combined + k * k, B, k);      // Second half
    
    // Perform computation starting at level 0 (worker treats this as a fresh subproblem)
    int** result = strassenMultiplyMPI(A, B, k, rank, num_procs, 0);
    
    // Send result back to parent (source of the work)
    int* flatResult = flattenMatrix(result, k);
    MPI_Send(flatResult, k * k, MPI_INT, parent_rank, TAG_RESULTADO, MPI_COMM_WORLD);
    
    // Clean up
    free(combined);
    free(flatResult);
    freeMatrix(A, k);
    freeMatrix(B, k);
    freeMatrix(result, k);
    
    // Worker naturally terminates after completing its task
}

// Initialize matrix with random values
void initializeRandomMatrix(int** matrix, int n, int seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = rand() % 10; // Values 0-9 for easy verification
        }
    }
}

// Sequential standard matrix multiplication for verification
int** sequentialStandardMultiply(int** A, int** B, int n) {
    int** C = initializeMatrix(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}