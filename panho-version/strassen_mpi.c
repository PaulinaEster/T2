#include "strassen_mpi.h"

/**
 * Hierarchical Divide-and-Conquer MPI Strassen Multiplication
 * 
 * Tree Structure:
 *   - Each process can distribute work to child processes
 *   - Children are assigned sequentially: rank uses (rank+1) to (rank+7)
 *   - Distribution continues until MAX_TREE_HEIGHT or no more processes
 * 
 * Example with 8 processes, 128x128 matrix:
 *   Level 0: Rank 0 (128x128) → distributes P1-P7 to ranks 1-7
 *   Level 1: Rank 1 (64x64)   → could distribute to ranks 2-8 (if available)
 *   Level 2: Each continues recursively until threshold or max depth
 */

int** strassenMultiplyMPI(int** A, int** B, int n, int rank, int num_procs, int level) {
    int k = n / 2;

    if (n == 1) {
        int** C = initializeMatrix(1);
        C[0][0] = A[0][0] * B[0][0];
        return C;
    }
    
    // Use standard multiplication for small matrices
    if (n <= MIN_SIZE_THRESHOLD) {
        return standardMultiply(A, B, n);
    }

    int** A11 = initializeMatrix(k);
    int** A12 = initializeMatrix(k);
    int** A21 = initializeMatrix(k);
    int** A22 = initializeMatrix(k);

    int** B11 = initializeMatrix(k);
    int** B12 = initializeMatrix(k);
    int** B21 = initializeMatrix(k);
    int** B22 = initializeMatrix(k);

    splitMatrix(A, A11, A12, A21, A22, k);
    splitMatrix(B, B11, B12, B21, B22, k);

    // Initialize result matrices for P1-P7
    int** P[7] = {NULL};

    // Decide whether to distribute work among processes
    // Check if matrix is large enough and we have available child processes
    if (shouldDistribute(n, level, num_procs, rank)) {
        // Calculate available child processes for this rank
        // Use a proper tree structure: rank uses children at positions rank*7+1 to rank*7+7
        // This ensures no overlap: rank 0 -> 1-7, rank 1 -> 8-14, rank 2 -> 15-21, etc.
        int num_children = 0;
        for (int i = 0; i < 7; i++) {
            int child_rank = rank * 7 + (i + 1);
            if (child_rank < num_procs) {
                num_children++;
            } else {
                break;
            }
        }
        
        // Send P1-P7 computations to available child processes
        for (int i = 0; i < num_children; i++) {
            int child_rank = rank * 7 + (i + 1); // Non-overlapping tree structure
            
            // Prepare matrices for Strassen product Pi
            int** tempA = NULL; int** tempB = NULL;
            
            switch (i) {
                case 0: // P1 = (A11 + A22) * (B11 + B22)
                    tempA = addMatrices(A11, A22, k);
                    tempB = addMatrices(B11, B22, k);
                    break;
                case 1: // P2 = (A21 + A22) * B11
                    tempA = addMatrices(A21, A22, k);
                    tempB = initializeMatrix(k);
                    copyMatrix(B11, tempB, k);
                    break;
                case 2: // P3 = A11 * (B12 - B22)
                    tempA = initializeMatrix(k);
                    copyMatrix(A11, tempA, k);
                    tempB = subtractMatrices(B12, B22, k);
                    break;
                case 3: // P4 = A22 * (B21 - B11)
                    tempA = initializeMatrix(k);
                    copyMatrix(A22, tempA, k);
                    tempB = subtractMatrices(B21, B11, k);
                    break;
                case 4: // P5 = (A11 + A12) * B22
                    tempA = addMatrices(A11, A12, k);
                    tempB = initializeMatrix(k);
                    copyMatrix(B22, tempB, k);
                    break;
                case 5: // P6 = (A21 - A11) * (B11 + B12)
                    tempA = subtractMatrices(A21, A11, k);
                    tempB = addMatrices(B11, B12, k);
                    break;
                case 6: // P7 = (A12 - A22) * (B21 + B22)
                    tempA = subtractMatrices(A12, A22, k);
                    tempB = addMatrices(B21, B22, k);
                    break;
            }
            
            // Send work to child process
            // Format: combined matrix data (A then B) only
            // Combine both matrices into a single buffer: [flatA][flatB]
            int* combined = (int*)malloc(2 * k * k * sizeof(int));
            serializeMatrix(tempA, combined, k);
            serializeMatrix(tempB, combined + k * k, k);
            
            // Send combined matrix data in one message
            MPI_Send(combined, 2 * k * k, MPI_INT, child_rank, TAG_MATRIZ + 1, MPI_COMM_WORLD);
            
            free(combined);
            freeMatrix(tempA, k);
            freeMatrix(tempB, k);
        }
        
        // 2b️⃣ Receive results from children and compute remaining P locally
        for (int i = 0; i < 7; i++) {
            if (i < num_children) {
                // Receive result from child process
                int child_rank = rank * 7 + (i + 1);
                int* flatResult = (int*)malloc(k * k * sizeof(int));
                MPI_Recv(flatResult, k * k, MPI_INT, child_rank, TAG_RESULTADO, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                P[i] = unflattenMatrix(flatResult, k);
                free(flatResult);
            } else {
                // Compute remaining products locally (no more children available)
                P[i] = computeStrassenProductMPI(A11, A12, A21, A22, B11, B12, B21, B22, k, rank, num_procs, level, i);
            }
        }
    } else {
        // No distribution - compute all P1-P7 locally
        // This happens when: matrix too small, no available processes, or max depth reached
        for (int i = 0; i < 7; i++) {
            P[i] = computeStrassenProductMPI(A11, A12, A21, A22, B11, B12, B21, B22, k, rank, num_procs, level, i);
        }
    }
    
    // 4️⃣ Combine results (P1..P7 → C11..C22 → C)
    int** temp1, **temp2;
    
    // C11 = P1 + P4 - P5 + P7
    temp1 = addMatrices(P[0], P[3], k);
    temp2 = subtractMatrices(temp1, P[4], k);
    int** C11 = addMatrices(temp2, P[6], k);
    freeMatrix(temp1, k);
    freeMatrix(temp2, k);
    
    // C12 = P3 + P5
    int** C12 = addMatrices(P[2], P[4], k);
    
    // C21 = P2 + P4
    int** C21 = addMatrices(P[1], P[3], k);
    
    // C22 = P1 - P2 + P3 + P6
    temp1 = subtractMatrices(P[0], P[1], k);
    temp2 = addMatrices(temp1, P[2], k);
    int** C22 = addMatrices(temp2, P[5], k);
    freeMatrix(temp1, k);
    freeMatrix(temp2, k);
    
    // Combine quadrants into final result
    int** C = initializeMatrix(n);
    combineBlocks(C, C11, C12, C21, C22, k);
    
    // Free all allocated memory
    freeMatrix(A11, k); freeMatrix(A12, k); freeMatrix(A21, k); freeMatrix(A22, k);
    freeMatrix(B11, k); freeMatrix(B12, k); freeMatrix(B21, k); freeMatrix(B22, k);
    for (int i = 0; i < 7; i++) {
        if (P[i]) freeMatrix(P[i], k);
    }
    freeMatrix(C11, k); freeMatrix(C12, k); freeMatrix(C21, k); freeMatrix(C22, k);
    
    return C;
}

// Helper function to compute individual Strassen products with recursive divide-and-conquer
int** computeStrassenProductMPI(int** A11, int** A12, int** A21, int** A22, 
                               int** B11, int** B12, int** B21, int** B22, 
                               int k, int rank, int num_procs, int level, int product_index) {
    int** tempA = NULL;
    int** tempB = NULL;
    int** result = NULL;
    
    // Prepare matrices according to Strassen's formulas
    switch (product_index) {
        case 0: // P1 = (A11 + A22) * (B11 + B22)
            tempA = addMatrices(A11, A22, k);
            tempB = addMatrices(B11, B22, k);
            break;
        case 1: // P2 = (A21 + A22) * B11
            tempA = addMatrices(A21, A22, k);
            tempB = initializeMatrix(k);
            copyMatrix(B11, tempB, k);
            break;
        case 2: // P3 = A11 * (B12 - B22)
            tempA = initializeMatrix(k);
            copyMatrix(A11, tempA, k);
            tempB = subtractMatrices(B12, B22, k);
            break;
        case 3: // P4 = A22 * (B21 - B11)
            tempA = initializeMatrix(k);
            copyMatrix(A22, tempA, k);
            tempB = subtractMatrices(B21, B11, k);
            break;
        case 4: // P5 = (A11 + A12) * B22
            tempA = addMatrices(A11, A12, k);
            tempB = initializeMatrix(k);
            copyMatrix(B22, tempB, k);
            break;
        case 5: // P6 = (A21 - A11) * (B11 + B12)
            tempA = subtractMatrices(A21, A11, k);
            tempB = addMatrices(B11, B12, k);
            break;
        case 6: // P7 = (A12 - A22) * (B21 + B22)
            tempA = subtractMatrices(A12, A22, k);
            tempB = addMatrices(B21, B22, k);
            break;
    }

    result = strassenMultiplyMPI(tempA, tempB, k, rank, num_procs, level + 1);

    freeMatrix(tempA, k);
    freeMatrix(tempB, k);

    return result;
}

// Determine if work should be distributed among processes
int shouldDistribute(int n, int level, int num_procs, int rank) {
    // Distribute if:
    // 1. Haven't reached maximum tree depth
    // 2. There are available child processes (first child = rank*7+1 must exist)
    // 3. Matrix is large enough to benefit from parallelism (children will get n/2)
    int first_child = rank * 7 + 1;
    return (level < MAX_TREE_HEIGHT && 
            first_child < num_procs &&
            n > MIN_SIZE_THRESHOLD);
}