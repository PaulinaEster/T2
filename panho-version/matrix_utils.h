#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// MPI communication tags
#define TAG_MATRIZ 100
#define TAG_RESULTADO 200

// Matrix operations
int** initializeMatrix(int n);
void copyMatrix(int** source, int** dest, int n);
void freeMatrix(int** matrix, int n);
void printMatrix(int** matrix, int n, const char* name);
int** addMatrices(int** A, int** B, int n);
int** subtractMatrices(int** A, int** B, int n);

// Matrix splitting and combining for Strassen
void splitMatrix(int** parent, int** A11, int** A12, int** A21, int** A22, int k);
void combineBlocks(int** C, int** C11, int** C12, int** C21, int** C22, int k);

// Matrix serialization for MPI communication
void serializeMatrix(int** matrix, int* buffer, int n);
void deserializeMatrix(int* buffer, int** matrix, int n);
int* flattenMatrix(int** matrix, int n);
int** unflattenMatrix(int* flat, int n);

// Sequential Strassen and Standard multiplication
int** strassenMultiply(int** A, int** B, int n);
int** standardMultiply(int** A, int** B, int n);

// Utility
int isPowerOfTwo(int n);

#endif // MATRIX_UTILS_H