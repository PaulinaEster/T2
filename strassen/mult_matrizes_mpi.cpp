// -n 8 specifies number of processes
// Run: mpiexec -n 8 TPP2\mult_matrizes_mpi.exe

#include <mpi.h>
#include <iostream>
#include <vector>
#include <utility>
#include <chrono>
#include <cstdlib>
#include <ctime>

using namespace std;

static bool isPowerOfTwo(int n) {
    return (n > 0) && ((n & (n - 1)) == 0);
}

vector<vector<int>> generateMatrix(int n) {
    vector<vector<int>> M(n, vector<int>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            M[i][j] = rand() % 31;
    return M;
}

vector<int> flattenMatrix(const vector<vector<int>>& M) {
    int n = (int)M.size();
    vector<int> flat(n * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            flat[i * n + j] = M[i][j];
    return flat;
}

vector<vector<int>> unflattenMatrix(const vector<int>& flat, int n) {
    vector<vector<int>> M(n, vector<int>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            M[i][j] = flat[i * n + j];
    return M;
}


vector<vector<int>> getSubmatrix(const vector<vector<int>>& M, int rowStart, int colStart, int size) {
    vector<vector<int>> sub(size, vector<int>(size));
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            sub[i][j] = M[rowStart + i][colStart + j];
    return sub;
}


void printMatrix(const vector<vector<int>>& M) {
    for (const auto& row : M) {
        for (int val : row)
            std::cout << val << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

vector<vector<int>> add(const vector<vector<int>>& M1, const vector<vector<int>>& M2, int n) {
    vector<vector<int>> temp(n, vector<int>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            temp[i][j] = M1[i][j] + M2[i][j];
    return temp;
}

vector<vector<int>> subtract(const vector<vector<int>>& M1, const vector<vector<int>>& M2, int n) {
    vector<vector<int>> temp(n, vector<int>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            temp[i][j] = M1[i][j] - M2[i][j];
    return temp;
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check number of processes
    if (size < 8) {
        if (rank == 0)
            std::cout << "This program requires at least 8 MPI processes.\n";
        MPI_Finalize();
        return 0;
    }

    // Broadcast matrix size 'n' from master to all processes. Master sets n; others receive.
    int n = 0;
    if (rank == 0) {
        n = atoi(argv[1]);
        if (n % 2 != 0 || !isPowerOfTwo(n)) {
            std::cout << "Invalid value! The algorithm requires N to be even and a power of 2.\n";
            n = -1; // indicate invalid
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (n == -1) {
        // All processes exit
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        // Master
        srand((unsigned)time(nullptr));
        std::cout << "Master process generating matrices A and B of size " << n << "x" << n << "...\n";
        vector<vector<int>> A = generateMatrix(n);
        vector<vector<int>> B = generateMatrix(n);

        if (n <= 16) { // only print small matrices
            std::cout << "Matrix A:\n";
            printMatrix(A);
            std::cout << "Matrix B:\n";
            printMatrix(B);
        }

        int k = n / 2;
        // Divide A
        auto A11 = getSubmatrix(A, 0, 0, k);
        auto A12 = getSubmatrix(A, 0, k, k);
        auto A21 = getSubmatrix(A, k, 0, k);
        auto A22 = getSubmatrix(A, k, k, k);
        // Divide B
        auto B11 = getSubmatrix(B, 0, 0, k);
        auto B12 = getSubmatrix(B, 0, k, k);
        auto B21 = getSubmatrix(B, k, 0, k);
        auto B22 = getSubmatrix(B, k, k, k);

        // Prepare pairs
        vector<pair<vector<vector<int>>, vector<vector<int>>>> pairs = {
            {A11, subtract(B12, B22, k)},                  // P1
            {add(A11, A12, k), B22},                       // P2
            {add(A21, A22, k), B11},                       // P3
            {A22, subtract(B21, B11, k)},                  // P4
            {add(A11, A22, k), add(B11, B22, k)},          // P5
            {subtract(A12, A22, k), add(B21, B22, k)},     // P6
            {subtract(A11, A21, k), add(B11, B12, k)}      // P7
        };

        // Send k and matrices to ranks 1..7
        for (int i = 0; i < 7; ++i) {
            vector<int> flatA = flattenMatrix(pairs[i].first);
            vector<int> flatB = flattenMatrix(pairs[i].second);
            MPI_Send(&k, 1, MPI_INT, i + 1, 10, MPI_COMM_WORLD);
            MPI_Send(flatA.data(), k * k, MPI_INT, i + 1, 11, MPI_COMM_WORLD);
            MPI_Send(flatB.data(), k * k, MPI_INT, i + 1, 12, MPI_COMM_WORLD);
        }

        // Receive P1..P7
        vector<vector<vector<int>>> P(7, vector<vector<int>>(k, vector<int>(k)));
        for (int i = 0; i < 7; ++i) {
            vector<int> flatP(k * k);
            MPI_Recv(flatP.data(), k * k, MPI_INT, i + 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            P[i] = unflattenMatrix(flatP, k);
        }

        // Assemble C
        auto C11 = subtract(add(add(P[4], P[3], k), P[5], k), P[1], k); // P5 + P4 + P6 - P2
        auto C12 = add(P[0], P[1], k);                                  // P1 + P2
        auto C21 = add(P[2], P[3], k);                                  // P3 + P4
        auto C22 = subtract(subtract(add(P[4], P[0], k), P[2], k), P[6], k); // P5 + P1 - P3 - P7

        vector<vector<int>> C(n, vector<int>(n));
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < k; ++j) {
                C[i][j] = C11[i][j];
                C[i][j + k] = C12[i][j];
                C[i + k][j] = C21[i][j];
                C[i + k][j + k] = C22[i][j];
            }

        if (n <= 16) {
            std::cout << "Multiplier result (matrix C = A x B via Strassen):\n";
            printMatrix(C);
        } else {
            std::cout << "Multiplication completed. (matrix too large to print)\n";
        }

    } else if (rank >= 1 && rank <= 7) {
        // Worker: receive k, A, B
        int k;
        MPI_Recv(&k, 1, MPI_INT, 0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        vector<int> flatA(k * k), flatB(k * k);
        MPI_Recv(flatA.data(), k * k, MPI_INT, 0, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(flatB.data(), k * k, MPI_INT, 0, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        vector<vector<int>> A = unflattenMatrix(flatA, k);
        vector<vector<int>> B = unflattenMatrix(flatB, k);

        vector<vector<int>> C(k, vector<int>(k));
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < k; ++j)
                for (int l = 0; l < k; ++l)
                    C[i][j] += A[i][l] * B[l][j];

        vector<int> flatC = flattenMatrix(C);
        MPI_Send(flatC.data(), k * k, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}