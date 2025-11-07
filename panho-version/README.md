# MPI Strassen Matrix Multiplication

This project implements the Strassen matrix multiplication algorithm using MPI (Message Passing Interface) with a divide-and-conquer approach for distributed parallel computing.

## Overview

The implementation follows the pseudo code structure provided, featuring:

- **Divide and Conquer**: Recursively divides matrices into quadrants
- **MPI Distribution**: Distributes the 7 Strassen products (P1-P7) among available processes
- **Dynamic Process Tree**: Creates a process tree with configurable height
- **Fallback to Sequential**: Uses sequential Strassen or standard multiplication when appropriate

## Algorithm Structure

```
strassenMultiplyMPI(A, B, n, rank, num_procs, level)
├── Base Case (n=1): Direct multiplication
├── Matrix Division: Split A and B into 4 quadrants each
├── Process Distribution Decision:
│   ├── If (level < height && num_procs > 1): Distribute P1-P7 to child processes
│   └── Else: Compute P1-P7 locally
└── Result Combination: Combine P1-P7 → C11,C12,C21,C22 → C
```

## Files

- `strassen_mpi.h/c`: Core MPI Strassen implementation
- `matrix_utils.h/c`: Matrix operations (init, split, combine, add, subtract)
- `main.c`: Test program with performance measurement
- `Makefile`: Build configuration

## Requirements

- MPI implementation (OpenMPI or MPICH)
- C compiler (gcc)
- Linux/Unix system

### Installation (Ubuntu/Debian)

```bash
# Install MPI and build tools
sudo apt-get update
sudo apt-get install -y build-essential libopenmpi-dev openmpi-bin

# Or use the provided target
make install-deps
```

## Building

```bash
# Build the project
make

# Build with debug symbols
make debug

# Check MPI installation
make check-mpi
```

## Usage

### Basic Usage

```bash
# Run with default settings (4x4 matrix, 2 processes)
make run

# Run with custom parameters
mpirun -np <num_processes> ./strassen_mpi <matrix_size>
```

### Examples

```bash
# 8x8 matrix with 4 processes
mpirun -np 4 ./strassen_mpi 8

# 16x16 matrix with 8 processes  
mpirun -np 8 ./strassen_mpi 16

# Using Makefile shortcuts
make run-custom NP=4 SIZE=8
make test
make performance
```

### Parameters

- `matrix_size`: Must be a power of 2 (2, 4, 8, 16, 32, ...)
- `num_processes`: Number of MPI processes to use

## Configuration

Key constants in `strassen_mpi.h`:

```c
#define MAX_TREE_HEIGHT 3        // Maximum process tree depth
#define MIN_SIZE_THRESHOLD 32    // Minimum size for parallel processing
```

## Algorithm Details

### Strassen's Formulas

The algorithm computes 7 products instead of 8:
- P1 = (A11 + A22) × (B11 + B22)
- P2 = (A21 + A22) × B11
- P3 = A11 × (B12 - B22)
- P4 = A22 × (B21 - B11)
- P5 = (A11 + A12) × B22
- P6 = (A21 - A11) × (B11 + B12)
- P7 = (A12 - A22) × (B21 + B22)

### Result Combination
- C11 = P1 + P4 - P5 + P7
- C12 = P3 + P5
- C21 = P2 + P4
- C22 = P1 - P2 + P3 + P6

### MPI Process Distribution

1. **Master Process (rank 0)**:
   - Initializes matrices
   - Distributes work to available processes
   - Combines final results
   - Performs verification

2. **Worker Processes**:
   - Receive matrix data and parameters
   - Perform recursive Strassen computation
   - Send results back to parent process

3. **Process Tree Structure**:
   ```
   Process 0 (Master)
   ├── Process 1 (P1)
   ├── Process 2 (P2)
   ├── Process 3 (P3)
   ├── Process 4 (P4)
   ├── Process 5 (P5)
   ├── Process 6 (P6)
   └── Process 7 (P7)
   ```

## Performance

### Complexity
- **Sequential Strassen**: O(n^2.807)
- **Standard Multiplication**: O(n^3)
- **Parallel Efficiency**: Depends on matrix size and process count

### Optimization Features
- Automatic fallback to standard multiplication for small matrices
- Configurable parallelization threshold
- Memory-efficient matrix operations
- Process tree height limiting

## Testing and Verification

The program includes automatic verification:

```bash
# Run comprehensive tests
make test

# Performance benchmarking
make performance
```

### Verification Process
1. Computes result using MPI Strassen
2. Computes same multiplication using standard algorithm
3. Compares results element-by-element
4. Reports timing and speedup information

## Troubleshooting

### Common Issues

1. **Matrix size not power of 2**:
   ```
   Error: Matrix size must be a power of 2 and >= 2
   ```
   Solution: Use sizes like 2, 4, 8, 16, 32, 64, ...

2. **MPI not found**:
   ```
   mpicc: command not found
   ```
   Solution: Install MPI using `make install-deps`

3. **Process communication errors**:
   - Ensure sufficient processes for the requested configuration
   - Check MPI installation and network configuration

### Debug Mode

```bash
# Build and run with debug information
make debug
mpirun -np 2 ./strassen_mpi 4
```

## Performance Tuning

1. **Adjust thresholds**:
   - Increase `MIN_SIZE_THRESHOLD` for better cache locality
   - Decrease `MAX_TREE_HEIGHT` to reduce communication overhead

2. **Process mapping**:
   - Use `mpirun --map-by core` for better CPU utilization
   - Consider NUMA topology for multi-socket systems

3. **Matrix sizes**:
   - Larger matrices benefit more from parallelization
   - Test different sizes to find optimal performance points

## Example Output

```
=== MPI Strassen Matrix Multiplication ===
Matrix size: 8x8
Number of processes: 4
Tree height limit: 3
Sequential threshold: 32
==========================================

Starting MPI Strassen multiplication...
MPI Strassen multiplication completed!
CPU Time: 0.002341 seconds
Wall Time: 0.001876 seconds

Verifying result with standard multiplication...
Standard multiplication time: 0.001234 seconds
✓ Verification PASSED - Results match!
Speedup: 0.66x

=== Performance Summary ===
Matrix operations: ~343
MFLOPs: 548.72
===========================
```