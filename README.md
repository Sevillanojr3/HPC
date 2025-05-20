# HPC - High Performance Computing Practices

This repository contains several practice exercises for High Performance Computing using C and CUDA.

## Prerequisites

### Required Dependencies

Install all required dependencies using the following commands:

```bash
# Install GCC compiler
sudo apt-get install gcc

# Install OpenMP
sudo apt-get install libomp-dev

# Install MPI
sudo apt-get install openmpi-bin libopenmpi-dev

# Install CUDA Toolkit
sudo apt-get install nvidia-cuda-toolkit

# Install bc (for calculations in benchmark script)
sudo apt-get install bc
```

### System Requirements

- Linux operating system
- NVIDIA GPU with CUDA support (for CUDA programs)
- At least 4GB of RAM recommended
- Sufficient disk space for matrix generation

### Verification

After installation, verify the tools are available:

```bash
# Check GCC version
gcc --version

# Check OpenMP
gcc -fopenmp -v

# Check MPI
mpicc --version

# Check CUDA
nvcc --version

# Check bc
bc --version
```

## Project Structure

- `practica1.c` - First practice exercise (Sequential implementation)
- `practica2.c` - Second practice exercise (OpenMP parallel implementation)
- `practica3.c` - Third practice exercise (MPI implementation)
- `practica4.cu` - Fourth practice exercise (CUDA implementation)
- `generar_matriz.c` - Matrix generation utility
- `benchmark.sh` - Script to measure performance and speedup

## Compilation Instructions

### C Programs

To compile the C programs, use the following commands:

```bash
# Compile practica1
gcc -o practica1 practica1.c

# Compile practica2
gcc -fopenmp -o practica2 practica2.c

# Compile practica3 (MPI version)
mpicc -o practica3_mpi practica3.c

# Compile matrix generator
gcc -o generar_matriz generar_matriz.c
```

### CUDA Program

To compile the CUDA program:

```bash
# Compile practica4
nvcc -o practica4 practica4.cu
```

## Running the Programs

1. First, generate the input matrix using the matrix generator:
```bash
./generar_matriz
```

2. Then run any of the practice programs:
```bash
# Sequential version
./practica1

# OpenMP version
./practica2

# MPI version
mpirun -np <number_of_processes> ./practica3_mpi

# CUDA version
./practica4
```

## Performance Benchmarking

The repository includes a benchmark script (`benchmark.sh`) that measures the performance and speedup of different implementations across various matrix sizes and thread configurations.

### Running the Benchmark

1. Make the script executable:
```bash
chmod +x benchmark.sh
```

2. Run the benchmark:
```bash
./benchmark.sh
```

The script will:
- Test matrix sizes: 500x500, 5000x5000, 50000x50000, and 100000x100000
- Test thread configurations: 1, 2, 4, 8, and 16 threads
- Compare sequential, OpenMP, MPI, and CUDA implementations
- Generate speedup tables and graphs

Results will be saved in the `results/` directory:
- `benchmark_results.csv`: Raw timing data
- `summary.txt`: Formatted summary table with speedup calculations

### Interpreting Results

The benchmark results show:
- Execution time for each implementation
- Speedup compared to sequential implementation
- Performance scaling with different thread counts
- Comparison between OpenMP, MPI, and CUDA implementations

## Notes

- Make sure you have the necessary permissions to execute the programs (`chmod +x` if needed)
- The CUDA program requires a NVIDIA GPU and proper CUDA drivers installed
- Some programs might require specific input parameters - check the source code for details
- The benchmark script requires the `bc` command-line calculator to be installed

## Troubleshooting

If you encounter any issues:
1. Ensure all dependencies are installed
2. Check that you have the correct permissions
3. Verify that your CUDA installation is working properly (for CUDA programs)
4. Make sure you're running the programs in the correct order (matrix generation first)
5. For benchmark issues, check that `bc` is installed and the script has execute permissions