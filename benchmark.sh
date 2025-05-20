#!/bin/bash

# Enable error reporting
set -e

# Function to check if a command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 no está instalado. Por favor, instala $2"
        exit 1
    fi
}

# Function to calculate speedup safely
calculate_speedup() {
    local seq_time=$1
    local par_time=$2
    
    # Skip if times are invalid
    if [[ -z "$seq_time" || -z "$par_time" || "$seq_time" == "0" || "$par_time" == "0" ]]; then
        echo "N/A"
        return
    fi
    
    # Convert to decimal point format for bc
    seq_time=$(echo $seq_time | tr ',' '.')
    par_time=$(echo $par_time | tr ',' '.')
    
    # Use bc for calculation with proper handling
    echo "scale=4; if($par_time>0) $seq_time / $par_time else 0" | bc -l
}

# Function to run benchmark for a specific size
run_benchmark() {
    local size=$1
    local threads=$2
    
    echo "====================================================="
    echo "Running benchmark for size $size with $threads threads/processes"
    echo "====================================================="
    
    # Create directory for generated files
    mkdir -p temp_files
    
    # Generate matrix and vector
    echo "Generating matrix and vector of size $size..."
    ./generar_matriz $size temp_files/
    
    matriz_file="temp_files/matriz_${size}x${size}.txt"
    vector_file="temp_files/vector_${size}.txt"
    
    # Verify files were generated correctly
    if [ ! -f "$matriz_file" ] || [ ! -f "$vector_file" ]; then
        echo "Error: Could not generate files for size $size" >&2
        return 1
    fi
    
    # Run sequential version (only once per size)
    if [ $threads -eq 1 ]; then
        echo "Running sequential version (no threads)..."
        output_sequential=$(./practica1 "$matriz_file" "$vector_file" 2>&1)
        time_sequential=$(echo "$output_sequential" | grep "Tiempo de ejecución:" | awk '{print $4}')
        if [ -z "$time_sequential" ]; then
            echo "Error: Could not get sequential execution time" >&2
            echo "Complete output:" >&2
            echo "$output_sequential" >&2
            return 1
        fi
        echo "Sequential time: $time_sequential seconds"
        
        # Save sequential time for this size
        echo "$size,$time_sequential" > temp_files/seq_time_$size.txt
    else
        # For other thread counts, use the stored sequential time
        if [ -f "temp_files/seq_time_$size.txt" ]; then
            time_sequential=$(cat temp_files/seq_time_$size.txt | cut -d',' -f2)
            echo "Using stored sequential time: $time_sequential seconds"
        else
            echo "Error: Sequential time not found for size $size" >&2
            echo "Run with threads=1 first" >&2
            return 1
        fi
    fi
    
    # Run OpenMP version with specified thread count
    echo "Running OpenMP version ($threads CPU threads)..."
    export OMP_NUM_THREADS=$threads
    output_openmp=$(./practica2 "$matriz_file" "$vector_file" 2>&1)
    time_openmp=$(echo "$output_openmp" | grep "Tiempo de ejecución:" | awk '{print $4}')
    if [ -z "$time_openmp" ]; then
        echo "Error: Could not get OpenMP execution time" >&2
        echo "Complete output:" >&2
        echo "$output_openmp" >&2
        time_openmp="N/A"
    else
        echo "OpenMP time: $time_openmp seconds"
    fi
    
    # Run MPI version with specified process count
    echo "Running MPI version ($threads processes)..."
    # Limit MPI processes to a reasonable number
    local mpi_processes=$threads
    if [ $mpi_processes -gt 8 ]; then
        echo "Warning: Limiting MPI processes to 8 (requested: $threads)"
        mpi_processes=8
    fi
    
    output_mpi=$(mpirun -np $mpi_processes ./practica3_mpi "$matriz_file" "$vector_file" 2>&1)
    time_mpi=$(echo "$output_mpi" | grep "Tiempo de ejecución MPI:" | awk '{print $4}')
    if [ -z "$time_mpi" ]; then
        echo "Error: Could not get MPI execution time" >&2
        echo "Complete output:" >&2
        echo "$output_mpi" >&2
        time_mpi="N/A"
    else
        echo "MPI time: $time_mpi seconds"
    fi
    
    # Run CUDA version
    echo "Running CUDA version (GPU)..."
    output_cuda=$(./practica4 "$matriz_file" "$vector_file" 2>&1)
    time_cuda=$(echo "$output_cuda" | grep "Tiempo de ejecución CUDA:" | awk '{print $4}')
    if [ -z "$time_cuda" ]; then
        echo "Error: Could not get CUDA execution time" >&2
        echo "Complete output:" >&2
        echo "$output_cuda" >&2
        time_cuda="N/A"
    else
        echo "CUDA time: $time_cuda seconds"
    fi
    
    # Calculate speedups safely
    speedup_openmp=$(calculate_speedup "$time_sequential" "$time_openmp")
    speedup_mpi=$(calculate_speedup "$time_sequential" "$time_mpi")
    speedup_cuda=$(calculate_speedup "$time_sequential" "$time_cuda")
    
    # Print results for this run
    echo "------- RESULTS -------"
    echo "Sequential: $time_sequential seconds"
    echo "OpenMP ($threads threads): $time_openmp seconds (Speedup: $speedup_openmp)"
    echo "MPI ($mpi_processes processes): $time_mpi seconds (Speedup: $speedup_mpi)"
    echo "CUDA: $time_cuda seconds (Speedup: $speedup_cuda)"
    
    # Save results to CSV
    echo "$size,$threads,$time_sequential,$time_openmp,$time_mpi,$time_cuda,$speedup_openmp,$speedup_mpi,$speedup_cuda" >> results/benchmark_results.csv
}

# Main function
main() {
    # Check required dependencies
    echo "Checking dependencies..."
    check_command "gcc" "GCC (gcc)"
    check_command "nvcc" "CUDA Toolkit (nvcc)"
    check_command "mpicc" "MPI (openmpi-bin y libopenmpi-dev)"
    check_command "bc" "bc (bc)"
    
    # Compile all programs with optimization flags
    echo "Compiling programs..."
    
    # Compile sequential version with optimization
    echo "Compiling sequential version..."
    gcc -O3 -o practica1 practica1.c
    
    # Compile OpenMP version with optimization
    echo "Compiling OpenMP version..."
    gcc -O3 -fopenmp -o practica2 practica2.c
    
    # Compile MPI version with optimization
    echo "Compiling MPI version..."
    mpicc -O3 -o practica3_mpi practica3.c
    
    # Compile CUDA version with proper flags and optimization
    echo "Compiling CUDA version..."
    nvcc -O3 -o practica4 practica4.cu -arch=sm_60 -Wno-deprecated-gpu-targets
    
    # Compile matrix generator with optimization
    echo "Compiling matrix generator..."
    gcc -O3 -o generar_matriz generar_matriz.c
    
    # Create results directory
    mkdir -p results
    
    # Create CSV header
    echo "Size,Threads,Sequential Time,OpenMP Time,MPI Time,CUDA Time,OpenMP Speedup,MPI Speedup,CUDA Speedup" > results/benchmark_results.csv
    
    # Run benchmarks for different sizes and thread configurations
    sizes=(500 1000 2000 5000 10000)
    threads=(1 2 4 8 16)
    
    echo "Starting benchmarks..."
    for size in "${sizes[@]}"; do
        for thread in "${threads[@]}"; do
            run_benchmark $size $thread
        done
    done
    
    # Generate summary table
    echo "Generating summary table..."
    echo "Results Summary" > results/summary.txt
    echo "==============" >> results/summary.txt
    echo "" >> results/summary.txt
    
    for size in "${sizes[@]}"; do
        echo "Matrix Size: $size x $size" >> results/summary.txt
        echo "----------------------------------------" >> results/summary.txt
        echo "Threads | Sequential | OpenMP | MPI | CUDA | OpenMP Speedup | MPI Speedup | CUDA Speedup" >> results/summary.txt
        echo "--------|------------|--------|-----|------|----------------|-------------|-------------" >> results/summary.txt
        
        for thread in "${threads[@]}"; do
            line=$(grep "^$size,$thread," results/benchmark_results.csv || echo "No data")
            if [ "$line" != "No data" ]; then
                IFS=',' read -r _ _ seq_time omp_time mpi_time cuda_time omp_speedup mpi_speedup cuda_speedup <<< "$line"
                printf "%7d | %10s | %6s | %3s | %4s | %14s | %11s | %12s\n" \
                       "$thread" "$seq_time" "$omp_time" "$mpi_time" "$cuda_time" "$omp_speedup" "$mpi_speedup" "$cuda_speedup" >> results/summary.txt
            else
                printf "%7d | %10s | %6s | %3s | %4s | %14s | %11s | %12s\n" \
                       "$thread" "N/A" "N/A" "N/A" "N/A" "N/A" "N/A" "N/A" >> results/summary.txt
            fi
        done
        echo "" >> results/summary.txt
    done
    
    # Clean up
    rm -rf temp_files
    
    echo "Benchmark completed. Results saved in results/ directory."
}

# Run main function
main 