#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a header file exists
header_exists() {
    echo "#include <$1>" | gcc -c -x c - >/dev/null 2>&1
}

# Check available implementations
HAS_MPI=false
HAS_CUDA=false

# Check MPI
if command_exists mpicc && header_exists mpi.h; then
    HAS_MPI=true
    echo "MPI is available"
else
    echo "MPI is not available - skipping MPI implementation"
fi

# Check CUDA
if command_exists nvcc && header_exists cuda_runtime.h; then
    HAS_CUDA=true
    echo "CUDA is available"
else
    echo "CUDA is not available - skipping CUDA implementation"
fi

# Compile available programs
echo "Compiling programs..."
gcc -o practica1 practica1.c
gcc -fopenmp -o practica2 practica2.c

if [ "$HAS_MPI" = true ]; then
    mpicc -o practica3_mpi practica3.c
fi

if [ "$HAS_CUDA" = true ]; then
    nvcc -o practica4 practica4.cu
fi

gcc -o generar_matriz generar_matriz.c

# Create results directory
mkdir -p results

# Function to convert time format to seconds
convert_time_to_seconds() {
    local time_str=$1
    local minutes=0
    local seconds=0
    
    # Check if time contains minutes
    if [[ $time_str == *m* ]]; then
        minutes=$(echo $time_str | cut -d'm' -f1)
        seconds=$(echo $time_str | cut -d'm' -f2 | cut -d's' -f1)
    else
        seconds=$(echo $time_str | cut -d's' -f1)
    fi
    
    # Convert to seconds and ensure it's a valid number
    echo "scale=3; $minutes * 60 + $seconds" | bc
}

# Function to calculate speedup safely
calculate_speedup() {
    local seq_time=$1
    local par_time=$2
    
    # Check if times are valid numbers
    if [[ ! $seq_time =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ ! $par_time =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "0.00"
        return
    fi
    
    # Check for division by zero
    if (( $(echo "$par_time == 0" | bc -l) )); then
        echo "0.00"
        return
    fi
    
    echo "scale=2; $seq_time / $par_time" | bc
}

# Function to run benchmark for a specific size
run_benchmark() {
    local size=$1
    local threads=$2
    
    echo "Running benchmark for size $size with $threads threads..."
    
    # Generate matrix and vector
    echo "Generating matrix and vector of size $size..."
    ./generar_matriz $size
    
    matriz_file="matriz_${size}x${size}.txt"
    vector_file="vector_${size}.txt"
    
    # Run sequential version
    echo "Running sequential version..."
    time_sequential=$(time ./practica1 "$matriz_file" "$vector_file" 2>&1 | grep "real" | awk '{print $2}')
    time_sequential=$(convert_time_to_seconds "$time_sequential")
    
    # Run OpenMP version with different thread counts
    echo "Running OpenMP version..."
    export OMP_NUM_THREADS=$threads
    time_openmp=$(time ./practica2 "$matriz_file" "$vector_file" 2>&1 | grep "real" | awk '{print $2}')
    time_openmp=$(convert_time_to_seconds "$time_openmp")
    
    # Initialize variables
    time_mpi="0.00"
    time_cuda="0.00"
    speedup_mpi="0.00"
    speedup_cuda="0.00"
    
    # Run MPI version if available
    if [ "$HAS_MPI" = true ]; then
        echo "Running MPI version..."
        time_mpi=$(time mpirun -np $threads ./practica3_mpi "$matriz_file" "$vector_file" 2>&1 | grep "real" | awk '{print $2}')
        time_mpi=$(convert_time_to_seconds "$time_mpi")
        speedup_mpi=$(calculate_speedup "$time_sequential" "$time_mpi")
    fi
    
    # Run CUDA version if available
    if [ "$HAS_CUDA" = true ]; then
        echo "Running CUDA version..."
        time_cuda=$(time ./practica4 "$matriz_file" "$vector_file" 2>&1 | grep "real" | awk '{print $2}')
        time_cuda=$(convert_time_to_seconds "$time_cuda")
        speedup_cuda=$(calculate_speedup "$time_sequential" "$time_cuda")
    fi
    
    # Calculate OpenMP speedup
    speedup_openmp=$(calculate_speedup "$time_sequential" "$time_openmp")
    
    # Save results
    echo "$size,$threads,$time_sequential,$time_openmp,$time_mpi,$time_cuda,$speedup_openmp,$speedup_mpi,$speedup_cuda" >> results/benchmark_results.csv
    
    # Clean up generated files
    rm "$matriz_file" "$vector_file"
}

# Create CSV header
echo "Size,Threads,Sequential Time,OpenMP Time,MPI Time,CUDA Time,OpenMP Speedup,MPI Speedup,CUDA Speedup" > results/benchmark_results.csv

# Run benchmarks for different sizes and thread configurations
sizes=(500 5000 50000 100000)
threads=(1 2 4 8 16)

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
        line=$(grep "^$size,$thread," results/benchmark_results.csv)
        IFS=',' read -r _ _ seq_time omp_time mpi_time cuda_time omp_speedup mpi_speedup cuda_speedup <<< "$line"
        printf "%7d | %10.3f | %6.3f | %3.3f | %4.3f | %14.2f | %11.2f | %12.2f\n" \
               "$thread" "$seq_time" "$omp_time" "$mpi_time" "$cuda_time" "$omp_speedup" "$mpi_speedup" "$cuda_speedup" >> results/summary.txt
    done
    echo "" >> results/summary.txt
done

echo "Benchmark completed. Results saved in results/ directory." 