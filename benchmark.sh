#!/bin/bash

# Compile all programs
echo "Compiling programs..."
gcc -o practica1 practica1.c
gcc -fopenmp -o practica2 practica2.c
gcc -o practica3 practica3.c
nvcc -o practica4 practica4.cu
gcc -o generar_matriz generar_matriz.c

# Create results directory
mkdir -p results

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
    
    # Run OpenMP version with different thread counts
    echo "Running OpenMP version..."
    export OMP_NUM_THREADS=$threads
    time_openmp=$(time ./practica2 "$matriz_file" "$vector_file" 2>&1 | grep "real" | awk '{print $2}')
    
    # Run CUDA version
    echo "Running CUDA version..."
    time_cuda=$(time ./practica4 "$matriz_file" "$vector_file" 2>&1 | grep "real" | awk '{print $2}')
    
    # Calculate speedup
    speedup_openmp=$(echo "scale=2; $time_sequential / $time_openmp" | bc)
    speedup_cuda=$(echo "scale=2; $time_sequential / $time_cuda" | bc)
    
    # Save results
    echo "$size,$threads,$time_sequential,$time_openmp,$time_cuda,$speedup_openmp,$speedup_cuda" >> results/benchmark_results.csv
    
    # Clean up generated files
    rm "$matriz_file" "$vector_file"
}

# Create CSV header
echo "Size,Threads,Sequential Time,OpenMP Time,CUDA Time,OpenMP Speedup,CUDA Speedup" > results/benchmark_results.csv

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
    echo "Threads | Sequential | OpenMP | CUDA | OpenMP Speedup | CUDA Speedup" >> results/summary.txt
    echo "--------|------------|--------|------|----------------|-------------" >> results/summary.txt
    
    for thread in "${threads[@]}"; do
        line=$(grep "^$size,$thread," results/benchmark_results.csv)
        IFS=',' read -r _ _ seq_time omp_time cuda_time omp_speedup cuda_speedup <<< "$line"
        printf "%7d | %10.3f | %6.3f | %4.3f | %14.2f | %12.2f\n" \
               "$thread" "$seq_time" "$omp_time" "$cuda_time" "$omp_speedup" "$cuda_speedup" >> results/summary.txt
    done
    echo "" >> results/summary.txt
done

echo "Benchmark completed. Results saved in results/ directory." 