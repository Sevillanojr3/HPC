#!/bin/bash

# Compilar todos los programas
echo "Compilando programas..."

# Compilar versión secuencial
echo "Compilando versión secuencial..."
gcc -O3 -o practica1 practica1.c && echo "  ✓ Secuencial OK" || echo "  ✗ Error en secuencial"

# Compilar versión OpenMP
echo "Compilando versión OpenMP..."
gcc -O3 -fopenmp -o practica2 practica2.c && echo "  ✓ OpenMP OK" || echo "  ✗ Error en OpenMP"

# Compilar versión MPI
echo "Compilando versión MPI..."
mpicc -O3 -o practica3_mpi practica3.c && echo "  ✓ MPI OK" || echo "  ✗ Error en MPI"

# Compilar versión CUDA (si nvcc está disponible)
echo "Compilando versión CUDA..."
if command -v nvcc &> /dev/null; then
    nvcc -O3 -o practica4 practica4.cu -arch=sm_60 -Wno-deprecated-gpu-targets && echo "  ✓ CUDA OK" || echo "  ✗ Error en CUDA"
else
    echo "  ⚠ CUDA no disponible (nvcc no encontrado)"
fi

# Compilar generador de matrices
echo "Compilando generador de matrices..."
gcc -O3 -o generar_matriz generar_matriz.c && echo "  ✓ Generador OK" || echo "  ✗ Error en generador"

echo "Creando directorio para resultados..."
mkdir -p results

echo "Comprobación completada." 