# HPC - Prácticas de Computación de Alto Rendimiento

Este repositorio contiene implementaciones de multiplicación de matrices dispersas por vectores utilizando diferentes paradigmas de programación paralela en C y CUDA.

## Prerrequisitos

### Dependencias Requeridas

Instala todas las dependencias necesarias usando los siguientes comandos:

```bash
# Instalar compilador GCC
sudo apt-get install gcc

# Instalar OpenMP
sudo apt-get install libomp-dev

# Instalar MPI
sudo apt-get install openmpi-bin libopenmpi-dev

# Instalar CUDA Toolkit
sudo apt-get install nvidia-cuda-toolkit

# Instalar bc (para cálculos en el script de benchmark)
sudo apt-get install bc
```

### Requisitos del Sistema

- Sistema operativo Linux
- GPU NVIDIA con soporte CUDA (para programas CUDA)
- Al menos 4GB de RAM recomendado
- Espacio en disco suficiente para la generación de matrices

### Verificación

Después de la instalación, verifica que las herramientas estén disponibles:

```bash
# Comprobar versión de GCC
gcc --version

# Comprobar OpenMP
gcc -fopenmp -v

# Comprobar MPI
mpicc --version

# Comprobar CUDA
nvcc --version

# Comprobar bc
bc --version
```

## Estructura del Proyecto

- `practica1.c` - Implementación secuencial de multiplicación matriz-vector
- `practica2.c` - Implementación paralela con OpenMP
- `practica3.c` - Implementación distribuida con MPI
- `practica4.cu` - Implementación en GPU con CUDA
- `generar_matriz.c` - Utilidad para generar matrices y vectores
- `benchmark.sh` - Script para medir rendimiento y aceleración

## Instrucciones de Compilación

### Programas en C

Para compilar los programas en C, utiliza los siguientes comandos:

```bash
# Compilar practica1 (versión secuencial)
gcc -O3 -o practica1 practica1.c

# Compilar practica2 (versión OpenMP)
gcc -O3 -fopenmp -o practica2 practica2.c

# Compilar practica3 (versión MPI)
mpicc -O3 -o practica3_mpi practica3.c

# Compilar generador de matrices
gcc -O3 -o generar_matriz generar_matriz.c
```

### Programa CUDA

Para compilar el programa CUDA:

```bash
# Compilar practica4 (versión CUDA)
nvcc -O3 -o practica4 practica4.cu -arch=sm_60 -Wno-deprecated-gpu-targets
```

## Ejecución de los Programas

1. Primero, genera la matriz y el vector de entrada:
```bash
# Generar matriz y vector de tamaño 1000
./generar_matriz 1000
```

2. Luego ejecuta cualquiera de los programas:
```bash
# Versión secuencial
./practica1 matriz_1000x1000.txt vector_1000.txt

# Versión OpenMP (especificar número de hilos)
export OMP_NUM_THREADS=4
./practica2 matriz_1000x1000.txt vector_1000.txt

# Versión MPI (especificar número de procesos)
mpirun -np 4 ./practica3_mpi matriz_1000x1000.txt vector_1000.txt

# Versión CUDA
./practica4 matriz_1000x1000.txt vector_1000.txt
```

## Benchmarking de Rendimiento

El repositorio incluye un script de benchmark (`benchmark.sh`) que mide el rendimiento y la aceleración de las diferentes implementaciones con varios tamaños de matriz y configuraciones de hilos.

### Ejecución del Benchmark

1. Haz el script ejecutable:
```bash
chmod +x benchmark.sh
```

2. Ejecuta el benchmark:
```bash
./benchmark.sh
```

El script:
- Prueba matrices de tamaño: 500x500, 1000x1000, 2000x2000, 5000x5000, 10000x10000
- Prueba configuraciones de hilos/procesos: 1, 2, 4, 8 y 16
- Compara implementaciones secuencial, OpenMP, MPI y CUDA
- Genera tablas de aceleración

Los resultados se guardan en el directorio `results/`:
- `benchmark_results.csv`: Datos de tiempo brutos
- `summary.txt`: Tabla resumen formateada con cálculos de aceleración

### Interpretación de Resultados

Los resultados del benchmark muestran:
- Tiempo de ejecución para cada implementación
- Aceleración comparada con la implementación secuencial
- Escalado de rendimiento con diferentes cantidades de hilos
- Comparación entre implementaciones OpenMP, MPI y CUDA

## Optimizaciones Implementadas

### Versión Secuencial
- Estructura de matriz dispersa eficiente
- Inicialización de resultado con memset
- Multiplicación directa de elementos no nulos

### Versión OpenMP
- Uso de arrays locales para cada hilo
- Schedule dynamic para mejor balanceo de carga
- Reducción manual para evitar false sharing

### Versión MPI
- Distribución equitativa de elementos entre procesos
- Gestión de casos con matrices vacías
- Reducción eficiente usando MPI_Reduce

### Versión CUDA
- Paralelización a nivel de elemento no nulo
- Uso eficiente de memoria GPU con atomicAdd
- Manejo de errores CUDA mejorado

## Notas

- Asegúrate de tener los permisos necesarios para ejecutar los programas (`chmod +x` si es necesario)
- El programa CUDA requiere una GPU NVIDIA y drivers CUDA instalados
- Algunos programas podrían requerir parámetros de entrada específicos
- El script de benchmark requiere que la calculadora `bc` esté instalada

## Solución de Problemas

Si encuentras algún problema:
1. Asegúrate de que todas las dependencias estén instaladas
2. Verifica que tengas los permisos correctos
3. Comprueba que tu instalación de CUDA funcione correctamente (para programas CUDA)
4. Asegúrate de estar ejecutando los programas en el orden correcto (primero la generación de matrices)
5. Para problemas con el benchmark, verifica que `bc` esté instalado y que el script tenga permisos de ejecución
6. Si hay errores de compilación en CUDA, intenta modificar la arquitectura en el comando de compilación (-arch=sm_XX)
7. Para matrices muy grandes, asegúrate de tener suficiente memoria RAM y espacio en disco