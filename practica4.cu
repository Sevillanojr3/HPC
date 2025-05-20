#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <sys/time.h>  // Para gettimeofday

// Estructura para almacenar elementos no nulos de la matriz dispersa
typedef struct {
    int fila;
    int columna;
    double valor;
} Elemento;

// Estructura para la matriz dispersa
typedef struct {
    Elemento* elementos;
    int num_elementos;
    int filas;
    int columnas;
} MatrizDispersa;

// Función para verificar errores CUDA
#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

// Kernel CUDA para multiplicar matriz dispersa por vector
__global__ void multiplicar_matriz_vector_cuda(Elemento* elementos, int num_elementos, double* vector, double* resultado, int filas) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_elementos) {
        Elemento e = elementos[idx];
        atomicAdd(&resultado[e.fila], e.valor * vector[e.columna]);
    }
}

// Función para leer la matriz dispersa
MatrizDispersa* leer_matriz_dispersa(FILE *archivo, int *filas, int *columnas) {
    if (fscanf(archivo, "%d %d", filas, columnas) != 2) {
        printf("Error al leer dimensiones\n");
        return NULL;
    }

    // Reservamos memoria para la matriz
    MatrizDispersa* matriz = (MatrizDispersa*)malloc(sizeof(MatrizDispersa));
    if (!matriz) {
        printf("Error al reservar memoria para la matriz\n");
        return NULL;
    }

    // Estimamos el número de elementos no nulos (10% de la matriz)
    int num_elementos = (*filas * *columnas) / 10;
    matriz->elementos = (Elemento*)malloc(num_elementos * sizeof(Elemento));
    if (!matriz->elementos) {
        printf("Error al reservar memoria para los elementos\n");
        free(matriz);
        return NULL;
    }

    // Leemos y guardamos solo elementos no nulos
    int idx = 0;
    double valor;
    for (int i = 0; i < *filas; i++) {
        for (int j = 0; j < *columnas; j++) {
            if (fscanf(archivo, "%lf", &valor) != 1) {
                printf("Error al leer elemento [%d,%d]\n", i, j);
                free(matriz->elementos);
                free(matriz);
                return NULL;
            }
            if (valor != 0) {
                if (idx >= num_elementos) {
                    // Redimensionar si es necesario
                    num_elementos *= 2;
                    void* temp_ptr = realloc(matriz->elementos, num_elementos * sizeof(Elemento));
                    if (!temp_ptr) {
                        printf("Error al redimensionar memoria\n");
                        free(matriz->elementos);
                        free(matriz);
                        return NULL;
                    }
                    matriz->elementos = (Elemento*)temp_ptr;
                }
                matriz->elementos[idx].fila = i;
                matriz->elementos[idx].columna = j;
                matriz->elementos[idx].valor = valor;
                idx++;
            }
        }
    }

    // Ajustar al tamaño real
    if (idx < num_elementos) {
        void* temp_ptr = realloc(matriz->elementos, idx * sizeof(Elemento));
        if (temp_ptr) {
            matriz->elementos = (Elemento*)temp_ptr;
        }
    }

    matriz->num_elementos = idx;
    matriz->filas = *filas;
    matriz->columnas = *columnas;
    return matriz;
}

// Función para leer el vector
double* leer_vector(FILE *archivo, int *dimension) {
    if (fscanf(archivo, "%d", dimension) != 1) {
        printf("Error al leer dimensión del vector\n");
        return NULL;
    }

    double *vector = (double *)malloc(*dimension * sizeof(double));
    if (!vector) {
        printf("Error al reservar memoria para el vector\n");
        return NULL;
    }

    // Leer todo el vector de una vez
    for (int i = 0; i < *dimension; i++) {
        if (fscanf(archivo, "%lf", &vector[i]) != 1) {
            printf("Error al leer elemento %d del vector\n", i);
            free(vector);
            return NULL;
        }
    }

    return vector;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Uso: %s <archivo_matriz> <archivo_vector>\n", argv[0]);
        return 1;
    }
    
    // Cargar datos
    int dim_vector;
    MatrizDispersa* matriz = leer_matriz_dispersa(fopen(argv[1], "r"), &matriz->filas, &matriz->columnas);
    double* vector = leer_vector(fopen(argv[2], "r"), &dim_vector);
    
    if (!matriz || !vector) {
        if (matriz) {
            free(matriz->elementos);
            free(matriz);
        }
        if (vector) free(vector);
        return 1;
    }
    
    // Verificar dimensiones
    if (matriz->columnas != dim_vector) {
        printf("Error: Las dimensiones no son compatibles\n");
        free(matriz->elementos);
        free(matriz);
        free(vector);
        return 1;
    }
    
    // Reservar memoria en GPU
    Elemento* d_elementos;
    double* d_vector;
    double* d_resultado;
    
    cudaMalloc(&d_elementos, matriz->num_elementos * sizeof(Elemento));
    cudaMalloc(&d_vector, dim_vector * sizeof(double));
    cudaMalloc(&d_resultado, matriz->filas * sizeof(double));
    
    // Copiar datos a GPU
    cudaMemcpy(d_elementos, matriz->elementos, matriz->num_elementos * sizeof(Elemento), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, vector, dim_vector * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_resultado, 0, matriz->filas * sizeof(double));
    
    // Configurar grid y blocks
    int block_size = 256;
    int num_blocks = (matriz->num_elementos + block_size - 1) / block_size;
    
    // Medir tiempo
    struct timeval inicio, fin;
    gettimeofday(&inicio, NULL);
    
    // Ejecutar kernel
    multiplicar_matriz_vector_cuda<<<num_blocks, block_size>>>(d_elementos, matriz->num_elementos, d_vector, d_resultado, matriz->filas);
    
    // Sincronizar y verificar errores
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Error CUDA: %s\n", cudaGetErrorString(error));
        cudaFree(d_elementos);
        cudaFree(d_vector);
        cudaFree(d_resultado);
        free(matriz->elementos);
        free(matriz);
        free(vector);
        return 1;
    }
    
    gettimeofday(&fin, NULL);
    double tiempo = (fin.tv_sec - inicio.tv_sec) + (fin.tv_usec - inicio.tv_usec) / 1000000.0;
    
    // Copiar resultado de vuelta a CPU
    double* resultado = (double*)malloc(matriz->filas * sizeof(double));
    cudaMemcpy(resultado, d_resultado, matriz->filas * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Imprimir tiempo
    printf("Tiempo de ejecución CUDA: %.6f segundos\n", tiempo);
    
    // Liberar memoria
    cudaFree(d_elementos);
    cudaFree(d_vector);
    cudaFree(d_resultado);
    free(matriz->elementos);
    free(matriz);
    free(vector);
    free(resultado);
    
    return 0;
}
