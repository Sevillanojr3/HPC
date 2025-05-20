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
    // Índice global del hilo
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Memoria compartida para almacenar resultados intermedios
    extern __shared__ double temp_resultado[];
    
    // Cada hilo procesa un elemento no nulo
    if (idx < num_elementos) {
        Elemento e = elementos[idx];
        double valor_prod = e.valor * vector[e.columna];
        atomicAdd(&resultado[e.fila], valor_prod);
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
    
    // Declarar todas las variables al inicio
    int dim_vector;
    int filas, columnas;
    FILE *archivo_matriz = NULL;
    FILE *archivo_vector = NULL;
    MatrizDispersa* matriz = NULL;
    double* vector = NULL;
    double* resultado = NULL;
    
    // Variables para CUDA
    Elemento* d_elementos = NULL;
    double* d_vector = NULL;
    double* d_resultado = NULL;
    cudaError_t error = cudaSuccess;
    
    // Variables para kernel
    int block_size = 256;
    int num_blocks = 0;
    int shared_mem_size = 0;
    
    // Variables para medir tiempo
    struct timeval inicio, fin;
    double tiempo = 0.0;
    
    // Cargar datos
    archivo_matriz = fopen(argv[1], "r");
    archivo_vector = fopen(argv[2], "r");
    
    if (!archivo_matriz || !archivo_vector) {
        printf("Error al abrir los archivos\n");
        if (archivo_matriz) fclose(archivo_matriz);
        if (archivo_vector) fclose(archivo_vector);
        return 1;
    }
    
    matriz = leer_matriz_dispersa(archivo_matriz, &filas, &columnas);
    vector = leer_vector(archivo_vector, &dim_vector);
    
    fclose(archivo_matriz);
    fclose(archivo_vector);
    
    if (!matriz || !vector) {
        if (matriz) {
            free(matriz->elementos);
            free(matriz);
        }
        if (vector) free(vector);
        return 1;
    }
    
    // Verificar dimensiones
    if (columnas != dim_vector) {
        printf("Error: Las dimensiones no son compatibles\n");
        free(matriz->elementos);
        free(matriz);
        free(vector);
        return 1;
    }
    
    // Verificar que hay elementos no nulos para procesar
    if (matriz->num_elementos == 0) {
        printf("Advertencia: La matriz está vacía (todos los elementos son cero)\n");
        resultado = (double*)calloc(matriz->filas, sizeof(double));
        if (resultado) free(resultado);
        free(matriz->elementos);
        free(matriz);
        free(vector);
        return 0;
    }
    
    // Asignar memoria para elementos
    error = cudaMalloc(&d_elementos, matriz->num_elementos * sizeof(Elemento));
    if (error != cudaSuccess) {
        printf("Error CUDA al reservar memoria para elementos: %s\n", cudaGetErrorString(error));
        goto cleanup;
    }
    
    // Asignar memoria para vector
    error = cudaMalloc(&d_vector, dim_vector * sizeof(double));
    if (error != cudaSuccess) {
        printf("Error CUDA al reservar memoria para vector: %s\n", cudaGetErrorString(error));
        goto cleanup;
    }
    
    // Asignar memoria para resultado
    error = cudaMalloc(&d_resultado, matriz->filas * sizeof(double));
    if (error != cudaSuccess) {
        printf("Error CUDA al reservar memoria para resultado: %s\n", cudaGetErrorString(error));
        goto cleanup;
    }
    
    // Copiar datos a GPU
    error = cudaMemcpy(d_elementos, matriz->elementos, matriz->num_elementos * sizeof(Elemento), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("Error CUDA al copiar elementos: %s\n", cudaGetErrorString(error));
        goto cleanup;
    }
    
    error = cudaMemcpy(d_vector, vector, dim_vector * sizeof(double), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("Error CUDA al copiar vector: %s\n", cudaGetErrorString(error));
        goto cleanup;
    }
    
    error = cudaMemset(d_resultado, 0, matriz->filas * sizeof(double));
    if (error != cudaSuccess) {
        printf("Error CUDA al inicializar resultado: %s\n", cudaGetErrorString(error));
        goto cleanup;
    }
    
    // Configurar la ejecución del kernel
    num_blocks = (matriz->num_elementos + block_size - 1) / block_size;
    
    // Mostrar información sobre la ejecución
    printf("Ejecutando kernel CUDA con %d bloques de %d hilos cada uno\n", num_blocks, block_size);
    printf("Elementos no nulos: %d, Matriz: %dx%d\n", matriz->num_elementos, matriz->filas, matriz->columnas);
    
    // Medir tiempo
    gettimeofday(&inicio, NULL);
    
    // Ejecutar kernel
    multiplicar_matriz_vector_cuda<<<num_blocks, block_size, shared_mem_size>>>(
        d_elementos, matriz->num_elementos, d_vector, d_resultado, matriz->filas);
    
    // Verificar errores del kernel
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Error CUDA en kernel: %s\n", cudaGetErrorString(error));
        goto cleanup;
    }
    
    // Sincronizar dispositivo
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        printf("Error CUDA al sincronizar: %s\n", cudaGetErrorString(error));
        goto cleanup;
    }
    
    gettimeofday(&fin, NULL);
    tiempo = (fin.tv_sec - inicio.tv_sec) + (fin.tv_usec - inicio.tv_usec) / 1000000.0;
    
    // Copiar resultado de vuelta a CPU
    resultado = (double*)calloc(matriz->filas, sizeof(double));
    if (!resultado) {
        printf("Error al reservar memoria para resultado en CPU\n");
        goto cleanup;
    }
    
    error = cudaMemcpy(resultado, d_resultado, matriz->filas * sizeof(double), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        printf("Error CUDA al copiar resultado: %s\n", cudaGetErrorString(error));
        if (resultado) free(resultado);
        resultado = NULL;
        goto cleanup;
    }
    
    // Imprimir tiempo
    printf("Tiempo de ejecución CUDA: %.6f segundos\n", tiempo);
    
    // Liberar memoria CPU
    if (resultado) free(resultado);
    
cleanup:
    // Liberar memoria GPU
    if (d_elementos) cudaFree(d_elementos);
    if (d_vector) cudaFree(d_vector);
    if (d_resultado) cudaFree(d_resultado);
    
    // Liberar memoria CPU de estructuras
    if (matriz) {
        if (matriz->elementos) free(matriz->elementos);
        free(matriz);
    }
    if (vector) free(vector);
    
    return 0;
}
