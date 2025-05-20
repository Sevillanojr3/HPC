#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

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

// Kernel CUDA optimizado para multiplicar matriz dispersa por vector
__global__ void matriz_vector_kernel(Elemento* elementos, int num_elementos, 
                                   double* vector, double* resultado, int filas) {
    // Calcular índice global del hilo
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Memoria compartida para el vector
    extern __shared__ double s_vector[];
    
    // Cargar parte del vector en memoria compartida
    int tid = threadIdx.x;
    int stride = blockDim.x;
    for (int i = tid; i < filas; i += stride) {
        s_vector[i] = vector[i];
    }
    __syncthreads();
    
    // Cada hilo procesa un elemento no nulo de la matriz
    if (idx < num_elementos) {
        int fila = elementos[idx].fila;
        int columna = elementos[idx].columna;
        double valor = elementos[idx].valor;
        
        // Multiplicar el elemento por el correspondiente del vector
        // y usamos atomicAdd para evitar condiciones de carrera
        double temp = valor * s_vector[columna];
        atomicAdd((double*)&resultado[fila], temp);
    }
}

// Función optimizada para multiplicar matriz dispersa por vector usando CUDA
void multiplicar_matriz_vector_cuda(MatrizDispersa* matriz, double* vector, double* resultado) {
    printf("Iniciando multiplicación CUDA...\n");
    printf("Dimensiones: %dx%d, Elementos no nulos: %d\n", 
           matriz->filas, matriz->columnas, matriz->num_elementos);
    
    // Variables para dispositivo (GPU)
    Elemento* d_elementos = NULL;
    double* d_vector = NULL;
    double* d_resultado = NULL;
    
    // Reservar memoria en GPU
    printf("Reservando memoria en GPU...\n");
    CUDA_CHECK(cudaMalloc((void**)&d_elementos, matriz->num_elementos * sizeof(Elemento)));
    CUDA_CHECK(cudaMalloc((void**)&d_vector, matriz->columnas * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_resultado, matriz->filas * sizeof(double)));
    
    // Inicializar el vector resultado a ceros en GPU
    printf("Inicializando resultado a ceros...\n");
    CUDA_CHECK(cudaMemset(d_resultado, 0, matriz->filas * sizeof(double)));
    
    // Crear streams para transferencias asíncronas
    printf("Creando streams CUDA...\n");
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));
    
    // Copiar datos de CPU a GPU de forma asíncrona
    printf("Copiando datos a GPU...\n");
    CUDA_CHECK(cudaMemcpyAsync(d_elementos, matriz->elementos, 
                              matriz->num_elementos * sizeof(Elemento), 
                              cudaMemcpyHostToDevice, stream1));
    CUDA_CHECK(cudaMemcpyAsync(d_vector, vector, 
                              matriz->columnas * sizeof(double), 
                              cudaMemcpyHostToDevice, stream2));
    
    // Esperar a que terminen las transferencias
    printf("Esperando transferencias...\n");
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));
    
    // Configurar la ejecución del kernel
    int blockSize = 256;
    int numBlocks = (matriz->num_elementos + blockSize - 1) / blockSize;
    int sharedMemSize = matriz->columnas * sizeof(double);
    
    printf("Ejecutando kernel con %d bloques de %d hilos...\n", numBlocks, blockSize);
    
    // Ejecutar el kernel
    matriz_vector_kernel<<<numBlocks, blockSize, sharedMemSize>>>
        (d_elementos, matriz->num_elementos, d_vector, d_resultado, matriz->filas);
    
    // Verificar errores del kernel
    CUDA_CHECK(cudaGetLastError());
    
    // Esperar a que termine el kernel
    printf("Esperando finalización del kernel...\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copiar el resultado de GPU a CPU
    printf("Copiando resultado a CPU...\n");
    CUDA_CHECK(cudaMemcpy(resultado, d_resultado, matriz->filas * sizeof(double), 
                         cudaMemcpyDeviceToHost));
    
    // Liberar memoria en GPU
    printf("Liberando memoria GPU...\n");
    CUDA_CHECK(cudaFree(d_elementos));
    CUDA_CHECK(cudaFree(d_vector));
    CUDA_CHECK(cudaFree(d_resultado));
    
    // Destruir streams
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    
    printf("Multiplicación CUDA completada.\n");
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
                    Elemento* temp = realloc(matriz->elementos, num_elementos * sizeof(Elemento));
                    if (!temp) {
                        printf("Error al redimensionar memoria\n");
                        free(matriz->elementos);
                        free(matriz);
                        return NULL;
                    }
                    matriz->elementos = temp;
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
        Elemento* temp = realloc(matriz->elementos, idx * sizeof(Elemento));
        if (temp) {
            matriz->elementos = temp;
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
    
    // Declarar variables
    MatrizDispersa* matriz = NULL;
    double* vector = NULL;
    double* resultado = NULL;
    int filas = 0, columnas = 0, dim_vector = 0;
    
    // Abrir archivos
    FILE *archivo_matriz = fopen(argv[1], "r");
    FILE *archivo_vector = fopen(argv[2], "r");
    
    if (!archivo_matriz || !archivo_vector) {
        printf("Error al abrir los archivos\n");
        if (archivo_matriz) fclose(archivo_matriz);
        if (archivo_vector) fclose(archivo_vector);
        return 1;
    }
    
    // Leer matriz dispersa y vector
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
    
    // Reservar memoria para el resultado
    resultado = (double*)malloc(filas * sizeof(double));
    if (!resultado) {
        printf("Error al reservar memoria para el resultado\n");
        free(matriz->elementos);
        free(matriz);
        free(vector);
        return 1;
    }
    
    // Medir tiempo de ejecución
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Realizar multiplicación
    multiplicar_matriz_vector_cuda(matriz, vector, resultado);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float tiempo;
    cudaEventElapsedTime(&tiempo, start, stop);
    
    // Imprimir tiempo de ejecución
    printf("Tiempo de ejecución: %.6f segundos\n", tiempo / 1000.0);
    
    // Liberar memoria
    free(matriz->elementos);
    free(matriz);
    free(vector);
    free(resultado);
    
    // Destruir eventos CUDA
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
