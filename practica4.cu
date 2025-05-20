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

// Función para leer la matriz dispersa
MatrizDispersa* leer_matriz_dispersa(FILE *archivo, int *filas, int *columnas) {
    if (fscanf(archivo, "%d %d", filas, columnas) != 2) {
        printf("Error al leer dimensiones\n");
        return NULL;
    }

    // Primero contamos elementos no nulos
    double valor;
    int num_elementos = 0;
    for (int i = 0; i < *filas; i++) {
        for (int j = 0; j < *columnas; j++) {
            if (fscanf(archivo, "%lf", &valor) != 1) {
                printf("Error al leer elemento [%d,%d]\n", i, j);
                return NULL;
            }
            if (valor != 0) num_elementos++;
        }
    }

    // Reservamos memoria para los elementos no nulos
    MatrizDispersa* matriz = (MatrizDispersa*)malloc(sizeof(MatrizDispersa));
    if (!matriz) {
        printf("Error al reservar memoria para la matriz\n");
        return NULL;
    }

    matriz->elementos = (Elemento*)malloc(num_elementos * sizeof(Elemento));
    if (!matriz->elementos) {
        printf("Error al reservar memoria para los elementos\n");
        free(matriz);
        return NULL;
    }

    // Volvemos al inicio del archivo
    rewind(archivo);
    fscanf(archivo, "%d %d", filas, columnas);

    // Leemos y guardamos solo elementos no nulos
    int idx = 0;
    for (int i = 0; i < *filas; i++) {
        for (int j = 0; j < *columnas; j++) {
            if (fscanf(archivo, "%lf", &valor) != 1) {
                printf("Error al leer elemento [%d,%d]\n", i, j);
                free(matriz->elementos);
                free(matriz);
                return NULL;
            }
            if (valor != 0) {
                matriz->elementos[idx].fila = i;
                matriz->elementos[idx].columna = j;
                matriz->elementos[idx].valor = valor;
                idx++;
            }
        }
    }

    matriz->num_elementos = num_elementos;
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

    for (int i = 0; i < *dimension; i++) {
        if (fscanf(archivo, "%lf", &vector[i]) != 1) {
            printf("Error al leer elemento %d del vector\n", i);
            free(vector);
            return NULL;
        }
    }

    return vector;
}

// Kernel CUDA para multiplicar matriz dispersa por vector
__global__ void matriz_vector_kernel(Elemento* elementos, int num_elementos, double* vector, double* resultado, int filas) {
    // Calcular índice global del hilo
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Cada hilo procesa un elemento no nulo de la matriz
    if (idx < num_elementos) {
        int fila = elementos[idx].fila;
        int columna = elementos[idx].columna;
        double valor = elementos[idx].valor;
        
        // Multiplicar el elemento por el correspondiente del vector
        // y usamos atomicAdd para evitar condiciones de carrera
        atomicAdd(&resultado[fila], valor * vector[columna]);
    }
}

// Función para multiplicar matriz dispersa por vector usando CUDA
void multiplicar_matriz_vector_cuda(MatrizDispersa* matriz, double* vector, double* resultado) {
    // Variables para dispositivo (GPU)
    Elemento* d_elementos;
    double* d_vector;
    double* d_resultado;
    
    // Reservar memoria en GPU
    cudaMalloc((void**)&d_elementos, matriz->num_elementos * sizeof(Elemento));
    cudaMalloc((void**)&d_vector, matriz->columnas * sizeof(double));
    cudaMalloc((void**)&d_resultado, matriz->filas * sizeof(double));
    
    // Inicializar el vector resultado a ceros en GPU
    cudaMemset(d_resultado, 0, matriz->filas * sizeof(double));
    
    // Copiar datos de CPU a GPU
    cudaMemcpy(d_elementos, matriz->elementos, matriz->num_elementos * sizeof(Elemento), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, vector, matriz->columnas * sizeof(double), cudaMemcpyHostToDevice);
    
    // Configurar la ejecución del kernel
    int blockSize = 256;
    int numBlocks = (matriz->num_elementos + blockSize - 1) / blockSize;
    
    // Ejecutar el kernel
    matriz_vector_kernel<<<numBlocks, blockSize>>>(d_elementos, matriz->num_elementos, d_vector, d_resultado, matriz->filas);
    
    // Esperar a que termine el kernel
    cudaDeviceSynchronize();
    
    // Copiar el resultado de GPU a CPU
    cudaMemcpy(resultado, d_resultado, matriz->filas * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Liberar memoria en GPU
    cudaFree(d_elementos);
    cudaFree(d_vector);
    cudaFree(d_resultado);
}

// Función auxiliar para manejo de errores CUDA
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
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
        printf("Error: Las dimensiones no son compatibles para la multiplicación\n");
        printf("Matriz: %dx%d, Vector: %d\n", filas, columnas, dim_vector);
        free(matriz->elementos);
        free(matriz);
        free(vector);
        return 1;
    }
    
    // Reservar memoria para el resultado
    resultado = (double*)calloc(filas, sizeof(double));
    if (!resultado) {
        printf("Error al reservar memoria para el resultado\n");
        free(matriz->elementos);
        free(matriz);
        free(vector);
        return 1;
    }
    
    // Mostrar información sobre la matriz y el cálculo
    printf("Multiplicación de matriz dispersa por vector con CUDA\n");
    printf("Matriz: %dx%d con %d elementos no nulos\n", 
           matriz->filas, matriz->columnas, matriz->num_elementos);
    
    // Configurar y comprobar dispositivo CUDA
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess || deviceCount == 0) {
        printf("No se encontraron dispositivos CUDA compatibles.\n");
        free(matriz->elementos);
        free(matriz);
        free(vector);
        free(resultado);
        return 1;
    }
    
    // Información del dispositivo CUDA
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Usando GPU: %s\n", deviceProp.name);
    
    // Crear eventos CUDA para medir el tiempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Registrar tiempo de inicio
    cudaEventRecord(start, 0);
    
    // Realizar multiplicación en CUDA
    multiplicar_matriz_vector_cuda(matriz, vector, resultado);
    
    // Registrar tiempo de fin
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    // Calcular tiempo transcurrido
    float tiempo_ms;
    cudaEventElapsedTime(&tiempo_ms, start, stop);
    
    // Mostrar resultado y tiempo
    printf("Resultado de la multiplicación matriz dispersa por vector (CUDA):\n");
    for (int i = 0; i < filas; i++) {
        printf("%.2f\n", resultado[i]);
    }
    
    printf("Tiempo de ejecución CUDA: %f milisegundos\n", tiempo_ms);
    printf("Tiempo de ejecución CUDA: %f segundos\n", tiempo_ms / 1000.0);
    
    // Destruir eventos CUDA
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Liberar memoria
    free(matriz->elementos);
    free(matriz);
    free(vector);
    free(resultado);
    
    return 0;
}
