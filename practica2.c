#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>  // Incluimos OpenMP
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

// Función optimizada para cargar un vector desde un archivo
double* cargar_vector(const char* nombre_archivo, int* dimension) {
    FILE *archivo = fopen(nombre_archivo, "r");
    if (!archivo) {
        printf("Error al abrir el archivo %s\n", nombre_archivo);
        return NULL;
    }
    
    if (fscanf(archivo, "%d", dimension) != 1) {
        printf("Error al leer la dimensión del vector\n");
        fclose(archivo);
        return NULL;
    }
    
    double* vector = (double*)malloc(*dimension * sizeof(double));
    if (!vector) {
        printf("Error al reservar memoria para el vector\n");
        fclose(archivo);
        return NULL;
    }
    
    // Leer vector de una vez
    for (int i = 0; i < *dimension; i++) {
        if (fscanf(archivo, "%lf", &vector[i]) != 1) {
            printf("Error al leer elemento %d del vector\n", i);
            free(vector);
            fclose(archivo);
            return NULL;
        }
    }
    
    fclose(archivo);
    return vector;
}

// Función optimizada para cargar una matriz dispersa
MatrizDispersa* cargar_matriz_dispersa(const char* nombre_archivo) {
    FILE *archivo = fopen(nombre_archivo, "r");
    if (!archivo) {
        printf("Error al abrir el archivo %s\n", nombre_archivo);
        return NULL;
    }
    
    MatrizDispersa* matriz = (MatrizDispersa*)malloc(sizeof(MatrizDispersa));
    if (!matriz) {
        printf("Error al reservar memoria para la matriz\n");
        fclose(archivo);
        return NULL;
    }
    
    if (fscanf(archivo, "%d %d", &matriz->filas, &matriz->columnas) != 2) {
        printf("Error al leer dimensiones de la matriz\n");
        free(matriz);
        fclose(archivo);
        return NULL;
    }
    
    // Estimamos el número de elementos no nulos (10% de la matriz)
    int num_elementos = (matriz->filas * matriz->columnas) / 10;
    matriz->elementos = (Elemento*)malloc(num_elementos * sizeof(Elemento));
    if (!matriz->elementos) {
        printf("Error al reservar memoria para los elementos\n");
        free(matriz);
        fclose(archivo);
        return NULL;
    }
    
    // Leemos y guardamos solo elementos no nulos
    int idx = 0;
    double valor;
    for (int i = 0; i < matriz->filas; i++) {
        for (int j = 0; j < matriz->columnas; j++) {
            if (fscanf(archivo, "%lf", &valor) != 1) {
                printf("Error al leer elemento (%d,%d) de la matriz\n", i, j);
                free(matriz->elementos);
                free(matriz);
                fclose(archivo);
                return NULL;
            }
            
            if (valor != 0.0) {
                if (idx >= num_elementos) {
                    // Redimensionar si es necesario
                    num_elementos *= 2;
                    Elemento* temp = realloc(matriz->elementos, num_elementos * sizeof(Elemento));
                    if (!temp) {
                        printf("Error al redimensionar memoria\n");
                        free(matriz->elementos);
                        free(matriz);
                        fclose(archivo);
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
    fclose(archivo);
    return matriz;
}

// Multiplicación optimizada de matriz dispersa por vector usando OpenMP
void multiplicar_matriz_dispersa_vector(const MatrizDispersa* matriz, const double* vector, double* resultado) {
    // Inicializar resultado con ceros
    memset(resultado, 0, matriz->filas * sizeof(double));
    
    // Multiplicación paralela usando la estructura dispersa
    #pragma omp parallel
    {
        // Crear un array local para cada hilo
        double* resultado_local = (double*)calloc(matriz->filas, sizeof(double));
        
        // Dividir el trabajo por filas para mejor localidad de cache
        #pragma omp for schedule(guided)
        for (int i = 0; i < matriz->num_elementos; i++) {
            Elemento e = matriz->elementos[i];
            resultado_local[e.fila] += e.valor * vector[e.columna];
        }
        
        // Reducir resultados locales de forma atómica
        #pragma omp critical
        {
            for (int i = 0; i < matriz->filas; i++) {
                resultado[i] += resultado_local[i];
            }
        }
        
        free(resultado_local);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Uso: %s <archivo_matriz> <archivo_vector>\n", argv[0]);
        return 1;
    }
    
    // Configurar número de hilos OpenMP
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    
    // Cargar datos
    int dim_vector;
    MatrizDispersa* matriz = cargar_matriz_dispersa(argv[1]);
    double* vector = cargar_vector(argv[2], &dim_vector);
    
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
    
    // Reservar memoria para el resultado
    double* resultado = (double*)malloc(matriz->filas * sizeof(double));
    if (!resultado) {
        printf("Error al reservar memoria para el resultado\n");
        free(matriz->elementos);
        free(matriz);
        free(vector);
        return 1;
    }
    
    // Medir tiempo de ejecución
    struct timeval inicio, fin;
    gettimeofday(&inicio, NULL);
    
    // Realizar multiplicación
    multiplicar_matriz_dispersa_vector(matriz, vector, resultado);
    
    gettimeofday(&fin, NULL);
    double tiempo = (fin.tv_sec - inicio.tv_sec) + (fin.tv_usec - inicio.tv_usec) / 1000000.0;
    
    // Imprimir tiempo de ejecución y número de hilos
    printf("Tiempo de ejecución: %.6f segundos (OpenMP con %d hilos)\n", tiempo, num_threads);
    
    // Liberar memoria
    free(matriz->elementos);
    free(matriz);
    free(vector);
    free(resultado);
    return 0;
}
