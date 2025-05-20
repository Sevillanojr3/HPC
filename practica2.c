#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>  // Incluimos OpenMP

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

// Función para cargar un vector desde un archivo
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
    
    printf("Dimensión del vector: %d\n", *dimension);
    
    double* vector = (double*)malloc(*dimension * sizeof(double));
    if (!vector) {
        printf("Error al reservar memoria para el vector (%lu bytes)\n", 
               (unsigned long)(*dimension * sizeof(double)));
        fclose(archivo);
        return NULL;
    }
    
    printf("Leyendo vector...\n");
    for (int i = 0; i < *dimension; i++) {
        if (fscanf(archivo, "%lf", &vector[i]) != 1) {
            printf("Error al leer elemento %d del vector\n", i);
            free(vector);
            fclose(archivo);
            return NULL;
        }
    }
    
    fclose(archivo);
    printf("Vector cargado exitosamente\n");
    return vector;
}

// Función optimizada para cargar una matriz dispersa desde un archivo grande
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
    
    printf("Dimensiones de la matriz: %d x %d\n", matriz->filas, matriz->columnas);
    
    // Calcular cuántos elementos se pueden leer por lote para evitar exceso de memoria
    const long BUFFER_MAX = 10000; // Ajustar según la memoria disponible
    const long elementos_totales = (long)matriz->filas * matriz->columnas;
    
    printf("Total de elementos en la matriz: %ld\n", elementos_totales);
    printf("Iniciando primera pasada para contar elementos no nulos...\n");
    
    // Primera pasada: contar elementos no nulos
    int elementos_no_nulos = 0;
    double valor;
    long elementos_procesados = 0;
    double porcentaje_anterior = -1.0;
    
    for (int i = 0; i < matriz->filas; i++) {
        for (int j = 0; j < matriz->columnas; j++) {
            if (fscanf(archivo, "%lf", &valor) != 1) {
                printf("Error al leer elemento (%d,%d) de la matriz\n", i, j);
                free(matriz);
                fclose(archivo);
                return NULL;
            }
            
            if (valor != 0.0) {
                elementos_no_nulos++;
            }
            
            elementos_procesados++;
            double porcentaje = (elementos_procesados * 100.0) / elementos_totales;
            if ((int)porcentaje > (int)porcentaje_anterior) {
                printf("\rProcesando primera pasada: %.1f%% completo (%d elementos no nulos encontrados)", 
                       porcentaje, elementos_no_nulos);
                fflush(stdout);
                porcentaje_anterior = porcentaje;
            }
        }
    }
    
    printf("\nPrimera pasada completada. Total de elementos no nulos: %d (%.2f%% de la matriz)\n", 
           elementos_no_nulos, (elementos_no_nulos * 100.0) / elementos_totales);
    
    // Rebobinar el archivo para leer de nuevo
    rewind(archivo);
    
    // Saltamos la línea de dimensiones
    fscanf(archivo, "%d %d", &matriz->filas, &matriz->columnas);
    
    // Reservar memoria para elementos no nulos
    printf("Reservando memoria para %d elementos no nulos (%.2f MB)...\n", 
           elementos_no_nulos, (elementos_no_nulos * sizeof(Elemento)) / (1024.0 * 1024.0));
           
    matriz->elementos = (Elemento*)malloc(elementos_no_nulos * sizeof(Elemento));
    if (!matriz->elementos) {
        printf("Error al reservar memoria para los elementos (%lu bytes)\n", 
               (unsigned long)(elementos_no_nulos * sizeof(Elemento)));
        free(matriz);
        fclose(archivo);
        return NULL;
    }
    matriz->num_elementos = elementos_no_nulos;
    
    // Segunda pasada: almacenar elementos no nulos
    printf("Iniciando segunda pasada para almacenar elementos no nulos...\n");
    int indice = 0;
    elementos_procesados = 0;
    porcentaje_anterior = -1.0;
    
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
                matriz->elementos[indice].fila = i;
                matriz->elementos[indice].columna = j;
                matriz->elementos[indice].valor = valor;
                indice++;
            }
            
            elementos_procesados++;
            double porcentaje = (elementos_procesados * 100.0) / elementos_totales;
            if ((int)porcentaje > (int)porcentaje_anterior) {
                printf("\rProcesando segunda pasada: %.1f%% completo", porcentaje);
                fflush(stdout);
                porcentaje_anterior = porcentaje;
            }
        }
    }
    
    printf("\nSegunda pasada completada. Matriz dispersa cargada exitosamente.\n");
    fclose(archivo);
    return matriz;
}

// Multiplicación optimizada de matriz dispersa por vector
void multiplicar_matriz_dispersa_vector(const MatrizDispersa* matriz, const double* vector, double* resultado) {
    // Inicializar resultado con ceros
    memset(resultado, 0, matriz->filas * sizeof(double));
    
    printf("Iniciando multiplicación paralela con %d hilos...\n", omp_get_max_threads());
    
    // Multiplicación paralela usando la estructura dispersa
    #pragma omp parallel
    {
        // Buen equilibrio de carga para matrices dispersas
        #pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < matriz->num_elementos; i++) {
            Elemento e = matriz->elementos[i];
            #pragma omp atomic
            resultado[e.fila] += e.valor * vector[e.columna];
        }
    }
    
    printf("Multiplicación completada\n");
}

// Función principal optimizada
void leer_y_multiplicar_paralelo(const char* archivo_matriz_nombre, const char* archivo_vector_nombre, double** resultado_final, int* filas_final) {
    printf("Cargando vector desde %s...\n", archivo_vector_nombre);
    
    // Cargar vector
    int dim_vector;
    double* vector = cargar_vector(archivo_vector_nombre, &dim_vector);
    if (!vector) {
        printf("Error al cargar el vector\n");
        *resultado_final = NULL;
        *filas_final = 0;
        return;
    }
    
    printf("Cargando matriz dispersa desde %s...\n", archivo_matriz_nombre);
    
    // Cargar matriz dispersa
    MatrizDispersa* matriz = cargar_matriz_dispersa(archivo_matriz_nombre);
    if (!matriz) {
        printf("Error al cargar la matriz dispersa\n");
        free(vector);
        *resultado_final = NULL;
        *filas_final = 0;
        return;
    }
    
    // Verificar compatibilidad
    if (matriz->columnas != dim_vector) {
        printf("Error: Las dimensiones no son compatibles para la multiplicación\n");
        printf("Matriz: %dx%d, Vector: %d\n", matriz->filas, matriz->columnas, dim_vector);
        free(vector);
        free(matriz->elementos);
        free(matriz);
        *resultado_final = NULL;
        *filas_final = 0;
        return;
    }
    
    // Reservar memoria para el resultado
    printf("Reservando memoria para el vector resultado (%d elementos)...\n", matriz->filas);
    double* resultado = (double*)calloc(matriz->filas, sizeof(double));
    if (!resultado) {
        printf("Error al reservar memoria para el resultado\n");
        free(vector);
        free(matriz->elementos);
        free(matriz);
        *resultado_final = NULL;
        *filas_final = 0;
        return;
    }
    
    // Realizar la multiplicación
    multiplicar_matriz_dispersa_vector(matriz, vector, resultado);
    
    // Guardar información antes de liberar memoria
    int filas_matriz = matriz->filas;
    
    // Liberar memoria y devolver resultado
    free(vector);
    free(matriz->elementos);
    free(matriz);
    
    *resultado_final = resultado;
    *filas_final = filas_matriz;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Uso: %s <archivo_matriz> <archivo_vector>\n", argv[0]);
        return 1;
    }

    // Medir tiempo de ejecución
    double tiempo_inicio, tiempo_fin;
    double* resultado = NULL;
    int filas = 0;
    
    printf("Ejecutando lectura y multiplicación matriz-vector con OpenMP...\n");
    printf("Número de hilos disponibles: %d\n", omp_get_max_threads());
    
    tiempo_inicio = omp_get_wtime();
    leer_y_multiplicar_paralelo(argv[1], argv[2], &resultado, &filas);
    tiempo_fin = omp_get_wtime();

    if (resultado && filas > 0) {
        // Imprimir resultado (solo los primeros 10 elementos si hay muchos)
        printf("Resultado de la multiplicación matriz dispersa por vector (paralelo):\n");
        int elementos_a_mostrar = (filas > 10) ? 10 : filas;
        for (int i = 0; i < elementos_a_mostrar; i++) {
            printf("resultado[%d] = %.2f\n", i, resultado[i]);
        }
        if (filas > 10) {
            printf("... (y %d elementos más)\n", filas - 10);
        }
        
        printf("Tiempo de ejecución paralelo: %f segundos\n", tiempo_fin - tiempo_inicio);
        
        // Liberar memoria
        free(resultado);
    } else {
        printf("Error en la ejecución\n");
        return 1;
    }

    return 0;
}
