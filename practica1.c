#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>  // Para medir el tiempo

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

// Función para multiplicar matriz dispersa por vector
void multiplicar_matriz_vector(MatrizDispersa* matriz, double* vector, double* resultado) {
    // Inicializar resultado a cero
    for (int i = 0; i < matriz->filas; i++) {
        resultado[i] = 0;
    }

    // Realizar multiplicación solo con elementos no nulos
    for (int i = 0; i < matriz->num_elementos; i++) {
        int fila = matriz->elementos[i].fila;
        int columna = matriz->elementos[i].columna;
        double valor = matriz->elementos[i].valor;
        resultado[fila] += valor * vector[columna];
        // Imprimimos el resultado parcial
        printf("Resultado[%d] = %.2f\n", fila, resultado[fila]);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Uso: %s <archivo_matriz> <archivo_vector>\n", argv[0]);
        return 1;
    }

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
    int filas, columnas, dim_vector;
    MatrizDispersa* matriz = leer_matriz_dispersa(archivo_matriz, &filas, &columnas);
    double* vector = leer_vector(archivo_vector, &dim_vector);

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
    double* resultado = (double*)malloc(filas * sizeof(double));
    if (!resultado) {
        printf("Error al reservar memoria para el resultado\n");
        free(matriz->elementos);
        free(matriz);
        free(vector);
        return 1;
    }

    // Medir tiempo de ejecución
    clock_t inicio = clock();
    
    // Realizar multiplicación
    multiplicar_matriz_vector(matriz, vector, resultado);
    
    clock_t fin = clock();
    double tiempo = (double)(fin - inicio) / CLOCKS_PER_SEC;

    // Imprimir resultado final y tiempo
    printf("\nResultado final de la multiplicación matriz dispersa por vector:\n");
    for (int i = 0; i < filas; i++) {
        printf("Resultado[%d] = %.2f\n", i, resultado[i]);
    }
    
    printf("\nTiempo de ejecución: %.6f segundos\n", tiempo);

    // Liberar memoria
    free(matriz->elementos);
    free(matriz);
    free(vector);
    free(resultado);

    return 0;
}
