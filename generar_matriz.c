#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define FILAS 100000
#define COLUMNAS 100000
#define DIM_VECTOR 100000
#define DENSIDAD 0.1  // 10% de elementos no nulos
#define MAX_VALOR 100

int main() {
    // Generar matriz dispersa
    FILE *archivo_matriz = fopen("matriz_100000x100000.txt", "w");
    if (archivo_matriz == NULL) {
        printf("Error al crear el archivo de matriz\n");
        return 1;
    }

    // Generar vector
    FILE *archivo_vector = fopen("vector_100000.txt", "w");
    if (archivo_vector == NULL) {
        printf("Error al crear el archivo de vector\n");
        fclose(archivo_matriz);
        return 1;
    }

    // Escribir dimensiones
    fprintf(archivo_matriz, "%d %d\n", FILAS, COLUMNAS);
    fprintf(archivo_vector, "%d\n", DIM_VECTOR);

    // Generar matriz dispersa
    srand(time(NULL));
    printf("Generando matriz dispersa de %dx%d...\n", FILAS, COLUMNAS);

    for (int i = 0; i < FILAS; i++) {
        for (int j = 0; j < COLUMNAS; j++) {
            if ((double)rand() / RAND_MAX < DENSIDAD) {
                int valor = rand() % MAX_VALOR + 1;  // Números enteros entre 1 y 100
                fprintf(archivo_matriz, "%d ", valor);
            } else {
                fprintf(archivo_matriz, "0 ");
            }
        }
        fprintf(archivo_matriz, "\n");
    }

    // Generar vector
    printf("Generando vector de %d elementos...\n", DIM_VECTOR);
    for (int i = 0; i < DIM_VECTOR; i++) {
        if ((double)rand() / RAND_MAX < DENSIDAD) {
            int valor = rand() % MAX_VALOR + 1;  // Números enteros entre 1 y 100
            fprintf(archivo_vector, "%d ", valor);
        } else {
            fprintf(archivo_vector, "0 ");
        }
    }
    fprintf(archivo_vector, "\n");

    fclose(archivo_matriz);
    fclose(archivo_vector);
    printf("Archivos generados exitosamente:\n");
    printf("- matriz_100000x100000.txt\n");
    printf("- vector_100000.txt\n");
    return 0;
} 