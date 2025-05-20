#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DENSIDAD 0.1  // 10% de elementos no nulos
#define MAX_VALOR 100

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Uso: %s <tamaño>\n", argv[0]);
        printf("Ejemplo: %s 5000\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);
    if (size <= 0) {
        printf("Error: El tamaño debe ser un número positivo\n");
        return 1;
    }

    // Generar matriz dispersa
    char matriz_filename[100];
    sprintf(matriz_filename, "matriz_%dx%d.txt", size, size);
    FILE *archivo_matriz = fopen(matriz_filename, "w");
    if (archivo_matriz == NULL) {
        printf("Error al crear el archivo de matriz\n");
        return 1;
    }

    // Generar vector
    char vector_filename[100];
    sprintf(vector_filename, "vector_%d.txt", size);
    FILE *archivo_vector = fopen(vector_filename, "w");
    if (archivo_vector == NULL) {
        printf("Error al crear el archivo de vector\n");
        fclose(archivo_matriz);
        return 1;
    }

    // Escribir dimensiones
    fprintf(archivo_matriz, "%d %d\n", size, size);
    fprintf(archivo_vector, "%d\n", size);

    // Generar matriz dispersa
    srand(time(NULL));
    printf("Generando matriz dispersa de %dx%d...\n", size, size);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
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
    printf("Generando vector de %d elementos...\n", size);
    for (int i = 0; i < size; i++) {
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
    printf("- %s\n", matriz_filename);
    printf("- %s\n", vector_filename);
    return 0;
} 