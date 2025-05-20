#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>  // Incluimos MPI en lugar de OpenMP

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

// Función para multiplicar matriz dispersa por vector (versión paralela con MPI)
void multiplicar_matriz_vector_mpi(MatrizDispersa* matriz, double* vector, double* resultado, int rank, int size) {
    int elementos_por_proceso, inicio, fin;
    
    // Calcular cuántos elementos procesa cada proceso
    elementos_por_proceso = matriz->num_elementos / size;
    inicio = rank * elementos_por_proceso;
    if (rank == size - 1) {
        // El último proceso toma los elementos restantes
        fin = matriz->num_elementos;
    } else {
        fin = inicio + elementos_por_proceso;
    }
    
    // Cada proceso inicializa su resultado local
    double* resultado_local = (double*)calloc(matriz->filas, sizeof(double));
    if (!resultado_local) {
        printf("Error: No se pudo asignar memoria para resultado_local en proceso %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Cada proceso calcula su parte de la multiplicación
    for (int i = inicio; i < fin; i++) {
        int fila = matriz->elementos[i].fila;
        int columna = matriz->elementos[i].columna;
        double valor = matriz->elementos[i].valor;
        resultado_local[fila] += valor * vector[columna];
    }
    
    // Sumamos todos los resultados locales en el proceso 0
    MPI_Reduce(resultado_local, resultado, matriz->filas, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    free(resultado_local);
}

int main(int argc, char *argv[]) {
    int rank, size;
    double tiempo_inicio, tiempo_fin;
    
    // Inicializar MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc != 3) {
        if (rank == 0) {
            printf("Uso: %s <archivo_matriz> <archivo_vector>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    // Declarar variables
    MatrizDispersa* matriz = NULL;
    double* vector = NULL;
    double* resultado = NULL;
    int filas = 0, columnas = 0, dim_vector = 0;
    
    // Solo el proceso 0 lee los archivos y distribuye los datos
    if (rank == 0) {
        // Abrir archivos
        FILE *archivo_matriz = fopen(argv[1], "r");
        FILE *archivo_vector = fopen(argv[2], "r");
        
        if (!archivo_matriz || !archivo_vector) {
            printf("Error al abrir los archivos\n");
            if (archivo_matriz) fclose(archivo_matriz);
            if (archivo_vector) fclose(archivo_vector);
            MPI_Abort(MPI_COMM_WORLD, 1);
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
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        // Verificar dimensiones
        if (columnas != dim_vector) {
            printf("Error: Las dimensiones no son compatibles para la multiplicación\n");
            printf("Matriz: %dx%d, Vector: %d\n", filas, columnas, dim_vector);
            free(matriz->elementos);
            free(matriz);
            free(vector);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        // Reservar memoria para el resultado
        resultado = (double*)calloc(filas, sizeof(double));
        if (!resultado) {
            printf("Error al reservar memoria para el resultado\n");
            free(matriz->elementos);
            free(matriz);
            free(vector);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }
    
    // Difundir las dimensiones a todos los procesos
    MPI_Bcast(&filas, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&columnas, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dim_vector, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Todos los procesos excepto el 0 crean espacio para el vector
    if (rank != 0) {
        vector = (double*)malloc(dim_vector * sizeof(double));
        if (!vector) {
            printf("Error al reservar memoria para el vector en proceso %d\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // También crear espacio para resultado (solo para la operación reduce)
        resultado = (double*)calloc(filas, sizeof(double));
        if (!resultado) {
            printf("Error al reservar memoria para el resultado en proceso %d\n", rank);
            free(vector);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    // Difundir el vector a todos los procesos
    MPI_Bcast(vector, dim_vector, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Crear la estructura de matriz en cada proceso si no es el proceso 0
    if (rank != 0) {
        matriz = (MatrizDispersa*)malloc(sizeof(MatrizDispersa));
        if (!matriz) {
            printf("Error al reservar memoria para la matriz en proceso %d\n", rank);
            free(vector);
            free(resultado);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        matriz->filas = filas;
        matriz->columnas = columnas;
    }
    
    // Difundir el número de elementos no nulos
    MPI_Bcast(&(matriz->num_elementos), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Cada proceso que no es el 0 crea espacio para los elementos
    if (rank != 0) {
        matriz->elementos = (Elemento*)malloc(matriz->num_elementos * sizeof(Elemento));
        if (!matriz->elementos) {
            printf("Error al reservar memoria para los elementos en proceso %d\n", rank);
            free(matriz);
            free(vector);
            free(resultado);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    // Crear un tipo MPI para Elemento
    MPI_Datatype MPI_ELEMENTO;
    int blocklengths[3] = {1, 1, 1};
    MPI_Aint displacements[3];
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_DOUBLE};
    
    // Calcular los desplazamientos en bytes
    Elemento dummy;
    MPI_Aint base_address;
    MPI_Get_address(&dummy, &base_address);
    MPI_Get_address(&dummy.fila, &displacements[0]);
    MPI_Get_address(&dummy.columna, &displacements[1]);
    MPI_Get_address(&dummy.valor, &displacements[2]);
    
    // Hacer desplazamientos relativos
    displacements[0] = MPI_Aint_diff(displacements[0], base_address);
    displacements[1] = MPI_Aint_diff(displacements[1], base_address);
    displacements[2] = MPI_Aint_diff(displacements[2], base_address);
    
    // Crear y comprometer el tipo
    MPI_Type_create_struct(3, blocklengths, displacements, types, &MPI_ELEMENTO);
    MPI_Type_commit(&MPI_ELEMENTO);
    
    // Difundir los elementos a todos los procesos
    MPI_Bcast(matriz->elementos, matriz->num_elementos, MPI_ELEMENTO, 0, MPI_COMM_WORLD);
    
    // Barrera de sincronización antes de empezar a medir tiempo
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Realizar multiplicación en paralelo y medir tiempo
    if (rank == 0) {
        printf("Ejecutando multiplicación matriz-vector con MPI...\n");
        printf("Número de procesos: %d\n", size);
    }
    
    tiempo_inicio = MPI_Wtime();
    multiplicar_matriz_vector_mpi(matriz, vector, resultado, rank, size);
    tiempo_fin = MPI_Wtime();
    
    // Imprimir resultado y tiempo en el proceso 0
    if (rank == 0) {
        printf("Resultado de la multiplicación matriz dispersa por vector (MPI):\n");
        for (int i = 0; i < filas; i++) {
            printf("%.2f\n", resultado[i]);
        }
        
        printf("Tiempo de ejecución MPI: %f segundos\n", tiempo_fin - tiempo_inicio);
        
        // Medimos también el tiempo total de ejecución
        double tiempo_total_inicio, tiempo_total_fin;
        MPI_Reduce(&tiempo_inicio, &tiempo_total_inicio, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&tiempo_fin, &tiempo_total_fin, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        printf("Tiempo total de ejecución (desde el primer proceso hasta el último): %f segundos\n", 
               tiempo_total_fin - tiempo_total_inicio);
    } else {
        // Los demás procesos participan en la reducción
        MPI_Reduce(&tiempo_inicio, NULL, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&tiempo_fin, NULL, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }
    
    // Liberar memoria en cada proceso
    free(matriz->elementos);
    free(matriz);
    free(vector);
    free(resultado);
    
    // Liberar el tipo MPI
    MPI_Type_free(&MPI_ELEMENTO);
    
    // Finalizar MPI
    MPI_Finalize();
    
    return 0;
}
