Programa de Multiplicación de Matriz Dispersa por Vector
====================================================

Este programa implementa la multiplicación de una matriz dispersa por un vector de forma secuencial. El programa lee una matriz dispersa y un vector desde archivos de texto y realiza su multiplicación, mostrando el resultado en la salida estándar.

Características:
--------------
- Lee una matriz dispersa y un vector desde archivos de texto
- Almacena solo los elementos no nulos de la matriz para optimizar memoria
- Verifica la compatibilidad de dimensiones para la multiplicación
- Realiza la multiplicación de forma secuencial
- Muestra el resultado en formato de vector

Compilación:
-----------
Para compilar el programa, utilice el siguiente comando:
gcc practica1.c -o practica1

Uso:
----
El programa se ejecuta desde la línea de comandos de la siguiente manera:
./practica1 <archivo_matriz> <archivo_vector>

Donde:
- <archivo_matriz>: Ruta al archivo que contiene la matriz dispersa
- <archivo_vector>: Ruta al archivo que contiene el vector

Formato de los archivos de entrada:
---------------------------------
1. Archivo de matriz dispersa:
   - Primera línea: dos números enteros separados por espacio que indican las dimensiones (filas columnas)
   - Líneas siguientes: elementos de la matriz separados por espacios (los ceros se pueden incluir)

2. Archivo de vector:
   - Primera línea: un número entero que indica la dimensión del vector
   - Línea siguiente: elementos del vector separados por espacios

Ejemplo de formato:

matriz.txt:
3 3
0 2 0
0 0 0
5 0 3

vector.txt:
3
1 2 3

Salida:
-------
El programa mostrará en la salida estándar:
1. Un mensaje indicando el resultado
2. Los elementos del vector resultado, con dos decimales

Notas importantes:
-----------------
- Las dimensiones deben ser compatibles para la multiplicación
- El número de columnas de la matriz debe ser igual a la dimensión del vector
- Los elementos de la matriz y el vector deben ser números reales
- La matriz puede contener ceros, pero solo se almacenarán los elementos no nulos 