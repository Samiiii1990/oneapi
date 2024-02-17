#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#include <chrono>  // Utilizo esta librería para medir los tiempos.

using namespace std;
using namespace sycl;

constexpr int nodos = 16;  // Acá se define la cantidad de nodos con la que queremos trabajar.
constexpr int block_length = 4;  // Acá se ajusta el tamaño del bloque.
constexpr int block_count = nodos; 
constexpr int max_distance = 100;
constexpr int infinite = nodos * max_distance;
constexpr int repetitions = 1; //Acá se setea la cantidad de iteraciones que se realizarán para la medición.


//Inicializar el grafo
void InicializarGrafo(vector<int>& grafo) {
    for (int i = 0; i < nodos; i++) {
        for (int j = 0; j < nodos; j++) {
            int cell = i * nodos + j;

            if (i == j) {
                grafo[cell] = 0;
            }
            else if (rand() % 2) {
                grafo[cell] = infinite;
            }
            else {
                grafo[cell] = rand() % max_distance + 1;
            }
        }
    }
}
//Implementa el algoritmo de Floyd-Warshall para calcular los caminos más cortos entre todos los pares de nodos del grafo
void FloydWarshall(vector<int>& grafo) {
    for (int k = 0; k < nodos; k++) {
        for (int i = 0; i < nodos; i++) {
            for (int j = 0; j < nodos; j++) {
                if (grafo[i * nodos + j] > grafo[i * nodos + k] + grafo[k * nodos + j]) {
                    grafo[i * nodos + j] = grafo[i * nodos + k] + grafo[k * nodos + j];
                }
            }
        }
    }
}

//Implementa una versión paralela del algoritmo de Floyd-Warshall utilizando la técnica de bloqueo (blocking) para dividir el cálculo en bloques. 
void BlockedFloydWarshall(queue& q, vector<int>& grafo) {
    for (int round = 0; round < block_count; round++) {
         //Definir el tamaño del bloque y buffers para las matrices A, B, y C
        auto R = range<2>(block_length, block_length);
        auto A = buffer<int, 2>(grafo.data(), R);
        auto B = buffer<int, 2>(grafo.data(), R);
        auto C = buffer<int, 2>(grafo.data(), R);

        //Enviar el trabajo a la cola (queue) mediante q.submit
        q.submit([&](handler& h) {
            auto accessor_A = A.get_access<access::mode::read>(h);
            auto accessor_B = B.get_access<access::mode::read>(h);
            auto accessor_C = C.get_access<access::mode::write>(h);
        
            // Definir el kernel paralelo utilizando h.parallel_for
            h.parallel_for<class KernelBlockedFloydWarshall>(
                nd_range<2>(R, R), [=](nd_item<2> item) {
                    // Obtener las coordenadas globales del índice
                    int i = item.get_global_id(0);
                    int j = item.get_global_id(1);
                    
                    // Iterar a través del bloque
                    for (int k = 0; k < block_length; k++) {
                        // Verificar si el camino a través de k es más corto
                        if (accessor_C[i][j] > accessor_A[i][k] + accessor_B[k][j]) {
                            // Actualizar el camino más corto
                            accessor_C[i][j] = accessor_A[i][k] + accessor_B[k][j];
                        }
                        // Sincronizar para garantizar coherencia en el bloque
                        item.barrier(access::fence_space::local_space);
                    }
                });
        }).wait();  // Esperar a que se complete el trabajo en la cola
    }
}

int main() {
    try {
        // Crear una cola SYCL con el selector predeterminado
        queue q{sycl::default_selector_v};
        
        // Crear un vector para representar el grafo y un vector secuencial
        vector<int> grafo(nodos * nodos);
        vector<int> sequential(grafo);

        // Inicializar el grafo.
        InicializarGrafo(grafo);

        // Prepara el JIT (compilador justo a tiempo) con una ejecución inicial
        BlockedFloydWarshall(q, grafo);

        // Medir tiempos de ejecución
        double transcurrido_s = 0;
        double transcurrido_p = 0;

        cout << "Repitiendo el cálculo " << repetitions << " veces para medir el tiempo de ejecución ...\n";

        // Realizar el cálculo y medir tiempos para un número especificado de repeticiones
        for (int i = 0; i < repetitions; i++) {
            cout << "Iteracion: " << (i + 1) << "\n";

            // Cálculo secuencial de todos los caminos más cortos
            copy(grafo.begin(), grafo.end(), sequential.begin());
            auto start_s = chrono::high_resolution_clock::now();
            FloydWarshall(sequential);
            auto end_s = chrono::high_resolution_clock::now();
            transcurrido_s += chrono::duration_cast<chrono::duration<double>>(end_s - start_s).count();

            // Cálculo paralelo de todos los caminos más cortos utilizando SYCL
            copy(grafo.begin(), grafo.end(), sequential.begin());
            auto start_p = chrono::high_resolution_clock::now();
            BlockedFloydWarshall(q, sequential);
            auto end_p = chrono::high_resolution_clock::now();
            transcurrido_p += chrono::duration_cast<chrono::duration<double>>(end_p - start_p).count();

           // Verificar si los resultados de las implementaciones son iguales
            if (!equal(grafo.begin(), grafo.end(), sequential.begin())) {
                cout << "Error al calcular correctamente todos los caminos más cortos entre pares.\n";
                break;
            }
        }
        // Imprimir resultados si los tiempos son mayores a cero
        if (transcurrido_s > 0 && transcurrido_p > 0) {
            cout << "¡Se calcularon exitosamente todos los caminos más cortos entre pares en paralelo!\n";
            transcurrido_s /= repetitions;
            transcurrido_p /= repetitions;
            cout << "Tiempo secuencial: " << transcurrido_s << " seg\n";
            cout << "Tiempo en paralelo: " << transcurrido_p << " seg\n";
            // Imprimir los caminos más cortos calculados
                cout << "Caminos más cortos:\n";
                for (int i = 0; i < nodos; i++) {
                    for (int j = 0; j < nodos; j++) {
                        int cell = i * nodos + j;
                        cout << "De " << i << " a " << j << ": " << grafo[cell] << "\n";
                    }
                }
            }
        }
    catch (const std::exception& e) {
        cout << "Se ha encontrado una excepción al calcular en el dispositivo.\n";
        terminate();
    }

    return 0;
}
