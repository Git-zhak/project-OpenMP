#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <omp.h> // Librería principal de OpenMP

// Resolución 8K UHD (7680 x 4320)
const int WIDTH = 7680;
const int HEIGHT = 4320;
const int MAX_ITER = 500;

// Estructura básica para manejar canales de color RGB
struct Pixel {
    unsigned char r, g, b;
};

// Tarea A: Generar el Conjunto de Mandelbrot
void generarMandelbrot(std::vector<Pixel>& imagen) {
    // Usamos dynamic con chunk de 64 según el perfilado empírico
    #pragma omp parallel for schedule(dynamic, 64)
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            double cr = -2.0 + (x * 3.0 / WIDTH);
            double ci = -1.2 + (y * 2.4 / HEIGHT);

            double zr = 0.0, zi = 0.0;
            int iter = 0;

            while (zr * zr + zi * zi <= 4.0 && iter < MAX_ITER) {
                double temp = zr * zr - zi * zi + cr;
                zi = 2.0 * zr * zi + ci;
                zr = temp;
                ++iter;
            }

            int idx = y * WIDTH + x;
            if (iter == MAX_ITER) {
                imagen[idx] = {0, 0, 0};
            } else {
                unsigned char c = static_cast<unsigned char>(255 * iter / MAX_ITER);
                imagen[idx] = {static_cast<unsigned char>(c * 3), static_cast<unsigned char>(c * 5), static_cast<unsigned char>(c * 8)};
            }
        }
    }
}

// Tarea B: Aplicar un filtro de convolución 2D (Desenfoque Gaussiano 5x5 pesado)
void aplicarFiltroGaussiano(const std::vector<Pixel>& origen, std::vector<Pixel>& destino) {
    const int K_SIZE = 5;
    const double kernel[5][5] = {
        {1/273.0,  4/273.0,  7/273.0,  4/273.0, 1/273.0},
        {4/273.0, 16/273.0, 26/273.0, 16/273.0, 4/273.0},
        {7/273.0, 26/273.0, 41/273.0, 26/273.0, 7/273.0},
        {4/273.0, 16/273.0, 26/273.0, 16/273.0, 4/273.0},
        {1/273.0,  4/273.0,  7/273.0,  4/273.0, 1/273.0}
    };

    int r_offset = K_SIZE / 2;

    // Usamos schedule(static) porque el filtro aplica exactamente la misma 
    // cantidad de operaciones matemáticas (25) a cada píxel. 
    // 'static' divide el trabajo en bloques iguales de antemano, reduciendo el overhead.
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            double red_sum = 0.0, green_sum = 0.0, blue_sum = 0.0;

            for (int ky = 0; ky < K_SIZE; ++ky) {
                for (int kx = 0; kx < K_SIZE; ++kx) {
                    int px = std::min(std::max(x + (kx - r_offset), 0), WIDTH - 1);
                    int py = std::min(std::max(y + (ky - r_offset), 0), HEIGHT - 1);

                    int pixel_idx = py * WIDTH + px;
                    double weight = kernel[ky][kx];

                    red_sum   += origen[pixel_idx].r * weight;
                    green_sum += origen[pixel_idx].g * weight;
                    blue_sum  += origen[pixel_idx].b * weight;
                }
            }

            int dest_idx = y * WIDTH + x;
            destino[dest_idx].r = static_cast<unsigned char>(red_sum);
            destino[dest_idx].g = static_cast<unsigned char>(green_sum);
            destino[dest_idx].b = static_cast<unsigned char>(blue_sum);
        }
    }
}

// Función auxiliar para guardar la imagen en formato PPM
void guardarImagenPPM(const std::string& nombreArchivo, const std::vector<Pixel>& imagen) {
    std::ofstream archivo(nombreArchivo, std::ios::out | std::ios::binary);
    if (!archivo) {
        std::cerr << "Error al abrir el archivo para escribir." << std::endl;
        return;
    }
    archivo << "P6\n" << WIDTH << " " << HEIGHT << "\n255\n";
    archivo.write(reinterpret_cast<const char*>(imagen.data()), imagen.size() * sizeof(Pixel));
    archivo.close();
}

// HISTOGRAMA
// sincronización con mutual exclusion (Atomic)
void calcularHistogramaAtomic(const std::vector<Pixel>& imagen, std::vector<int>& histograma) {
    std::fill(histograma.begin(), histograma.end(), 0);

    #pragma omp parallel for
    for (size_t i = 0; i < imagen.size(); ++i) {
        // Calcular promedio para escala de grises (0-255)
        int gris = (imagen[i].r + imagen[i].g + imagen[i].b) / 3;
        
        // Para exclusión mutua
        #pragma omp atomic
        histograma[gris]++;
    }
}

// Variables locales y Reduction
void calcularHistogramaReduction(const std::vector<Pixel>& imagen, std::vector<int>& histograma) {
    std::fill(histograma.begin(), histograma.end(), 0);
    
    // para facilitar la sintaxis de reduction en OpenMP
    int hist_local[256] = {0};

    // Esto crea copias privadas del arreglo para cada hilo de forma automática
    #pragma omp parallel for reduction(+:hist_local[:256])
    for (size_t i = 0; i < imagen.size(); ++i) {
        int gris = (imagen[i].r + imagen[i].g + imagen[i].b) / 3;
        hist_local[gris]++; // Cero contención, cero bloqueos
    }

    // Copiamos el resultado al vector final
    for (int i = 0; i < 256; ++i) {
        histograma[i] = hist_local[i];
    }
}

int main() {
    // omp_get_max_threads() nos dice cuántos hilos están disponibles en el sistema
    std::cout << "Iniciando procesamiento paralelo con " << omp_get_max_threads() << " hilos (Resolucion 8K)..." << std::endl;

    std::vector<Pixel> imagenOriginal(WIDTH * HEIGHT);
    std::vector<Pixel> imagenFiltrada(WIDTH * HEIGHT);

    // Ejecución Tarea A
    double startA = omp_get_wtime();
    generarMandelbrot(imagenOriginal);
    double endA = omp_get_wtime();
    std::cout << "Tarea A (Mandelbrot) completada en: " << (endA - startA) << " segundos." << std::endl;

    // Ejecución Tarea B
    double startB = omp_get_wtime();
    aplicarFiltroGaussiano(imagenOriginal, imagenFiltrada);
    double endB = omp_get_wtime();
    std::cout << "Tarea B (Filtro Gaussiano) completada en: " << (endB - startB) << " segundos." << std::endl;

    std::cout << "Tiempo total de computo: " << ((endA - startA) + (endB - startB)) << " segundos." << std::endl;

    std::cout << "\n---Iniciando Histograma" << std::endl;
    std::vector<int> histAtomic(256);
    std::vector<int> histReduction(256);

    // Prueba con Atomic
    double startC1 = omp_get_wtime();
    calcularHistogramaAtomic(imagenFiltrada, histAtomic);
    double endC1 = omp_get_wtime();
    std::cout << "Histograma (Atomic/Critical) completado en: " << (endC1 - startC1) << " segundos." << std::endl;

    // Prueba con Reduction
    double startC2 = omp_get_wtime();
    calcularHistogramaReduction(imagenFiltrada, histReduction);
    double endC2 = omp_get_wtime();
    std::cout << "Histograma (Reduction/Local) completado en: " << (endC2 - startC2) << " segundos." << std::endl;

    std::cout << "Guardando imagenes en disco..." << std::endl;
    guardarImagenPPM("mandelbrot_8k_omp.ppm", imagenOriginal);
    guardarImagenPPM("mandelbrot_8k_omp_blurred.ppm", imagenFiltrada);
    std::cout << "¡Proceso finalizado con exito!" << std::endl;

    return 0;
}