// game_of_life_cuda.cu

#include <cstdio>
#include <cstdlib>
#include <random>
#include <cuda_runtime.h>

using ubyte = unsigned char;

// Kernel: un hilo por celda (map)
__global__ void life_step_kernel(const ubyte* grid, ubyte* next,
                                 int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int worldSize = width * height;

    for (int cell = idx; cell < worldSize; cell += stride) {
        int x = cell % width;
        int y = cell / width;

        // vecinos con wrap-around
        int xm1 = (x + width - 1) % width;
        int xp1 = (x + 1) % width;
        int ym1 = (y + height - 1) % height;
        int yp1 = (y + 1) % height;

        int count = 0;
        // fila superior
        count += grid[ym1*width + xm1];
        count += grid[ym1*width +  x ];
        count += grid[ym1*width + xp1];
        // misma fila
        count += grid[y*width + xm1];
        count += grid[y*width + xp1];
        // fila inferior
        count += grid[yp1*width + xm1];
        count += grid[yp1*width +  x ];
        count += grid[yp1*width + xp1];

        ubyte alive = grid[y*width + x];
        next[y*width + x] = (count == 3 || (count == 2 && alive)) ? 1 : 0;
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "Usage: %s <rows> <cols> <iterations>\n", argv[0]);
        return 1;
    }
    int rows = std::atoi(argv[1]);
    int cols = std::atoi(argv[2]);
    int iters = std::atoi(argv[3]);
    int worldSize = rows * cols;

    // --- Reserva CPU
    ubyte* h_grid     = (ubyte*)std::malloc(worldSize);
    ubyte* h_next     = (ubyte*)std::malloc(worldSize);
    if (!h_grid || !h_next) {
        std::perror("malloc");
        return 1;
    }

    // Inicializa aleatorio
    {
        std::mt19937 gen(123);
        std::uniform_int_distribution<> d(0,1);
        for (int i = 0; i < worldSize; i++)
            h_grid[i] = (ubyte)d(gen);
    }

    // --- Reserva GPU
    ubyte *d_grid, *d_next;
    cudaMalloc(&d_grid, worldSize);
    cudaMalloc(&d_next, worldSize);

    // Copia inicial
    cudaMemcpy(d_grid, h_grid, worldSize, cudaMemcpyHostToDevice);

    // Configura kernel
    int threadsPerBlock = 256;
    int blocks = (worldSize + threadsPerBlock - 1) / threadsPerBlock;

    // Iteraciones
    for (int it = 0; it < iters; ++it) {
        life_step_kernel<<<blocks,threadsPerBlock>>>(d_grid, d_next, cols, rows);
        cudaDeviceSynchronize();
        std::swap(d_grid, d_next);
    }

    // Trae de vuelta
    cudaMemcpy(h_grid, d_grid, worldSize, cudaMemcpyDeviceToHost);

    // Imprime el estado final (opcional)
//    for (int y = 0; y < rows; y++) {
//        for (int x = 0; x < cols; x++) {
//            std::printf("%d", h_grid[y*cols + x]);
//        }
//        std::putchar('\n');
//    }

    // Limpia
    cudaFree(d_grid);
    cudaFree(d_next);
    std::free(h_grid);
    std::free(h_next);

    return 0;
}