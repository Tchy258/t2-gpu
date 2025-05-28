// GameOfLifeCUDA.cu

#include "game_of_life_cuda.hpp"
#include <cuda_runtime.h>
#include <random>
#include <cstdio>
#include <cstdlib>

#include "constants.h"  // define GRID_ROWS, GRID_COLS

__global__ void life_step_kernel(const ubyte* grid, ubyte* next,
                                 int width, int height)
{
    int idx    = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    int worldSize = width * height;

    for (int cell = idx; cell < worldSize; cell += stride) {
        int x = cell % width, y = cell / width;
        int xm1 = (x + width  - 1) % width;
        int xp1 = (x + 1) % width;
        int ym1 = (y + height - 1) % height;
        int yp1 = (y + 1) % height;

        int count = 0;
        count += grid[ym1*width + xm1];
        count += grid[ym1*width +  x ];
        count += grid[ym1*width + xp1];
        count += grid[y*width + xm1];
        count += grid[y*width + xp1];
        count += grid[yp1*width + xm1];
        count += grid[yp1*width +  x ];
        count += grid[yp1*width + xp1];

        ubyte alive = grid[y*width + x];
        next[y*width + x] = (count == 3 || (count == 2 && alive)) ? 1 : 0;
    }
}

// utilidad para chequear errores
static void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA Error (%s): %s\n", msg, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

// Constructor por defecto
GameOfLifeCUDA::GameOfLifeCUDA()
 : rows(GRID_ROWS), cols(GRID_COLS),
   worldSize(size_t(rows) * cols),
   bytes(worldSize * sizeof(ubyte)),
   h_grid(worldSize), h_next(worldSize)
{
    blocks = (worldSize + threadsPerBlock - 1) / threadsPerBlock;
    allocDevice();
}

GameOfLifeCUDA::~GameOfLifeCUDA() {
    freeDevice();
}

void GameOfLifeCUDA::allocDevice() {
    checkCuda(cudaMalloc(&d_grid, bytes), "malloc d_grid");
    checkCuda(cudaMalloc(&d_next, bytes), "malloc d_next");
}

void GameOfLifeCUDA::freeDevice() {
    if (d_grid)  cudaFree(d_grid);
    if (d_next)  cudaFree(d_next);
}

void GameOfLifeCUDA::initialize() {
    // host a cero
    std::fill(h_grid.begin(), h_grid.end(), 0);
    std::fill(h_next.begin(), h_next.end(), 0);
    // subir al device
    checkCuda(cudaMemcpy(d_grid, h_grid.data(), bytes, cudaMemcpyHostToDevice),
              "memcpy init to d_grid");
}

void GameOfLifeCUDA::initializeRandom() {
    std::mt19937 gen(123);
    std::uniform_int_distribution<> d(0,1);
    for (size_t i = 0; i < worldSize; ++i)
        h_grid[i] = (ubyte)d(gen);
    // subir al device y duplicar en el buffer “next”
    checkCuda(cudaMemcpy(d_grid, h_grid.data(), bytes, cudaMemcpyHostToDevice),
              "memcpy rand to d_grid");
    checkCuda(cudaMemcpy(d_next, h_grid.data(), bytes, cudaMemcpyHostToDevice),
              "memcpy rand to d_next");
}

void GameOfLifeCUDA::step() {
    // lanzar kernel y esperar
    life_step_kernel<<<blocks, threadsPerBlock>>>(d_grid, d_next, cols, rows);
    checkCuda(cudaGetLastError(), "launch life_step_kernel");
    checkCuda(cudaDeviceSynchronize(), "sync after kernel");
    // swap de buffers
    std::swap(d_grid, d_next);
}

// std::vector<ubyte> GameOfLifeCUDA::getGrid() const {
//     // leer estado actual
//     std::vector<ubyte> copy(worldSize);
//     checkCuda(cudaMemcpy(copy.data(), d_grid, bytes, cudaMemcpyDeviceToHost),
//               "memcpy d_grid->host");
//     return copy;
// }
