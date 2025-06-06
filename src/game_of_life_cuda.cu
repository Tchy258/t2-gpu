#include "game_of_life_cuda.hpp"
#include <cuda_runtime.h>
#ifdef ARRAY_2D
    #include "cuda_kernel_2d.cu"
#else
    #include "cuda_kernel_1d.cu"
#endif


static void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA Error (%s): %s\n", msg, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

GameOfLifeCUDA::GameOfLifeCUDA()
    : rows(GRID_ROWS), cols(GRID_COLS),
      worldSize(size_t(rows) * cols),
      bytes(worldSize * sizeof(ubyte)),
      h_grid(worldSize), h_next(worldSize)
{
    #ifdef ARRAY_2D
    blocksX = (cols + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
    blocksY = (rows + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;
    #else
    blocks = (worldSize + blockSize - 1) / blockSize;
    #endif
    allocDevice();
}

GameOfLifeCUDA::~GameOfLifeCUDA() {
    freeDevice();
}

void GameOfLifeCUDA::allocDevice() {
#ifdef ARRAY_2D
    d_grid_rows.resize(rows);
    d_next_rows.resize(rows);

    for (int y = 0; y < rows; ++y) {
        checkCuda(cudaMalloc(&d_grid_rows[y], cols * sizeof(ubyte)), "malloc grid row");
        checkCuda(cudaMalloc(&d_next_rows[y], cols * sizeof(ubyte)), "malloc next row");
    }

    checkCuda(cudaMalloc(&d_grid, rows * sizeof(ubyte*)), "malloc d_grid**");
    checkCuda(cudaMalloc(&d_next, rows * sizeof(ubyte*)), "malloc d_next**");

    checkCuda(cudaMemcpy(d_grid, d_grid_rows.data(), rows * sizeof(ubyte*), cudaMemcpyHostToDevice), "copy grid row ptrs");
    checkCuda(cudaMemcpy(d_next, d_next_rows.data(), rows * sizeof(ubyte*), cudaMemcpyHostToDevice), "copy next row ptrs");
#else
    checkCuda(cudaMalloc(&d_grid, bytes), "malloc d_grid");
    checkCuda(cudaMalloc(&d_next, bytes), "malloc d_next");
#endif
}

void GameOfLifeCUDA::freeDevice() {
#ifdef ARRAY_2D
    for (int y = 0; y < rows; ++y) {
        cudaFree(d_grid_rows[y]);
        cudaFree(d_next_rows[y]);
    }
    cudaFree(d_grid);
    cudaFree(d_next);
#else
    if (d_grid) cudaFree(d_grid);
    if (d_next) cudaFree(d_next);
#endif
}

void GameOfLifeCUDA::initialize() {
    std::fill(h_grid.begin(), h_grid.end(), 0);
    std::fill(h_next.begin(), h_next.end(), 0);
    uploadGrid();
}

void GameOfLifeCUDA::initializeRandom() {
    std::mt19937 gen(123);
    std::uniform_int_distribution<> d(0, 1);
    for (size_t i = 0; i < worldSize; ++i)
        h_grid[i] = (ubyte)d(gen);
    uploadGrid();
}

void GameOfLifeCUDA::uploadGrid() {
#ifdef ARRAY_2D
    for (int y = 0; y < rows; ++y) {
        checkCuda(cudaMemcpy(d_grid_rows[y], &h_grid[y * cols], cols, cudaMemcpyHostToDevice), "copy row grid");
        checkCuda(cudaMemcpy(d_next_rows[y], &h_grid[y * cols], cols, cudaMemcpyHostToDevice), "copy row next");
    }
#else
    checkCuda(cudaMemcpy(d_grid, h_grid.data(), bytes, cudaMemcpyHostToDevice), "copy grid");
    checkCuda(cudaMemcpy(d_next, h_grid.data(), bytes, cudaMemcpyHostToDevice), "copy next");
#endif
}

void GameOfLifeCUDA::step() {
#ifdef ARRAY_2D
    dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 blocks(blocksX, blocksY);
    life_step_kernel2d<<<blocks, threads>>>(d_grid, d_next, cols, rows);
#else
    life_step_kernel1d<<<blocks, blockSize>>>(d_grid, d_next, cols, rows);
#endif
    checkCuda(cudaGetLastError(), "launch kernel");
    checkCuda(cudaDeviceSynchronize(), "sync");

#ifdef ARRAY_2D
    std::swap(d_grid_rows, d_next_rows);
    checkCuda(cudaMemcpy(d_grid, d_grid_rows.data(), rows * sizeof(ubyte*), cudaMemcpyHostToDevice), "swap grid**");
    checkCuda(cudaMemcpy(d_next, d_next_rows.data(), rows * sizeof(ubyte*), cudaMemcpyHostToDevice), "swap next**");
#else
    std::swap(d_grid, d_next);
#endif
    uploadGrid();
}
