#ifndef GAME_OF_LIFE_CUDA_H
#define GAME_OF_LIFE_CUDA_H

#include "game_of_life.hpp"
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
using ubyte = unsigned char;

class GameOfLifeCUDA : public GameOfLife {
public:
    GameOfLifeCUDA();
    ~GameOfLifeCUDA();

    void initialize();
    void initializeRandom();
    void step();
    unsigned char countAliveCells(unsigned int x0, unsigned int x1, unsigned int x2, 
                                            unsigned int y0, unsigned int y1, unsigned int y2) {
                                                throw std::logic_error("Counting should be done inside the kernel");
                                            }
    ARRAY_TYPE(unsigned char,) getGrid() const override {
        throw std::logic_error("Not implemented for CUDA");
    }

private:
    void allocDevice();
    void freeDevice();
    void uploadGrid();

    unsigned long long rows, cols;
    size_t worldSize;
    size_t bytes;

    std::vector<ubyte> h_grid;
    std::vector<ubyte> h_next;

#ifdef ARRAY_2D
    ubyte** d_grid = nullptr;
    ubyte** d_next = nullptr;
    std::vector<ubyte*> d_grid_rows;
    std::vector<ubyte*> d_next_rows;
    unsigned long long blocksX;
    unsigned long long blocksY;
#else
    unsigned long long blocks;
    unsigned long long blockSize = BLOCK_SIZE_X * BLOCK_SIZE_Y;
    ubyte* d_grid = nullptr;
    ubyte* d_next = nullptr;
#endif
};

#endif
