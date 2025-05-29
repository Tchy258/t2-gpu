#ifndef GAME_OF_LIFE_CUDA_H
#define GAME_OF_LIFE_CUDA_H

#include "game_of_life.hpp"
#include "constants.h"            // define GRID_ROWS, GRID_COLS
#include <vector>
using ubyte = unsigned char;

class GameOfLifeCUDA : public GameOfLife {
public:
    GameOfLifeCUDA();            // constructor por defecto
    ~GameOfLifeCUDA();

    void initialize() override;
    void initializeRandom() override;
    void step() override;

    unsigned char countAliveCells(unsigned int x0, unsigned int x1, unsigned int x2, 
                                            unsigned int y0, unsigned int y1, unsigned int y2) {
                                                throw std::logic_error("Counting should be done inside the kernel");
                                            }
    ARRAY_TYPE(unsigned char,) getGrid() const override {
        throw std::logic_error("Not implemented for CUDA");
    }

private:
    int rows, cols;
    size_t worldSize, bytes;

    std::vector<ubyte> h_grid, h_next;
    ubyte *d_grid = nullptr, *d_next = nullptr;

    int threadsPerBlock = BLOCK_SIZE_X*BLOCK_SIZE_Y;
    int blocks;

    void allocDevice();
    void freeDevice(); 
};

#endif
