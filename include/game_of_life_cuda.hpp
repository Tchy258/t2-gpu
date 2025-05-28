#ifndef GAME_OF_LIFE_CUDA_HPP
#define GAME_OF_LIFE_CUDA_HPP

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

    std::vector<ubyte> getGrid() const;  // como en OpenCL: devuelve copia

private:
    int rows, cols;
    size_t worldSize, bytes;

    std::vector<ubyte> h_grid, h_next;
    ubyte *d_grid = nullptr, *d_next = nullptr;

    int threadsPerBlock = 256;
    int blocks;

    void allocDevice();
    void freeDevice();
};

#endif
