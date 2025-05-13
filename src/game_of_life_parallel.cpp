#include "game_of_life_parallel.hpp"

typedef unsigned char ubyte;

GameOfLifeParallel::GameOfLifeParallel() {
    ARRAY_ALLOC(ubyte, grid);
    ARRAY_ALLOC(ubyte, nextGrid);
}

GameOfLifeParallel::~GameOfLifeParallel() {
    ARRAY_DELETE(grid);
    ARRAY_DELETE(nextGrid);
}

inline ubyte GameOfLifeParallel::countAliveCells(unsigned int x0, unsigned int x1, unsigned int x2,
                                                unsigned int y0, unsigned int y1, unsigned int y2) 
{
    return ARRAY_ACCESS(grid,y0,x0) + ARRAY_ACCESS(grid,y0,x1) + ARRAY_ACCESS(grid,y0,x2)
        + ARRAY_ACCESS(grid,y1,x0) + ARRAY_ACCESS(grid,y1,x2)
        + ARRAY_ACCESS(grid,y2,x0) + ARRAY_ACCESS(grid,y2,x1) + ARRAY_ACCESS(grid,y2,x2);
}

void GameOfLifeParallel::initialize() {
    for (int i = 0; i < GRID_ROWS; i++) {
        for (int j = 0; j < GRID_COLS; j++) {
            ARRAY_ACCESS(grid, i, j) = 0;
            ARRAY_ACCESS(nextGrid, i, j) = 0;
        }
    }
}

void GameOfLifeParallel::initializeRandom() {
    int seed = 123;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0.0,1.0);
    for (int i = 0; i < GRID_ROWS; i++) {
        for (int j = 0; j < GRID_COLS; j++) {
            double coinFlip = dis(gen);
            ubyte val = coinFlip < 0.5 ? 1 : 0;
            ARRAY_ACCESS(grid, i, j) = val;
            ARRAY_ACCESS(nextGrid, i, j) = val;
        }
    }
}

void GameOfLifeParallel::setCellInNextGrid(unsigned int index) {
    unsigned int x1 = index % GRID_COLS;
    unsigned int y1 = index / GRID_COLS;
    unsigned int y0 = (y1 + GRID_ROWS - 1) % GRID_ROWS;
    unsigned int y2 = (y1 + 1) % GRID_ROWS;
    unsigned int x0 = (x1 + GRID_COLS - 1) % GRID_COLS;
    unsigned int x2 = (x1 + 1) % GRID_COLS;
    ubyte aliveCells = countAliveCells(x0,x1,x2,y0,y1,y2);
    ARRAY_ACCESS(nextGrid, y1, x1) = aliveCells == 3 || (aliveCells == 2 && ARRAY_ACCESS(grid,y1,x1)) ? 1 : 0;
}


void GameOfLifeParallel::step() {
    unsigned int chunkSize = totalSize / NUM_THREADS;
    std::thread threads[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        unsigned int startIndex = i * chunkSize;
        unsigned int endIndex = (i == NUM_THREADS - 1) ? totalSize : (i + 1) * chunkSize;
        threads[i] = std::thread([startIndex, endIndex, this]() {
            for (unsigned int index = startIndex; index < endIndex; index++) {
                setCellInNextGrid(index);
            }
        });
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }

    std::swap(grid, nextGrid);
}


ARRAY_TYPE(unsigned char,) GameOfLifeParallel::getGrid() const {
    return grid;
}

std::ostream &operator<<(std::ostream &os, const GameOfLifeParallel &game) {
    for (int y = 0; y < GRID_ROWS; y++) {
        os << "[ ";
        for (int x = 0; x < GRID_COLS; x++) {
            int value = (int) ARRAY_ACCESS(game.getGrid(),y,x);
            os << value << (x != GRID_COLS - 1 ? ", " : "");
        }
        os << "]" << std::endl;
    }
    return os;
}