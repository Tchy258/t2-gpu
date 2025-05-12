#include "game_of_life_serial.hpp"

typedef unsigned char ubyte;

GameOfLifeSerial::GameOfLifeSerial() {
    ARRAY_ALLOC(ubyte, grid);
    ARRAY_ALLOC(ubyte, nextGrid);
}

GameOfLifeSerial::~GameOfLifeSerial() {
    ARRAY_DELETE(grid);
    ARRAY_DELETE(nextGrid);
}

inline ubyte GameOfLifeSerial::countAliveCells(unsigned int x0, unsigned int x1, unsigned int x2,
                                                unsigned int y0, unsigned int y1, unsigned int y2) 
{
    return ARRAY_ACCESS(grid,y0,x0) + ARRAY_ACCESS(grid,y0,x1) + ARRAY_ACCESS(grid,y0,x2)
        + ARRAY_ACCESS(grid,y1,x0) + ARRAY_ACCESS(grid,y1,x2)
        + ARRAY_ACCESS(grid,y2,x0) + ARRAY_ACCESS(grid,y2,x1) + ARRAY_ACCESS(grid,y2,x2);
}

void GameOfLifeSerial::initialize() {
    for (int i = 0; i < GRID_ROWS; i++) {
        for (int j = 0; j < GRID_COLS; j++) {
            ARRAY_ACCESS(grid, i, j) = 0;
            ARRAY_ACCESS(nextGrid, i, j) = 0;
        }
    }
}

void GameOfLifeSerial::initializeRandom() {
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


void GameOfLifeSerial::step() {
    for (int y = 0; y < GRID_ROWS; ++y) {
        unsigned int y0 = ((y + GRID_ROWS - 1) % GRID_ROWS);
        unsigned int y1 = y;
        unsigned int y2 = ((y + 1) % GRID_ROWS);
        for (int x = 0; x < GRID_COLS; ++x) {
            unsigned int x0 = (x + GRID_COLS - 1) % GRID_COLS;
            unsigned int x2 = (x + 1) % GRID_COLS;

            ubyte aliveCells = countAliveCells(x0,x,x2,y0,y1,y2);
            ARRAY_ACCESS(nextGrid, y, x) = aliveCells == 3 || (aliveCells == 2 && ARRAY_ACCESS(grid,y,x)) ? 1 : 0;
        }
    }

    std::swap(grid, nextGrid);
}


const unsigned char* GameOfLifeSerial::getGrid() const {
    return grid;
}

std::ostream &operator<<(std::ostream &os, const GameOfLifeSerial &game) {
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