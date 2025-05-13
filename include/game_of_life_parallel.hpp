#ifndef GAME_OF_LIFE_PARALLEL_H
#define GAME_OF_LIFE_PARALLEL_H

#include "game_of_life.hpp"
#include <cstring>
#include <utility>
#include <iostream>
#include <thread>

#define NUM_THREADS (12)

class GameOfLifeParallel : public GameOfLife {
private:
    ARRAY_TYPE(unsigned char, grid);
    ARRAY_TYPE(unsigned char, nextGrid);
    void setCellInNextGrid(unsigned int index);
    unsigned long long totalSize = GRID_COLS * GRID_ROWS;

public:
    GameOfLifeParallel();
    ~GameOfLifeParallel();

    void initialize() override;
    void initializeRandom() override;
    void step() override;
    inline unsigned char countAliveCells(unsigned int x0, unsigned int x1, unsigned int x2,
                                        unsigned int y0, unsigned int y1, unsigned int y2) final;
    ARRAY_TYPE(unsigned char,) getGrid() const override;
    friend std::ostream &operator<<(std::ostream &os, const GameOfLifeParallel &game);
};


#endif