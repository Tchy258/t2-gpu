#ifndef GAME_OF_LIFE_H
#define GAME_OF_LIFE_H

#include <cstddef>
#include "array_macros.h"
#include <random>

class GameOfLife {
public:
    virtual ~GameOfLife() = default;

    virtual void initialize() = 0;
    virtual void initializeRandom() = 0;
    // virtual unsigned char countAliveCells(unsigned int x0, unsigned int x1, unsigned int x2, 
    //                                         unsigned int y0, unsigned int y1, unsigned int y2) = 0;
    virtual void step() = 0;
    //virtual ARRAY_TYPE(unsigned char,) getGrid() const = 0;

};

#endif