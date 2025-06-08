#ifndef GAME_OF_LIFE_OPENCL_HPP
#define GAME_OF_LIFE_OPENCL_HPP
#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_TARGET_OPENCL_VERSION 300
#include "game_of_life.hpp"
#include <vector>
#include <CL/opencl.hpp>


using ubyte = unsigned char;

struct CLBuffer {
    cl::Buffer dev;
    std::vector<ubyte> host;
};

class GameOfLifeOpenCL : public GameOfLife {
public:
    GameOfLifeOpenCL();
    ~GameOfLifeOpenCL() override = default;

    void initialize() override;
    void initializeRandom() override;
    void step() override;
    unsigned char countAliveCells(unsigned int x0, unsigned int x1, unsigned int x2,
                                        unsigned int y0, unsigned int y1, unsigned int y2) final {
                                            throw std::logic_error("Counting should be done in kernel");
                                        };
    ARRAY_TYPE(unsigned char,) getGrid() const override;
    void copyGridToHost() override;
private:
    static void initializeOpenCL();
    static bool clInitialized;
    static cl::Context      context_cpp;
    static cl::CommandQueue queue_cpp;
    static cl::Program      program;
    static cl::Kernel       lifeKernel;

    void uploadGrid();
    CLBuffer grid, nextGrid;
    size_t   worldSize;
    #ifdef ARRAY_2D
    mutable std::vector<unsigned char*> rowPtrs;
    size_t blocksX;
    size_t blocksY;
    #else
    size_t blocks;
    size_t blockSize = BLOCK_SIZE_X * BLOCK_SIZE_Y;
    #endif
};

#endif // GAME_OF_LIFE_OPENCL_HPP
