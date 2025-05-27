#ifndef GAME_OF_LIFE_OPENCL_HPP
#define GAME_OF_LIFE_OPENCL_HPP

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
    //std::vector<ubyte> getGrid() const override;

private:
    static void initializeOpenCL();

    static bool clInitialized;
    static cl::Context      context_cpp;
    static cl::CommandQueue queue_cpp;
    static cl::Program      program;
    static cl::Kernel       lifeKernel;

    CLBuffer grid, nextGrid;
    size_t   worldSize;
};

#endif // GAME_OF_LIFE_OPENCL_HPP
