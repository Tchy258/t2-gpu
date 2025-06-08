#include "game_of_life_opencl.hpp"
#include <fstream>
#include <random>
#include <stdexcept>


using ubyte = unsigned char;

// OpenCL globals
bool GameOfLifeOpenCL::clInitialized = false;
cl::Context GameOfLifeOpenCL::context_cpp;
cl::CommandQueue GameOfLifeOpenCL::queue_cpp;
cl::Program GameOfLifeOpenCL::program;
cl::Kernel GameOfLifeOpenCL::lifeKernel;


void GameOfLifeOpenCL::initializeOpenCL() {
    if (clInitialized) return;

    // 1) Plataforma y dispositivo
    std::vector<cl::Platform> plats;
    cl::Platform::get(&plats);
    if (plats.empty()) throw std::runtime_error("No OpenCL platforms");
    std::vector<cl::Device> devs;
    plats[0].getDevices(CL_DEVICE_TYPE_GPU|CL_DEVICE_TYPE_CPU, &devs);
    if (devs.empty()) throw std::runtime_error("No OpenCL devices");

    // 2) Contexto y cola
    context_cpp = cl::Context(devs);
    queue_cpp   = cl::CommandQueue(context_cpp, devs[0], CL_QUEUE_PROFILING_ENABLE);

    // 3) Leer el kernel
    #ifdef ARRAY_2D
        std::ifstream sourceFile("kernel1D.cl");
    #else
        std::ifstream sourceFile("kernel2D.cl");
    #endif
    
    if (!sourceFile) throw std::runtime_error("Cannot open kernel file");
    std::string sourceCode( std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
    cl::Program::Sources src{{ sourceCode.c_str(), sourceCode.size() }};
    program = cl::Program(context_cpp, src);

    // 4) Compilar
    program.build(devs);

    // 5) Crear el kernel
    lifeKernel = cl::Kernel(program, "life");

    clInitialized = true;
}

GameOfLifeOpenCL::GameOfLifeOpenCL() {
    initializeOpenCL();

    worldSize = size_t(GRID_ROWS) * GRID_COLS;
    size_t bytes = worldSize * sizeof(ubyte);
    #ifdef ARRAY_2D
        unsigned int finalSizeX = BLOCK_SIZE_X * TILE_WIDTH;
        unsigned int finalSizeY = BLOCK_SIZE_Y * TILE_HEIGHT;
        blocksX = ((GRID_COLS + finalSizeX - 1) / finalSizeX);
        blocksY = ((GRID_ROWS + finalSizeY - 1) / finalSizeY);
    #else
        unsigned int finalSize = blockSize * CELLS_PER_THREAD;
        blocks = (worldSize + finalSize - 1) / finalSize;
    #endif

    // reserva host + device
    grid.host.assign(worldSize, 0);
    nextGrid.host.assign(worldSize, 0);
    grid.dev  = cl::Buffer(context_cpp, CL_MEM_READ_WRITE, bytes);
    nextGrid.dev = cl::Buffer(context_cpp, CL_MEM_READ_WRITE, bytes);
}

void GameOfLifeOpenCL::initialize() {
    std::fill(grid.host.begin(), grid.host.end(), 0);
    std::fill(nextGrid.host.begin(), nextGrid.host.end(), 0);
}

void GameOfLifeOpenCL::initializeRandom() {
    std::mt19937 gen(123);
    std::uniform_int_distribution<> d(0,1);
    for (size_t i = 0; i < worldSize; ++i) {
        grid.host[i] = (ubyte)d(gen);
        nextGrid.host[i] = grid.host[i];
    }
    uploadGrid();
}

void GameOfLifeOpenCL::uploadGrid() {
    size_t bytes = worldSize * sizeof(ubyte);

    // 1) subir grid.host -> grid.dev
    queue_cpp.enqueueWriteBuffer(grid.dev, CL_FALSE, 0, bytes, grid.host.data());
}

void GameOfLifeOpenCL::copyGridToHost() {
    size_t bytes = worldSize * sizeof(ubyte);

    queue_cpp.enqueueReadBuffer(grid.dev, CL_TRUE, 0, bytes, nextGrid.host.data());

    std::swap(grid.host, nextGrid.host);
}

void GameOfLifeOpenCL::step() {

    unsigned int arg = 0;
    lifeKernel.setArg(arg++, grid.dev);
    lifeKernel.setArg(arg++, nextGrid.dev);
    lifeKernel.setArg(arg++, (cl_uint)GRID_COLS);
    lifeKernel.setArg(arg++, (cl_uint)GRID_ROWS);
    // 3) disparar kernel
    #ifdef ARRAY_2D
    lifeKernel.setArg(arg++, TILE_WIDTH);
    lifeKernel.setArg(arg++, TILE_HEIGHT);
    cl::NDRange global(blocksX, blocksY);
    cl::NDRange local(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    #else
    lifeKernel.setArg(arg++, CELLS_PER_THREAD);
    cl::NDRange local(BLOCK_SIZE_X * BLOCK_SIZE_Y);
    cl::NDRange global(blocks);
    #endif
    queue_cpp.enqueueNDRangeKernel(lifeKernel, cl::NullRange, global, local);
    queue_cpp.finish();

    std::swap(grid.dev,  nextGrid.dev);
}

ARRAY_TYPE(unsigned char,) GameOfLifeOpenCL::getGrid() const {
    #ifdef ARRAY_2D
    size_t numRows = GRID_ROWS;
    rowPtrs.clear();
    rowPtrs.reserve(numRows);

    for (size_t i = 0; i < numRows; ++i) {
        rowPtrs.push_back(const_cast<ubyte*>(&grid.host[i * GRID_COLS]));
    }

    return rowPtrs.data();
    #else
    return const_cast<ubyte*>(grid.host.data());
    #endif
}
