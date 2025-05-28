#include "game_of_life_opencl.hpp"
#include "benchmark.h"


int main(int argc, char** argv) {
    std::cout << "Test del juego de la vida con OpenCL" << std::endl;
    //if (argc < 2) {
    //    std::cerr << "Uso: " << argv[0] << " <iterations> [output.csv]" << std::endl;
    //    return 1;
    //}
    GameOfLife* game = new GameOfLifeOpenCL();
    //int iterations = std::atoi(argv[1]);
    int iterations = 3;
    benchmark(game, iterations, argc > 2 ? argv[2] : std::string());
    delete game;
    return 0;
}
