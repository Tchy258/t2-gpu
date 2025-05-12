#include "game_of_life_parallel.hpp"
#include "benchmark.h"

int main(int argc, char** argv) {
    std::cout << "Test del juego de la vida con implementación paralela en CPU\n";
    GameOfLife* game = new GameOfLifeParallel();
    int iterations = atoi(argv[1]);
    benchmark(game, iterations, argc > 2 ? argv[2] : "");
    delete game;
    return 0;
}