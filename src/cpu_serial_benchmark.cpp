#include "game_of_life_serial.hpp"
#include "benchmark.h"

int main(int argc, char** argv) {
    std::cout << "Test del juego de la vida con implementación serial en CPU\n";
    GameOfLife* game = new GameOfLifeSerial();
    int iterations = atoi(argv[1]);
    benchmark(game, iterations, argc > 2 ? argv[2] : "");
    delete game;
    return 0;
}