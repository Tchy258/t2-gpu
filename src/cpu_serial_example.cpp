#include "game_of_life_serial.hpp"
#include<iostream>
#include<thread>
#include<chrono>

int main(int argc, char** argv){
    std::cout << "Test del juego de la vida con implementación serial en CPU\n";
    GameOfLifeSerial game = GameOfLifeSerial();
    game.initializeRandom();
    int iterations = atoi(argv[1]);
    while (iterations-- > 0) {
        std::cout << game;
        std::this_thread::sleep_for(std::chrono::seconds(3));
        std::cout << std::endl;
        game.step();
    }
}