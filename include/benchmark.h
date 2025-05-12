#ifndef BENCHMARK_H
#define BENCHMARK_H
#include "game_of_life.hpp"
#include<iostream>
#include<thread>
#include<chrono>
#include <fstream>
#include <string>

void benchmark(GameOfLife* game, int iterations, const std::string& fileName = "");

#endif