#include "benchmark.h"

void benchmark(GameOfLife* game, int iterations, const std::string& fileName) {
    std::ofstream file;
    if (!fileName.empty()) {
        file = std::ofstream(fileName);
        file << "iterationNumber,iterationDuration(s),cellsPerSecond" << std::endl;
    }
    game->initializeRandom();
    int totalIterations = iterations;
    unsigned long long totalCellsEvaluated = GRID_COLS * GRID_ROWS * totalIterations;
    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> iterationTotal{0};

    while (iterations-- > 0) {
        auto iterStart = std::chrono::high_resolution_clock::now();
        game->step();
        auto iterEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> iterationDuration = iterEnd - iterStart;
        iterationTotal += iterationDuration;
        if (!fileName.empty() && file.is_open()) {
            file << totalIterations - iterations << "," << iterationDuration.count() / 1000.0 << "," << (GRID_COLS * GRID_ROWS) / (iterationDuration.count() / 1000.0) << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    double seconds = iterationTotal.count() / 1000.0;
    double cellsPerSecond = totalCellsEvaluated / seconds;

    std::cout << "Elapsed time: " << duration.count() << " ms\n";
    std::cout << "Time in step(): " << iterationTotal.count() << " ms\n";
    std::cout << "Evaluated cells per second: " << cellsPerSecond << "\n";
    if (!fileName.empty() && file.is_open()) {
        file.close();
    }
}
