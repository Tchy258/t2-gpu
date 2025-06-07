#include "benchmark.h"

void benchmark(GameOfLife* game, int iterations, const std::string& fileName) {
    #ifdef ARRAY_2D
    std::cout << "2D" << std::endl;
    #else
    std::cout << "1D" << std::endl;
    #endif
    std::ofstream file;
    if (!fileName.empty()) {
        file = std::ofstream(fileName);
        file << "iterationNumber,iterationDuration(s),timeToCopy(s),timeProcessingOnly(s),cellsPerSecond,cellsPerSecondNoCopy" << std::endl;
    }
    auto start = std::chrono::high_resolution_clock::now();
    game->initializeRandom();
    int totalIterations = iterations;
    unsigned long long totalCellsEvaluated = GRID_COLS * GRID_ROWS * totalIterations;
    std::chrono::duration<double, std::milli> iterationTotal{0};
    std::chrono::duration<double, std::milli> copyTimeTotal{0};
    while (iterations-- > 0) {
        auto iterStart = std::chrono::high_resolution_clock::now();
        game->step();
        auto iterEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> iterationDuration = iterEnd - iterStart;
        auto copyStart = std::chrono::high_resolution_clock::now();
        game->copyGridToHost();
        auto copyEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> copyDuration = copyEnd - copyStart;
        copyTimeTotal += copyDuration;
        if (!fileName.empty() && file.is_open()) {
            file << totalIterations - iterations << "," 
            << iterationDuration.count() / 1000.0 << ","
            << copyDuration.count() << ","
            << (iterationDuration - copyDuration).count() / 1000.0 << ","
            << (GRID_COLS * GRID_ROWS) / (iterationDuration.count() / 1000.0) << ","
            << (GRID_COLS * GRID_ROWS) / ((iterationDuration - copyDuration).count() / 1000.0) << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    double seconds = iterationTotal.count() / 1000.0;
    double secondsCopy = copyTimeTotal.count() / 1000.0;
    double cellsPerSecond = totalCellsEvaluated / seconds;
    double cellsNoCopy = totalCellsEvaluated / secondsCopy;

    std::cout << "Total elapsed time: " << duration.count() << " ms\n";
    std::cout << "Time in step(): " << iterationTotal.count() << " ms\n";
    std::cout << "Time spent copying to host: " << copyTimeTotal.count() << "ms\n";
    std::cout << "Evaluated cells per second: " << cellsPerSecond << "\n";
    std::cout << "Evaluated cells with no copying per second: " << cellsNoCopy << "\n";
    if (!fileName.empty() && file.is_open()) {
        file.close();
    }
}
