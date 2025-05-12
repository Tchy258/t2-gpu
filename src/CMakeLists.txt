if (USE_OPENCL)

add_executable(opencl_test opencl_test.cpp)
target_compile_features(opencl_test PRIVATE cxx_auto_type)

target_include_directories(opencl_test PRIVATE "${CMAKE_SOURCE_DIR}/extern/OpenCL-Headers")
target_link_libraries(opencl_test OpenCL::OpenCL)

endif()

if (USE_CPU)
    add_compile_definitions(USE_CPU)

    add_library(Benchmark STATIC benchmark.cpp)
    target_include_directories(Benchmark PRIVATE "${PROJECT_SOURCE_DIR}/include")

    add_library(CPUSerialImpl STATIC game_of_life_serial.cpp)
    target_include_directories(CPUSerialImpl PRIVATE "${PROJECT_SOURCE_DIR}/include")
    add_executable(GameOfLifeCPUSerial cpu_serial_example.cpp)
    target_include_directories(GameOfLifeCPUSerial PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(GameOfLifeCPUSerial CPUSerialImpl)

    add_library(CPUParallelImpl STATIC game_of_life_parallel.cpp)
    target_include_directories(CPUParallelImpl PRIVATE "${PROJECT_SOURCE_DIR}/include")
    add_executable(GameOfLifeCPUParallel cpu_parallel_example.cpp)
    target_include_directories(GameOfLifeCPUParallel PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(GameOfLifeCPUParallel CPUParallelImpl)

    add_executable(CPUSerialBenchmark cpu_serial_benchmark.cpp)
    target_include_directories(CPUSerialBenchmark PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(CPUSerialBenchmark CPUSerialImpl Benchmark)

    add_executable(CPUParallelBenchmark cpu_parallel_benchmark.cpp)
    target_include_directories(CPUParallelBenchmark PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(CPUParallelBenchmark CPUParallelImpl Benchmark)

endif()

if (USE_CUDA)
# Poner aquí los ejecutables y librerías de cuda
endif()