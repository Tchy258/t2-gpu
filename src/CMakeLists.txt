add_library(Benchmark STATIC benchmark.cpp)
target_include_directories(Benchmark PRIVATE "${PROJECT_SOURCE_DIR}/include")
if (USE_OPENCL)
  add_library(ProjectHeaders INTERFACE)
  target_include_directories(ProjectHeaders INTERFACE "${PROJECT_SOURCE_DIR}/include")
  target_link_libraries(ProjectHeaders INTERFACE OpenCL-Headers OpenCL-CLHPP)

  add_library(OpenCLImplLib STATIC game_of_life_opencl.cpp)


  target_link_libraries(OpenCLImplLib
    PRIVATE
      ProjectHeaders
      OpenCL::OpenCL        # from ICD loader
  )

  add_executable(OpenCLBenchmark opencl_benchmark.cpp)
  target_link_libraries(OpenCLBenchmark
    PRIVATE
      ProjectHeaders
      OpenCLImplLib
      Benchmark
  )
  file(GLOB KERNEL_FILES "${CMAKE_SOURCE_DIR}/src/*.cl")
  foreach(KERNEL_FILE ${KERNEL_FILES})
      configure_file(${KERNEL_FILE} ${CMAKE_BINARY_DIR}/src/ COPYONLY)
  endforeach()

endif()

if (USE_CUDA)
  enable_language(CUDA)
  add_library(cuda_life STATIC game_of_life_cuda.cu)
  #add_executable(cuda_life game_of_life_cuda.cu)
  target_include_directories(cuda_life PRIVATE "${PROJECT_SOURCE_DIR}/include")

  add_executable(CudaBenckmark cuda_benchmark.cpp)
  target_include_directories(CudaBenckmark PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_link_libraries(CudaBenckmark 
    cuda_life
    Benchmark)
endif()

if (USE_CPU)

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

