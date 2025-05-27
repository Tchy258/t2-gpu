if (USE_OPENCL)

  add_library(Benchmark STATIC benchmark.cpp)
  target_include_directories(Benchmark PRIVATE "${PROJECT_SOURCE_DIR}/include")

  option(OPENCL_ICD_LOADER_HEADERS_DIR
  "Path to the OpenCL-Headers"
  "${CMAKE_SOURCE_DIR}/extern/OpenCL-Headers")

  add_library(OpenCLImplLib STATIC game_of_life_opencl.cpp)
  target_include_directories(OpenCLImplLib
    PUBLIC
      "${PROJECT_SOURCE_DIR}/include"                       # tus .hpp
      "${CMAKE_SOURCE_DIR}/extern/OpenCL-CLHPP/include"     # CL/opencl.hpp
  )
  target_link_libraries(OpenCLImplLib
  PRIVATE
    OpenCL::Headers    # para <CL/cl.h>
    OpenCL::OpenCL     # la librería ICD loader
  )
    
  add_executable(OpenCLBenchmark opencl_benchmark.cpp)
  target_include_directories(OpenCLBenchmark
    PRIVATE
      "${PROJECT_SOURCE_DIR}/include"
      "${CMAKE_SOURCE_DIR}/extern/OpenCL-CLHPP/include"
  )
  target_link_libraries(OpenCLBenchmark
    OpenCLImplLib   # tu librería OpenCL+GameOfLife
    Benchmark       # tu librería con la función benchmark()
  )


  add_executable(testOpenCL prueba.cpp)

  target_include_directories(testOpenCL PRIVATE ${CMAKE_SOURCE_DIR}/extern/OpenCL-CLHPP/include)

  target_link_libraries(testOpenCL
  PRIVATE
    OpenCL::Headers    # para <CL/cl.h>
    OpenCL::OpenCL     # la librería ICD loader
  )

endif()

if (USE_CUDA)
  enable_language(CUDA)
  add_executable(cuda_life game_of_life_cuda.cu)
  target_compile_definitions(cuda_life PRIVATE USE_CUDA)
  target_include_directories(cuda_life PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_compile_features(cuda_life PRIVATE cxx_std_17)
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

