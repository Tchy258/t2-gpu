add_executable(opencl_test opencl_test.cpp)
target_compile_features(opencl_test PRIVATE cxx_auto_type)

target_include_directories(opencl_test PRIVATE "${CMAKE_SOURCE_DIR}/extern/OpenCL-Headers")
target_link_libraries(opencl_test OpenCL::OpenCL)

if (USE_CPU_SERIAL)
    add_compile_definitions(USE_CPU)
    add_library(CPUSerialImpl STATIC game_of_life_serial.cpp)
    target_include_directories(CPUSerialImpl PRIVATE "${PROJECT_SOURCE_DIR}/include")
    add_executable(GameOfLifeCPUSerial cpu_serial_example.cpp)
    target_include_directories(GameOfLifeCPUSerial PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(GameOfLifeCPUSerial CPUSerialImpl)
endif()
