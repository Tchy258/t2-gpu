cmake_minimum_required(VERSION 3.10.0)

project(Tarea2 VERSION 0.1.0 LANGUAGES C CXX)

if (USE_OPENCL)
  add_subdirectory(extern)
endif()

add_subdirectory(src)

include(CTest)
enable_testing()

