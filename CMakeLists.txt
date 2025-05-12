cmake_minimum_required(VERSION 3.10.0)
project(Tarea2 VERSION 0.1.0 LANGUAGES C CXX)

add_subdirectory(extern)
add_subdirectory(src)

include(CTest)
enable_testing()

