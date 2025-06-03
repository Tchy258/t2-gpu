cmake_minimum_required(VERSION 3.14.0)

set(CMAKE_CXX_STANDARD 11)

project(Tarea2 VERSION 0.1.0 LANGUAGES C CXX)

if (USE_OPENCL)
   add_subdirectory(extern)
endif()

# Handle MSVC separately to avoid $(VAR) expansion issue
if(MSVC)
    foreach(var GRID_ROWS GRID_COLS BLOCK_SIZE_X BLOCK_SIZE_Y)
        if(DEFINED ${var})
            add_compile_options("/D${var}=${${var}}")
        endif()
    endforeach()

    if(DEFINED ARRAY_2D)
        add_compile_options("/DARRAY_2D")
    endif()
else()
    foreach(var GRID_ROWS GRID_COLS BLOCK_SIZE_X BLOCK_SIZE_Y)
        if(DEFINED ${var})
            add_compile_definitions(${var}=${${var}})
        endif()
    endforeach()

    if(DEFINED ARRAY_2D)
        add_compile_definitions(ARRAY_2D)
    endif()
endif()


add_subdirectory(src)

include(CTest)
enable_testing()

