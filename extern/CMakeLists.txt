# Header-only library
add_subdirectory(OpenCL-Headers)

# Let ICD loader know where headers are
option(OPENCL_ICD_LOADER_HEADERS_DIR "Use custom headers" "${CMAKE_SOURCE_DIR}/extern/OpenCL-Headers")
add_subdirectory(OpenCL-ICD-Loader)

# Header-only C++ bindings
add_subdirectory(OpenCL-CLHPP)

# Define INTERFACE targets for the headers
add_library(OpenCL-Headers INTERFACE)
target_include_directories(OpenCL-Headers INTERFACE "${CMAKE_SOURCE_DIR}/extern/OpenCL-Headers")

add_library(OpenCL-CLHPP INTERFACE)
target_include_directories(OpenCL-CLHPP INTERFACE "${CMAKE_SOURCE_DIR}/extern/OpenCL-CLHPP/include")
