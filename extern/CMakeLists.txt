add_subdirectory(OpenCL-Headers)

option(OPENCL_ICD_LOADER_HEADERS_DIR "Use custom headers" "${CMAKE_SOURCE_DIR}/extern/OpenCL-Headers")
add_subdirectory(OpenCL-ICD-Loader)
