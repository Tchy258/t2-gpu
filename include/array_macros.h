#ifndef ARRAY_MACROS
#define ARRAY_MACROS
#include "constants.h"

    #ifdef USE_CPU

        #ifdef ARRAY_2D
            #define ARRAY_TYPE(T, NAME) T** NAME
            #define ARRAY_ALLOC(T, NAME) do {                      \
                NAME = new T*[GRID_ROWS];                          \
                for (int i = 0; i < GRID_ROWS; ++i)                \
                    NAME[i] = new T[GRID_COLS];                    \
            } while (0)
            #define ARRAY_ACCESS(NAME, i, j) NAME[i][j]
            #define ARRAY_DELETE(NAME) do {                        \
                for (int i = 0; i < GRID_ROWS; ++i)                \
                    delete[] NAME[i];                              \
                delete[] NAME;                                     \
            } while (0)

        #else
            #define ARRAY_TYPE(T, NAME) T* NAME
            #define ARRAY_ALLOC(T, NAME) \
                NAME = new T[(GRID_ROWS) * (GRID_COLS)]
            #define ARRAY_ACCESS(NAME, i, j) NAME[(i)*(GRID_COLS) + (j)]
            #define ARRAY_DELETE(NAME) delete[] NAME
        #endif

    #endif

    #ifdef USE_CUDA
    #include <cuda_runtime.h>

        #ifdef ARRAY_2D
            #define ARRAY_TYPE(T, NAME) T** NAME
            #define ARRAY_ALLOC(T, NAME) do {                                     \
                NAME = nullptr;                                                   \
                cudaMalloc(&NAME, GRID_ROWS * sizeof(T*));                        \
                T* temp[GRID_ROWS];                                               \
                for (int i = 0; i < GRID_ROWS; ++i)                               \
                    cudaMalloc(&temp[i], GRID_COLS * sizeof(T));                  \
                cudaMemcpy(NAME, temp, GRID_ROWS * sizeof(T*), cudaMemcpyHostToDevice); \
            } while (0)
            #define ARRAY_ACCESS(NAME, i, j) NAME[i][j]
            #define ARRAY_DELETE(NAME) do {                                       \
                T** temp_host = new T*[GRID_ROWS];                                \
                cudaMemcpy(temp_host, NAME, GRID_ROWS * sizeof(T*), cudaMemcpyDeviceToHost); \
                for (int i = 0; i < GRID_ROWS; ++i)                               \
                    cudaFree(temp_host[i]);                                       \
                delete[] temp_host;                                               \
                cudaFree(NAME);                                                   \
            } while (0)

        #else
            #define ARRAY_TYPE(T, NAME) T* NAME
            #define ARRAY_ALLOC(T, NAME) \
                cudaMalloc(&NAME, (GRID_ROWS)*(GRID_COLS)*sizeof(T))
            #define ARRAY_ACCESS(NAME, i, j) NAME[(i)*(GRID_COLS) + (j)]
            #define ARRAY_DELETE(NAME) cudaFree(NAME)
        #endif

    #endif

    #ifdef USE_OPENCL
        

        #ifdef ARRAY_2D
            #define ARRAY_TYPE(T, NAME) cl_mem NAME
            #define ARRAY_ALLOC(T, NAME) do {                                  \
                NAME = clCreateBuffer(context, CL_MEM_READ_WRITE,              \
                    GRID_ROWS * GRID_COLS * sizeof(T), nullptr, nullptr);     \
            } while (0)
            #define ARRAY_ACCESS(T, NAME, i, j)                                    \
                ((T*)clEnqueueMapBuffer(queue, NAME, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, \
                (i)*(GRID_COLS) + (j), sizeof(T), 0, nullptr, nullptr, nullptr))[(i)*(GRID_COLS) + (j)]

            #define ARRAY_DELETE(NAME) clReleaseMemObject(NAME)

        #else
            #define ARRAY_ALLOC(T, NAME) do {                                \
                cl::Buffer buf  = cl::Buffer(context_cpp, CL_MEM_READ_WRITE,             \
                                    sizeof(T)*GRID_ROWS*GRID_COLS);             \
            } while(0)
            #define ARRAY_TYPE(T, NAME) T NAME
            #define ARRAY_ACCESS(T, NAME, i, j)                               \
                ((T*)clEnqueueMapBuffer(queue, NAME, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, \
                (i)*(GRID_COLS) + (j), sizeof(T), 0, nullptr, nullptr, nullptr))[(i)*(GRID_COLS) + (j)]

            #define ARRAY_DELETE(NAME)
        #endif

    #endif

#endif