#ifndef ARRAY_MACROS_H
#define ARRAY_MACROS_H

#include "constants.h"
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
