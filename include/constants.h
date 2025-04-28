// Game settings
#ifndef GRID_ROWS
#define GRID_ROWS 1024
#endif
#ifndef GRID_COLS
#define GRID_COLS 1024
#endif

// GPU Settings
#ifndef BLOCK_SIZE_X
#define BLOCK_SIZE_X 32
#endif
#ifndef BLOCK_SIZE_Y
#define BLOCK_SIZE_Y 32
#endif

// Macros for conditionally picking 1d or 2d arrays
#ifdef ARRAY_2D
    #define ARRAY_TYPE(T, NAME) T NAME[GRID_ROWS][GRID_COLS]
    #define ARRAY_ACCESS(NAME, i, j)  NAME[i][j]
#else
    #define ARRAY_TYPE(T, NAME) T NAME[(GRID_ROWS)*(GRID_COLS)]
    #define ARRAY_ACCESS(NAME, i, j)  NAME[(i)*(GRID_COLS) + (j)]
#endif