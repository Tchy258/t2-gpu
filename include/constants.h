// Game settings
#ifndef CONSTANTS_H
#define CONSTANTS_H

#ifndef GRID_ROWS
#define GRID_ROWS (32768ULL)
#endif
#ifndef GRID_COLS
#define GRID_COLS (32768ULL)
#endif

//#ifndef ARRAY_2D
//#define ARRAY_2D
//#endif

// GPU Settings
#ifndef BLOCK_SIZE_X
#define BLOCK_SIZE_X 32
#endif
#ifndef BLOCK_SIZE_Y
#define BLOCK_SIZE_Y 32
#endif

#ifdef ARRAY_2D
    #define TILE_WIDTH (static_cast<unsigned int>(sqrt(BLOCK_SIZE_X)))
    #define TILE_HEIGHT (static_cast<unsigned int>(sqrt(BLOCK_SIZE_Y)))
#else
    #define CELLS_PER_THREAD (static_cast<unsigned int>(sqrt(BLOCK_SIZE_X * BLOCK_SIZE_Y)))
#endif

#endif
