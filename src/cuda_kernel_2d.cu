
using ubyte = unsigned char;
__global__ void life_step_kernel2d(const ubyte* const* grid, ubyte** next,
                                   int width, int height)
{
    // Flatten 2D thread/block index to a single thread ID
    int local_id  = threadIdx.y * blockDim.x + threadIdx.x;
    int block_id  = blockIdx.y * gridDim.x + blockIdx.x;
    int thread_id = block_id * blockDim.x * blockDim.y + local_id;

    int total_threads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    int world_size = width * height;

    for (int cell = thread_id; cell < world_size; cell += total_threads) {
        int x = cell % width;
        int y = cell / width;

        int xm1 = (x + width - 1) % width;
        int xp1 = (x + 1) % width;
        int ym1 = (y + height - 1) % height;
        int yp1 = (y + 1) % height;

        int count = 0;
        count += grid[ym1][xm1];
        count += grid[ym1][ x ];
        count += grid[ym1][xp1];
        count += grid[ y ][xm1];
        count += grid[ y ][xp1];
        count += grid[yp1][xm1];
        count += grid[yp1][ x ];
        count += grid[yp1][xp1];

        ubyte alive = grid[y][x];
        next[y][x] = (count == 3 || (count == 2 && alive)) ? 1 : 0;
    }
}