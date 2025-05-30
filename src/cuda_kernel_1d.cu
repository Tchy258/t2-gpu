
using ubyte = unsigned char;

extern "C"  
__global__ void life_step_kernel1d(const ubyte* grid, ubyte* next,
                                 int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int worldSize = width * height;

    for (int cell = idx; cell < worldSize; cell += stride) {
        int x = cell % width, y = cell / width;
        int xm1 = (x + width - 1) % width;
        int xp1 = (x + 1) % width;
        int ym1 = (y + height - 1) % height;
        int yp1 = (y + 1) % height;

        int count = 0;
        count += grid[ym1 * width + xm1];
        count += grid[ym1 * width +  x ];
        count += grid[ym1 * width + xp1];
        count += grid[y    * width + xm1];
        count += grid[y    * width + xp1];
        count += grid[yp1 * width + xm1];
        count += grid[yp1 * width +  x ];
        count += grid[yp1 * width + xp1];

        ubyte alive = grid[y * width + x];
        next[y * width + x] = (count == 3 || (count == 2 && alive)) ? 1 : 0;
    }
}