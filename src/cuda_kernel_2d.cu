
using ubyte = unsigned char;
extern "C"
__global__ void life_step_kernel2d(const ubyte* const* grid, ubyte** next, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int xm1 = (x + width - 1) % width;
    int xp1 = (x + 1) % width;
    int ym1 = (y + height - 1) % height;
    int yp1 = (y + 1) % height;

    int count = 0;
    count += grid[ym1][xm1];
    count += grid[ym1][x];
    count += grid[ym1][xp1];
    count += grid[y][xm1];
    count += grid[y][xp1];
    count += grid[yp1][xm1];
    count += grid[yp1][x];
    count += grid[yp1][xp1];

    ubyte alive = grid[y][x];
    next[y][x] = (count == 3 || (count == 2 && alive)) ? 1 : 0;
}
