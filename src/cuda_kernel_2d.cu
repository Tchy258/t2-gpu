
using ubyte = unsigned char;
__global__ void life_step_kernel2d(
    ubyte* const* grid,
    ubyte** next,
    int width,
    int height,
    int cells_per_thread_x,
    int cells_per_thread_y
) {
    const int x_base = blockIdx.x * blockDim.x * cells_per_thread_x + threadIdx.x;
    const int y_base = blockIdx.y * blockDim.y * cells_per_thread_y + threadIdx.y;
    
    for (int y_offset = 0; y_offset < cells_per_thread_y; y_offset++) {
        const int y = y_base + y_offset * blockDim.y;
        if (y >= height) continue;
        
        for (int x_offset = 0; x_offset < cells_per_thread_x; x_offset++) {
            const int x = x_base + x_offset * blockDim.x;
            if (x >= width) continue;
            
            ubyte alive = 0;
            // Check all 8 neighbors
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    
                    const int nx = (x + dx + width) % width;
                    const int ny = (y + dy + height) % height;
                    alive += grid[ny][nx];
                }
            }
            
            const ubyte cell = grid[y][x];
            next[y][x] = (alive == 3) || (cell && alive == 2) ? 1 : 0;
        }
    }
}