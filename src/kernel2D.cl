__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
                             CLK_ADDRESS_REPEAT | 
                             CLK_FILTER_NEAREST;

__kernel void life(
    read_only image2d_t current,
    write_only image2d_t next,
    const int width,
    const int height,
    const int cells_per_thread_x,
    const int cells_per_thread_y
) {
    const int x_base = get_global_id(0) * cells_per_thread_x;
    const int y_base = get_global_id(1) * cells_per_thread_y;
    
    for (int y_offset = 0; y_offset < cells_per_thread_y; y_offset++) {
        const int y = y_base + y_offset;
        if (y >= height) continue;
        
        for (int x_offset = 0; x_offset < cells_per_thread_x; x_offset++) {
            const int x = x_base + x_offset;
            if (x >= width) continue;
            
            int2 coord = (int2)(x, y);
            uchar alive = 0;
            
            // Check all 8 neighbors
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    
                    const int2 ncoord = (int2)(x + dx, y + dy);
                    uint4 pixel = read_imageui(current, sampler, ncoord);
                    alive += pixel.x;
                }
            }
            
            uint4 cell = read_imageui(current, sampler, coord);
            uchar new_state = (alive == 3) || (cell.x && alive == 2);
            write_imageui(next, coord, (uint4)(new_state, 0, 0, 0));
        }
    }
}