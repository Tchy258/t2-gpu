__kernel void life_coarse(read_only image2d_t grid,
                          write_only image2d_t next,
                          const uint cols,
                          const uint rows)
{
    uint gid = get_global_id(0);
    uint gsize = get_global_size(0);
    uint total = cols * rows;

    for (uint idx = gid; idx < total; idx += gsize) {
        int x = idx % cols;
        int y = idx / cols;

        // Manual wrapping
        int x0 = (x + cols - 1) % cols;
        int x2 = (x + 1) % cols;
        int y0 = (y + rows - 1) % rows;
        int y2 = (y + 1) % rows;

        // Load all neighbors manually using wrapped coordinates
        uchar alive =
            read_imageui(grid, (int2)(x0, y0)).x + read_imageui(grid, (int2)(x , y0)).x + read_imageui(grid, (int2)(x2, y0)).x +
            read_imageui(grid, (int2)(x0, y )).x                                      + read_imageui(grid, (int2)(x2, y )).x +
            read_imageui(grid, (int2)(x0, y2)).x + read_imageui(grid, (int2)(x , y2)).x + read_imageui(grid, (int2)(x2, y2)).x;

        uchar c = read_imageui(grid, (int2)(x, y)).x;
        write_imageui(next, (int2)(x, y), (uint4)((alive == 3 || (alive == 2 && c)) ? 1 : 0, 0, 0, 0));
    }
}
