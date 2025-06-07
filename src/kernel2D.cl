__constant sampler_t sam = CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST;

__kernel void life(read_only  image2d_t grid,
                   write_only image2d_t next,
                   const uint cols,
                   const uint rows)
{
    uint gid = get_global_id(0);
    uint gsize = get_global_size(0);
    uint total = cols * rows;

    for (uint idx = gid; idx < total; idx += gsize) {
        int2 xy = (int2)(idx % cols, idx / cols);

        #define P(x,y) (xy + (int2)(x,y))
        uchar alive =
            read_imageui(grid,sam,P(-1,-1)).x + read_imageui(grid,sam,P(0,-1)).x + read_imageui(grid,sam,P(1,-1)).x +
            read_imageui(grid,sam,P(-1, 0)).x                                     + read_imageui(grid,sam,P(1, 0)).x +
            read_imageui(grid,sam,P(-1, 1)).x + read_imageui(grid,sam,P(0, 1)).x + read_imageui(grid,sam,P(1, 1)).x ;

        uchar c = read_imageui(grid,sam,xy).x;
        write_imageui(next, xy, (uint4)((alive == 3 || (alive == 2 && c)) ? 1 : 0, 0, 0, 0));
    }
}
