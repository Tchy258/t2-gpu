#ifdef ARRAY_2D   
/* ---------- 2-D ------------- */
__constant sampler_t sam = CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST;

__kernel void life(read_only  image2d_t grid,
                   write_only image2d_t next,
                   const uint cols,
                   const uint rows)
{
    int2 xy = (int2)(get_global_id(0)%cols,
                     get_global_id(0)/cols);
    if (xy.y>=rows) return;

    int2 xm = (int2)(cols, rows);                // para envolver
    #define P(x,y) (xy + (int2)(x,y))
    uchar alive =
        read_imageui(grid,sam,P(-1,-1)).x + read_imageui(grid,sam,P(0,-1)).x + read_imageui(grid,sam,P(1,-1)).x +
        read_imageui(grid,sam,P(-1, 0)).x                                     + read_imageui(grid,sam,P(1, 0)).x +
        read_imageui(grid,sam,P(-1, 1)).x + read_imageui(grid,sam,P(0, 1)).x + read_imageui(grid,sam,P(1, 1)).x ;

    uchar c = read_imageui(grid,sam,xy).x;
    write_imageui(next, xy, (uint4)((alive==3 || (alive==2&&c))?1:0,0,0,0));
}

#else
/* ---------- 1-D -------------- */
__kernel void life(__global const uchar* grid,
                   __global       uchar* next,
                   const uint cols,
                   const uint rows)
{
    uint idx = get_global_id(0);
    if (idx >= cols*rows) return;
    uint y = idx / cols, x = idx % cols;
    uint x0=(x+cols-1)%cols, x2=(x+1)%cols;
    uint y0=(y+rows-1)%rows, y2=(y+1)%rows;

    #define AT(g,yy,xx) g[(yy)*cols + (xx)]
    uchar alive =
        AT(grid,y0,x0)+AT(grid,y0,x)+AT(grid,y0,x2)+
        AT(grid,y ,x0)           +AT(grid,y ,x2)+
        AT(grid,y2,x0)+AT(grid,y2,x)+AT(grid,y2,x2);

    uchar c = AT(grid,y,x);
    next[idx] = (alive==3 || (alive==2&&c)) ? 1 : 0;
}
#endif