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