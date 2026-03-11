#include "config.h"


__global__ void bilagrid_uniform_sample_forward_kernel(
    const float* __restrict__ bilagrid, // [N,12,L,H,W]
    const float* __restrict__ rgb,  // [N,m,h,w,3]
    float* __restrict__ output,  // [N,m,h,w,3]
    int N, int L, int H, int W,
    int m, int h, int w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * m * h * w;
    if (idx >= total) return;

    int tmp = idx;
    int wi = tmp % w; tmp /= w;
    int hi = tmp % h; tmp /= h;
    int mi = tmp % m; tmp /= m;
    int ni = tmp;

    // read coords
    int g_offset = (((ni * m + mi) * h + hi) * w + wi) * 3;

    // input and output colors
    float sr = rgb[g_offset+0];
    float sg = rgb[g_offset+1];
    float sb = rgb[g_offset+2];
    float dr = 0.0, dg = 0.0, db = 0.0;

    // grid coords
    float gx = (float)wi / (float)(w-1);
    float gy = (float)hi / (float)(h-1);
    float gz = kC2G_r * sr + kC2G_g * sg + kC2G_b * sb;
    float x = gx * (W - 1);
    float y = gy * (H - 1);
    float z = gz * (L - 1);

    // find corner indices
    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int z0 = (int)floorf(z);
    int x1 = min(x0+1, W-1);
    int y1 = min(y0+1, H-1);
    int z1 = z0 + 1;
    z0 = min(max(z0, 0), L-1);
    z1 = min(max(z1, 0), L-1);

    // interpolation parameters
    float fx = x - (float)x0;
    float fy = y - (float)y0;
    float fz = z - (float)z0;

    // interpolate and and affine in one loop
    #pragma unroll
    for (int ci = 0; ci < 12; ci++) {
        // base pointer for this volume
        int base = (ni*12 + ci)*L*H*W;

        // fetch 8 corners
        auto v000 = bilagrid[base+(z0*H+y0)*W+x0];
        auto v001 = bilagrid[base+(z0*H+y0)*W+x1];
        auto v010 = bilagrid[base+(z0*H+y1)*W+x0];
        auto v011 = bilagrid[base+(z0*H+y1)*W+x1];
        auto v100 = bilagrid[base+(z1*H+y0)*W+x0];
        auto v101 = bilagrid[base+(z1*H+y0)*W+x1];
        auto v110 = bilagrid[base+(z1*H+y1)*W+x0];
        auto v111 = bilagrid[base+(z1*H+y1)*W+x1];

        // trilinear interp
        float c00 = v000*(1.0f-fx) + v001*fx;
        float c01 = v010*(1.0f-fx) + v011*fx;
        float c10 = v100*(1.0f-fx) + v101*fx;
        float c11 = v110*(1.0f-fx) + v111*fx;
        float c0 = c00*(1.0f-fy) + c01*fy;
        float c1 = c10*(1.0f-fy) + c11*fy;
        float val = c0*(1.0f-fz) + c1*fz;

        // affine transform
        int si = ci % 4;
        int di = ci / 4;
        (di == 0 ? dr : di == 1 ? dg : db) += val * 
            (si==0 ? sr : si==1 ? sg : si==2 ? sb : 1.0f);
    }

    output[g_offset+0] = isfinite(dr) ? dr : 0.5f;
    output[g_offset+1] = isfinite(dg) ? dg : 0.5f;
    output[g_offset+2] = isfinite(db) ? db : 0.5f;
}


void bilagrid_uniform_sample_forward(
    const float* bilagrid,
    const float* rgb,
    float* output,
    int N, int L, int H, int W,
    int m, int h, int w
) {
    int total = N * m * h * w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    bilagrid_uniform_sample_forward_kernel<<<blocks, threads>>>(
        bilagrid, rgb, output,
        N, L, H, W, m, h, w
    );
    // cudaDeviceSynchronize();
}
