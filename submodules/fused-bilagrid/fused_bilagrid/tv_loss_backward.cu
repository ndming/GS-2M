
__global__ void tv_loss_backward_kernel(
    const float* __restrict__ bilagrid,   // [N,12,L,H,W]
    const float v_tv_loss,                   // scalar gradient dL/d(tv_loss)
    float* __restrict__ v_bilagrid,     // [N,12,L,H,W]
    int N, int L, int H, int W
) {
    int wi = blockIdx.x * blockDim.x + threadIdx.x;
    int hi = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.z * blockDim.z + threadIdx.z;
    if (wi >= W || hi >= H || idx >= (N * L)) return;

    int li = idx % L; idx /= L;
    int ni = idx;

    float s = v_tv_loss / (6*N);
    float sx = s / (float)(L * H * (W - 1));
    float sy = s / (float)(L * (H - 1) * W);
    float sz = s / (float)((L - 1) * H * W);

    for (int ci = 0; ci < 12; ci++) {

        int cell_idx = (((ni * 12 + ci) * L + li) * H + hi) * W + wi;

        float half_grad = 0.0f;
        float val = bilagrid[cell_idx];

        if (wi > 0) {
            float val0 = bilagrid[cell_idx - 1];
            half_grad += (val - val0) * sx;
        }
        if (wi < W - 1) {
            float val0 = bilagrid[cell_idx + 1];
            half_grad += (val - val0) * sx;
        }
        if (hi > 0) {
            float val0 = bilagrid[cell_idx - W];
            half_grad += (val - val0) * sy;
        }
        if (hi < H - 1) {
            float val0 = bilagrid[cell_idx + W];
            half_grad += (val - val0) * sy;
        }
        if (li > 0) {
            float val0 = bilagrid[cell_idx - W*H];
            half_grad += (val - val0) * sz;
        }
        if (li < L - 1) {
            float val0 = bilagrid[cell_idx + W*H];
            half_grad += (val - val0) * sz;
        }

        v_bilagrid[cell_idx] = half_grad;
    }
}


void tv_loss_backward(
    const float* bilagrid,
    const float v_tv_loss,
    float* v_bilagrid,
    int N, int L, int H, int W
) {
    dim3 block(4, 4, 4);
    dim3 grid(
        (W + block.x - 1) / block.x,
        (H + block.y - 1) / block.y,
        (N*L + block.z - 1) / block.z
    );
    tv_loss_backward_kernel<<<grid, block>>>(
        bilagrid, v_tv_loss, v_bilagrid, N, L, H, W
    );
}
