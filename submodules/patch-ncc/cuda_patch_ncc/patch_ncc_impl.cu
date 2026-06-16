// RaDe-GS implementation of the NCC loss, first introduced for Gaussian Splatting by PGSR.
// This may not be the optimal solution. Future work could explore distributing the loops across multiple threads to enable coalesced memory access and boost parallelism.
#include "mathUtils.h"
#include "patch_ncc_impl.h"
#include <cooperative_groups.h>
#include <math_constants.h>

namespace cg = cooperative_groups;

#define FAST_LOAD 1

__device__ constexpr int ceil_int(float x) {
    int i = static_cast<int>(x);
    return (x == static_cast<float>(i)) ? i : (x > 0.0f ? i + 1 : i);
}

// forward-mode differentiation
template <int RADIUS, bool HALF_STEP = true>
__global__ void forward_mode_differentiation_CUDA(
    const int P,
    const float* __restrict__ depths,
    const float3* __restrict__ normals,
    const int2* __restrict__ uvs,
    const float* R, // r to n
    const float* T, // r to n
    const float* __restrict__ image_r,
    const float* __restrict__ image_n,
    const float fx_r, const float fy_r,
    const float cx_r, const float cy_r,
    const float fx_n, const float fy_n,
    const float cx_n, const float cy_n,
    const int image_height_r,
    const int image_width_r,
    const int image_height_n,
    const int image_width_n,
    float* __restrict__ ncc,
    float* __restrict__ grad_depths,
    float3* __restrict__ grad_normals,
    bool* __restrict__ valid) {
    static_assert(RADIUS > 0, "RADIUS must be positive");
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;
    constexpr int LINE_SIZE        = RADIUS * 2 + 1;
    constexpr int TOTAL_SIZE       = LINE_SIZE * LINE_SIZE;
    constexpr float TOTAL_SIZE_INV = 1.f / static_cast<float>(TOTAL_SIZE);
    constexpr float RADIUS_F       = static_cast<float>(RADIUS) * (HALF_STEP ? 0.5 : 1.f);

    const int2 uv_center_r = uvs[idx];
    const float depth      = depths[idx];
    const float3 normal    = normals[idx];
    const float3 point_n_r = {(uv_center_r.x - cx_r) / fx_r, (uv_center_r.y - cy_r) / fy_r, 1.f};
    float distance         = -dot(point_n_r, normal) * depth;

    bool all_inside = uv_center_r.x - RADIUS_F > 0 &&
                      uv_center_r.x + RADIUS_F < image_width_r - 1 &&
                      uv_center_r.y - RADIUS_F > 0 &&
                      uv_center_r.y + RADIUS_F < image_height_r - 1;
    if (!all_inside)
        return;

    float33 R_mat = {{R[0], R[1], R[2]},
                     {R[3], R[4], R[5]},
                     {R[6], R[7], R[8]}};
    float3 T_mat  = {T[0], T[1], T[2]};
    float33 Hn;
    // normalized H
    Hn[0] = R_mat[0] - normal.x / distance * T_mat;
    Hn[1] = R_mat[1] - normal.y / distance * T_mat;
    Hn[2] = R_mat[2] - normal.z / distance * T_mat;

    // K_n * Hn
    float33 H;
    // K_n * Hn
#pragma unroll
    for (int i = 0; i < 3; i++) {
        H[i] = make_float3(fx_n * Hn[i].x + cx_n * Hn[i].z, fy_n * Hn[i].y + cy_n * Hn[i].z, Hn[i].z);
    }

    // Hn * K_r_inv
    H[2] = H[0] * (-cx_r / fx_r) + H[1] * (-cy_r / fy_r) + H[2];
    H[0] = H[0] / fx_r;
    H[1] = H[1] / fy_r;

    float3 H_uc = H * make_float3(static_cast<float>(uv_center_r.x), static_cast<float>(uv_center_r.y), 1.f);

    auto inside_n = [image_height_n, image_width_n](float2 uv) {
        return uv.x - RADIUS_F > 0 &&
               uv.x + RADIUS_F < image_width_n - 1 &&
               uv.y - RADIUS_F > 0 &&
               uv.y + RADIUS_F < image_height_n - 1;
    };

    constexpr int RADIUS_SCALED                            = ceil_int(RADIUS * 0.5);
    constexpr int LINE_LENGTH_SCALED                       = RADIUS_SCALED * 2 + 1;
    [[maybe_unused]] float cached_line[LINE_LENGTH_SCALED];
    if constexpr (HALF_STEP) {
        const int v = uv_center_r.y - RADIUS_SCALED;
#pragma unroll
        for (int i = 0; i < LINE_LENGTH_SCALED; i++) {
            cached_line[i] = image_r[v * image_width_r + uv_center_r.x - RADIUS_SCALED + i];
        }
    }
    float sum_c_r     = 0;
    float sum_c_n     = 0;
    float sum_c_r2    = 0;
    float sum_c_n2    = 0;
    float sum_c_r_c_n = 0;

    float3 grad_n  = {0};
    float3 grad_n2 = {0};
    float3 grad_rn = {0};

    const float3 aux = T_mat / distance;
    for (int dv = -RADIUS; dv <= RADIUS; dv++) {
        float dv_f    = HALF_STEP ? 0.5f * dv : dv;
        bool is_odd_v = (dv & 1) != 0;
        int v0r       = uv_center_r.y + (is_odd_v ? __float2int_rd(dv_f) : dv / 2 - 1);
        int v1r       = uv_center_r.y + (is_odd_v ? __float2int_ru(dv_f) : dv / 2);

        [[maybe_unused]] float last_c = 0;
        if constexpr (HALF_STEP) {
            if (is_odd_v)
                last_c = __ldg(&image_r[v1r * image_width_r + uv_center_r.x - RADIUS_SCALED]);
        }
        [[maybe_unused]] float w_v0 = is_odd_v ? 0.5f : 1.f;
        [[maybe_unused]] float w_v1 = is_odd_v ? 0.5f : 0.f;

        const float3 H_uc_v = H_uc + dv_f * H[1];
        for (int du = -RADIUS; du <= RADIUS; du++) {
            float du_f  = HALF_STEP ? 0.5f * du : du;
            float2 uv_r = {uv_center_r.x + du_f, uv_center_r.y + dv_f};
            // sample values from the reference image.
            float c_r;
            if constexpr (HALF_STEP) {
#if FAST_LOAD
                bool is_odd_u = (du & 1) != 0;
                float w_u0    = is_odd_u ? 0.5f : 1.f;
                float w_u1    = is_odd_u ? 0.5f : 0.f;
                int u0r       = is_odd_u ? __float2int_rd(du_f) : du / 2;
                int u1r       = is_odd_u ? __float2int_ru(du_f) : du / 2;
                float c00     = cached_line[u0r + RADIUS_SCALED];
                float c01     = cached_line[u1r + RADIUS_SCALED];

                float c10 = last_c;
                float c11 = 0;
                if (is_odd_v && is_odd_u) {
                    // load data
                    c11 = image_r[v1r * image_width_r + uv_center_r.x + u1r];
                    // update cache
                    cached_line[u0r + RADIUS_SCALED] = last_c;
                    last_c                           = c11;
                }
                c_r = (c00 * w_u0 + c01 * w_u1) * w_v0 + (c10 * w_u0 + c11 * w_u1) * w_v1;
#else
                int u0r = __float2int_rd(uv_r.x);
                int u1r = __float2int_ru(uv_r.x);
                u0r -= u0r == u1r;
                float c00r = __ldg(&image_r[v0r * image_width_r + u0r]);
                float c01r = __ldg(&image_r[v0r * image_width_r + u1r]);
                float c10r = __ldg(&image_r[v1r * image_width_r + u0r]);
                float c11r = __ldg(&image_r[v1r * image_width_r + u1r]);
                float w00  = (v1r - uv_r.y) * (u1r - uv_r.x);
                float w01  = (v1r - uv_r.y) * (uv_r.x - u0r);
                float w10  = (uv_r.y - v0r) * (u1r - uv_r.x);
                float w11  = (uv_r.y - v0r) * (uv_r.x - u0r);
                c_r        = w00 * c00r + w01 * c01r + w10 * c10r + w11 * c11r;
#endif
            } else {
                c_r = image_r[(dv + uv_center_r.y) * image_width_r + du + uv_center_r.x];
            }
            // fetch values from the neighboring image.
            float3 H_uv = H_uc_v + du_f * H[0];
            float2 uv_n = {H_uv.x / H_uv.z, H_uv.y / H_uv.z};
            bool inside = inside_n(uv_n);
            all_inside = inside && all_inside;

            float eps_u = nextafterf(uv_n.x, CUDART_INF_F);
            float eps_v = nextafterf(uv_n.y, CUDART_INF_F);

            int u0 = __float2int_rd(uv_n.x);
            int v0 = __float2int_rd(uv_n.y);
            int u1 = __float2int_ru(eps_u);
            int v1 = __float2int_ru(eps_v);

            u0 = clamp(u0, 0, image_width_n - 1);
            u1 = clamp(u1, 0, image_width_n - 1);
            v0 = clamp(v0, 0, image_height_n - 1);
            v1 = clamp(v1, 0, image_height_n - 1);

            float c00n = __ldg(&image_n[v0 * image_width_n + u0]);
            float c01n = __ldg(&image_n[v0 * image_width_n + u1]);
            float c10n = __ldg(&image_n[v1 * image_width_n + u0]);
            float c11n = __ldg(&image_n[v1 * image_width_n + u1]);
            float wv0  = v1 - uv_n.y;
            float wv1  = uv_n.y - v0;
            float wu0  = u1 - uv_n.x;
            float wu1  = uv_n.x - u0;
            float c_n  = wv0 * (wu0 * c00n + wu1 * c01n) + wv1 * (wu0 * c10n + wu1 * c11n);

            sum_c_r += c_r;
            sum_c_n += c_n;
            sum_c_r2 += c_r * c_r;
            sum_c_n2 += c_n * c_n;
            sum_c_r_c_n += c_r * c_n;

            float2 dc_n_duv = {
                -c00n * wv0 + c01n * wv0 - c10n * wv1 + c11n * wv1,
                -c00n * wu0 - c01n * wu1 + c10n * wu0 + c11n * wu1};
            float3 dc_n_dH_uv = {
                dc_n_duv.x / H_uv.z,
                dc_n_duv.y / H_uv.z,
                (-dc_n_duv.x * uv_n.x - dc_n_duv.y * uv_n.y) / H_uv.z};
            float3 dc_n_dHn_left  = {dc_n_dH_uv.x * fx_n,
                                     dc_n_dH_uv.y * fy_n,
                                     dc_n_dH_uv.x * cx_n + dc_n_dH_uv.y * cy_n + dc_n_dH_uv.z};
            float3 dc_n_dHn_right = {
                (uv_r.x - cx_r) / fx_r,
                (uv_r.y - cy_r) / fy_r,
                1.f};
            float3 grad_aux = dc_n_dHn_right * dot(dc_n_dHn_left, aux);
            grad_n += grad_aux;
            grad_n2 += 2.f * c_n * grad_aux;
            grad_rn += c_r * grad_aux;
        }
        if constexpr (HALF_STEP && FAST_LOAD) {
            if (is_odd_v)
                cached_line[2 * RADIUS_SCALED] = last_c;
        }
    }
    const float cross      = sum_c_r_c_n - sum_c_r * sum_c_n * TOTAL_SIZE_INV;
    const float variance_r = sum_c_r2 - sum_c_r * sum_c_r * TOTAL_SIZE_INV;
    const float variance_n = sum_c_n2 - sum_c_n * sum_c_n * TOTAL_SIZE_INV;
    const float output_ncc = cross * cross / (variance_r * variance_n + 1e-8f);

    float grad_cross      = 2.f * cross / (variance_r * variance_n + 1e-8f);
    float grad_variance_n = -output_ncc / (variance_n + 1e-8f);

    float grad_sum_c_n     = (-grad_cross * sum_c_r - grad_variance_n * 2.f * sum_c_n) * TOTAL_SIZE_INV;
    float grad_sum_c_n2    = grad_variance_n;
    float grad_sum_c_r_c_n = grad_cross;

    float3 grad_aux     = grad_sum_c_n * grad_n + grad_sum_c_n2 * grad_n2 + grad_sum_c_r_c_n * grad_rn;
    float3 grad_normal  = -grad_aux;
    float grad_distance = dot(grad_aux, normal) / distance;

    grad_normal += -depth * grad_distance * point_n_r;
    float grad_depth = -dot(point_n_r, normal) * grad_distance;

    bool valid_patch  = all_inside && variance_r > 5e-6f && variance_n > 5e-6f;
    ncc[idx]          = valid_patch ? output_ncc : 0.f;
    grad_depths[idx]  = valid_patch ? grad_depth : 0.f;
    grad_normals[idx] = valid_patch ? grad_normal : make_float3(0);
    valid[idx]        = valid_patch;
}

void forward_mode_differentiation(
    const int P,
    const float* depths,
    const float* normals,
    const int* uv,
    const float* R, // r to n
    const float* T, // r to n
    const float* image_r,
    const float* image_n,
    const float fx_r, const float fy_r,
    const float cx_r, const float cy_r,
    const float fx_n, const float fy_n,
    const float cx_n, const float cy_n,
    const int image_height_r,
    const int image_width_r,
    const int image_height_n,
    const int image_width_n,
    float* ncc,
    float* grad_depths,
    float* grad_normals,
    bool* valid) {
    forward_mode_differentiation_CUDA<3, true><<<(P + 255) / 256, 256>>>(
        P,
        depths,
        reinterpret_cast<const float3*>(normals),
        reinterpret_cast<const int2*>(uv),
        R,
        T,
        image_r,
        image_n,
        fx_r,
        fy_r,
        cx_r,
        cy_r,
        fx_n,
        fy_n,
        cx_n,
        cy_n,
        image_height_r,
        image_width_r,
        image_height_n,
        image_width_n,
        ncc,
        grad_depths,
        reinterpret_cast<float3*>(grad_normals),
        valid);
}