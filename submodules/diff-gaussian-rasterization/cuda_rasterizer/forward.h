/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD {
    // Perform initial steps for each Gaussian prior to rasterization.
    void preprocess(
        int P, int D, int M,
        const float* orig_points,
        const glm::vec3* scales,
        const float scale_modifier,
        const glm::vec4* rotations,
        const float* opacities,
        const float* shs,
        bool* clamped,
        const float* cov3D_precomp,
        const float* colors_precomp,
        const float* viewmatrix,
        const float* projmatrix,
        const glm::vec3* cam_pos,
        int W, int H,
        float focal_x, float focal_y,
        float tan_fovx, float tan_fovy,
        int* out_radii,
        float2* out_img_points,
        float* out_z_depths,
        float* out_cov3Ds,
        float* out_colors,
        float4* out_conic_opacities,
        dim3 grid,
        uint32_t* out_tiles_touched,
        bool prefiltered);

    // Main rasterization method.
    void render(
        dim3 grid, dim3 block,
        const uint2* ranges,
        const uint32_t* indices,
        int W, int H,
        const float* viewmatrix,
        const float* cam_pos,
        const float2* img_points,
        const float* precomputed_colors,
        const float* features,
        const float4* conic_opacities,
        float* out_final_Ts,
        uint32_t* n_contrib,
        const float* bg_colors,
        int featureCount,
        float* out_colors,
        int* out_observe,
        float* out_buffer);
}

#endif