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

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
    void render(
        dim3 grid, dim3 block,
        const uint2* ranges,
        const uint32_t* point_list,
        int W, int H,
        float fx, float fy,
        const float* bg_color,
        int featureCount,
        const float2* means2D,
        const float4* conic_opacities,
        const float* colors,
        const float* features,
        const float* final_Ts,
        const uint32_t* n_contrib,
        const float* buffer,
        const float* grad_colors,
        const float* grad_buffer,
        float4* dL_dmeans2D,
        float4* dL_dconics,
        float* dL_dopacities,
        float* dL_dcolors,
        float* dL_dfeatures);

    void preprocess(
        int P, int D, int M,
        const float3* means,
        const int* radii,
        const float* shs,
        const bool* clamped,
        const glm::vec3* scales,
        const glm::vec4* rotations,
        const float scale_modifier,
        const float* cov3Ds,
        const float* view,
        const float* proj,
        float focal_x, float focal_y,
        float tan_fovx, float tan_fovy,
        const glm::vec3* campos,
        const float4* dL_dmeans2D,
        const float* dL_dconics,
        const float* dL_dcolors,
        glm::vec3* dL_dmeans3D,
        float* dL_dcov3D,
        float* dL_dshs,
        glm::vec3* dL_dscales,
        glm::vec4* dL_drots);
}

#endif