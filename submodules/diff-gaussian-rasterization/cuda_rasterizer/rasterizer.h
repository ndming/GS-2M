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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <functional>

namespace CudaRasterizer {
    class Rasterizer {
    public:
        static void markVisible(
            int P,
            const float* means3D,
            const float* viewmatrix,
            const float* projmatrix,
            bool* present);

        static int forward(
            std::function<char* (size_t)> geometryBuffer,
            std::function<char* (size_t)> binningBuffer,
            std::function<char* (size_t)> imageBuffer,
            int P, int D, int M,
            const float* background,
            int width, int height,
            const float* means3D,
            const float* shs,
            const float* colors_precomp,
            const float* opacities,
            const float* scales,
            float scale_modifier,
            const float* rotations,
            const float* cov3D_precomp,
            const float* features,
            const float* viewmatrix,
            const float* projmatrix,
            const float* cam_pos,
            float tan_fovx,
            float tan_fovy,
            bool prefiltered,
            int featureCount,
            float* out_colors,
            int* out_radii,
            int* out_observe,
            float* out_buffer,
            float* out_depth);

        static void backward(
            int P, int D, int M, int R,
            const float* background,
            int width, int height,
            const float* means3D,
            const float* shs,
            const float* colors_precomp,
            const float* scales,
            const float scale_modifier,
            const float* rotations,
            const float* cov3D_precomp,
            const float* features,
            const float* viewmatrix,
            const float* projmatrix,
            const float* campos,
            const float tan_fovx,
            const float tan_fovy,
            const int* radii,
            const float* buffer,
            char* geom_buffer,
            char* binning_buffer,
            char* image_buffer,
            int featureCount,
            const float* grad_colors,
            const float* grad_buffer,
            const float* grad_depth,
            float* dL_dmeans2D,
            float* dL_dconics,
            float* dL_dopacities,
            float* dL_dcolors,
            float* dL_dmeans3D,
            float* dL_dcov3D,
            float* dL_dshs,
            float* dL_dscales,
            float* dL_drots,
            float* dL_dfeatures);
    };
};

#endif