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

#pragma once
#include <torch/extension.h>
#include <tuple>

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& features,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    float tan_fovx,
    float tan_fovy,
    int image_height,
    int image_width,
    const torch::Tensor& sh,
    int degree,
    const torch::Tensor& campos,
    bool prefiltered,
    int featureCount);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& radii,
    const torch::Tensor& buffer,
    const torch::Tensor& colors,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& features,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    float tan_fovx,
    float tan_fovy,
    const torch::Tensor& grad_colors,
    const torch::Tensor& grad_buffer,
    const torch::Tensor& sh,
    int degree,
    const torch::Tensor& campos,
    const torch::Tensor& geomBuffer,
    int R, // total tiles touched
    const torch::Tensor& binningBuffer,
    const torch::Tensor& imageBuffer,
    int featureCount);

torch::Tensor markVisible(
    const torch::Tensor& means3D,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix);