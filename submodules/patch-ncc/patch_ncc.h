#pragma once
#include <cstdio>
#include <string>
#include <torch/extension.h>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
WarpPatchNCC(
    const torch::Tensor& depths,
    const torch::Tensor& normals,
    const torch::Tensor& uvs,
    const torch::Tensor& R, // r to n
    const torch::Tensor& T, // r to n
    const torch::Tensor& image_r,
    const torch::Tensor& image_n,
    const float fx_r, const float fy_r,
    const float cx_r, const float cy_r,
    const float fx_n, const float fy_n,
    const float cx_n, const float cy_n);
