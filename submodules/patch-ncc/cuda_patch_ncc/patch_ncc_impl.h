#pragma once

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
    bool* valid);