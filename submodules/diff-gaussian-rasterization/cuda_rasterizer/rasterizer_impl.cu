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

#include "rasterizer.h"
#include "rasterizer_impl.h"
#include <cuda.h>
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n) {
    uint32_t msb = sizeof(n) * 4;
    uint32_t step = msb;
    while (step > 1) {
        step /= 2;
        if (n >> msb)
            msb += step;
        else
            msb -= step;
    }
    if (n >> msb)
        msb++;
    return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
        const float* orig_points,
        const float* viewmatrix,
        const float* projmatrix,
        bool* present) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    float3 p_view;
    present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
        int P,
        const float2* points_xy,
        const float* depths,
        const uint32_t* offsets,
        uint64_t* gaussian_keys_unsorted,
        uint32_t* gaussian_values_unsorted,
        int* radii,
        dim3 grid) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    // Generate no key/value pair for invisible Gaussians
    if (radii[idx] > 0)
    {
        // Find this Gaussian's offset in buffer for writing keys/values.
        uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
        uint2 rect_min, rect_max;

        getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

        // For each tile that the bounding rect overlaps, emit a 
        // key/value pair. The key is |  tile ID  |      depth      |,
        // and the value is the ID of the Gaussian. Sorting the values 
        // with this key yields Gaussian IDs in a list, such that they
        // are first sorted by tile and then by depth. 
        for (int y = rect_min.y; y < rect_max.y; y++)
        {
            for (int x = rect_min.x; x < rect_max.x; x++)
            {
                uint64_t key = y * grid.x + x;
                key <<= 32;
                key |= *((uint32_t*)&depths[idx]);
                gaussian_keys_unsorted[off] = key;
                gaussian_values_unsorted[off] = idx;
                off++;
            }
        }
    }
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= L)
        return;

    // Read tile ID from key. Update start/end of tile range if at limit.
    uint64_t key = point_list_keys[idx];
    uint32_t currtile = key >> 32;
    if (idx == 0)
        ranges[currtile].x = 0;
    else
    {
        uint32_t prevtile = point_list_keys[idx - 1] >> 32;
        if (currtile != prevtile)
        {
            ranges[prevtile].y = idx;
            ranges[currtile].x = idx;
        }
    }
    if (idx == L - 1)
        ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
        const int P,
        const float* means3D,
        const float* viewmatrix,
        const float* projmatrix,
        bool* present) {
    checkFrustum <<<(P + 255) / 256, 256>>> (
        P,
        means3D,
        viewmatrix, projmatrix,
        present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P) {
    GeometryState geom;
    obtain(chunk, geom.z_depths, P, 128);
    obtain(chunk, geom.clamped, P * 3, 128);
    obtain(chunk, geom.internal_radii, P, 128);
    obtain(chunk, geom.means2D, P, 128);
    obtain(chunk, geom.cov3D, P * 6, 128);
    obtain(chunk, geom.conic_opacity, P, 128);
    obtain(chunk, geom.rgb, P * 3, 128);
    obtain(chunk, geom.tiles_touched, P, 128);
    cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
    obtain(chunk, geom.scanning_space, geom.scan_size, 128);
    obtain(chunk, geom.point_offsets, P, 128);
    return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N) {
    ImageState img;
    obtain(chunk, img.accum_alpha, N, 128);
    obtain(chunk, img.n_contrib, N, 128);
    obtain(chunk, img.ranges, N, 128);
    return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P) {
    BinningState binning;
    obtain(chunk, binning.point_list, P, 128);
    obtain(chunk, binning.point_list_unsorted, P, 128);
    obtain(chunk, binning.point_list_keys, P, 128);
    obtain(chunk, binning.point_list_keys_unsorted, P, 128);
    cub::DeviceRadixSort::SortPairs(
        nullptr, binning.sorting_size,
        binning.point_list_keys_unsorted, binning.point_list_keys,
        binning.point_list_unsorted, binning.point_list, P);
    obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
    return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
        std::function<char* (size_t)> geometryBuffer,
        std::function<char* (size_t)> binningBuffer,
        std::function<char* (size_t)> imageBuffer,
        const int P, const int D, const int M,
        const float* background,
        const int width, const int height,
        const float* means3D,
        const float* shs,
        const float* colors_precomp,
        const float* opacities,
        const float* scales,
        const float scale_modifier,
        const float* rotations,
        const float* cov3D_precomp,
        const float* features,
        const float* viewmatrix,
        const float* projmatrix,
        const float* cam_pos,
        const float tan_fovx,
        const float tan_fovy,
        const bool prefiltered,
        const int featureCount,
        float* out_colors,
        int* out_radii,
        int* out_observe,
        float* out_buffer) {
    const float focal_y = height / (2.0f * tan_fovy);
    const float focal_x = width / (2.0f * tan_fovx);

    size_t chunk_size = required<GeometryState>(P);
    char* chunkptr = geometryBuffer(chunk_size);
    GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

    if (out_radii == nullptr) {
        out_radii = geomState.internal_radii;
    }

    dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);

    // Dynamically resize image-based auxiliary buffers during training
    size_t img_chunk_size = required<ImageState>(width * height);
    char* img_chunkptr = imageBuffer(img_chunk_size);
    ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

    if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
    {
        throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
    }

    // Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
    FORWARD::preprocess(
        P, D, M,
        means3D,
        (glm::vec3*)scales,
        scale_modifier,
        (glm::vec4*)rotations,
        opacities,
        shs,
        geomState.clamped,
        cov3D_precomp,
        colors_precomp,
        viewmatrix, projmatrix,
        (glm::vec3*)cam_pos,
        width, height,
        focal_x, focal_y,
        tan_fovx, tan_fovy,
        out_radii,
        geomState.means2D,
        geomState.z_depths,
        geomState.cov3D,
        geomState.rgb,
        geomState.conic_opacity,
        tile_grid,
        geomState.tiles_touched,
        prefiltered);

    // Compute prefix sum over full list of touched tile counts by Gaussians
    // E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
    cub::DeviceScan::InclusiveSum(
        geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P);

    // Retrieve total number of Gaussian instances to launch and resize aux buffers
    int num_rendered;
    cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost);

    size_t binning_chunk_size = required<BinningState>(num_rendered);
    char* binning_chunkptr = binningBuffer(binning_chunk_size);
    BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

    // For each instance to be rendered, produce adequate [ tile | depth ] key 
    // and corresponding dublicated Gaussian indices to be sorted
    duplicateWithKeys <<<(P + 255) / 256, 256>>> (
        P,
        geomState.means2D,
        geomState.z_depths,
        geomState.point_offsets,
        binningState.point_list_keys_unsorted,
        binningState.point_list_unsorted,
        out_radii,
        tile_grid);

    int bit = getHigherMsb(tile_grid.x * tile_grid.y);

    // Sort complete list of (duplicated) Gaussian indices by keys
    cub::DeviceRadixSort::SortPairs(
        binningState.list_sorting_space,
        binningState.sorting_size,
        binningState.point_list_keys_unsorted, binningState.point_list_keys,
        binningState.point_list_unsorted, binningState.point_list,
        num_rendered, 0, 32 + bit);

    cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2));

    // Identify start and end of per-tile workloads in sorted list
    if (num_rendered > 0)
        identifyTileRanges <<<(num_rendered + 255) / 256, 256>>> (
            num_rendered,
            binningState.point_list_keys,
            imgState.ranges);

    // Let each tile blend its range of Gaussians independently in parallel
    // Use precomputed colors if provided, otherwise use RGBs computed from SHs.
    const float* precomputed_colors = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
    FORWARD::render(
        tile_grid, block,
        imgState.ranges,
        binningState.point_list,
        width, height,
        viewmatrix,
        cam_pos,
        geomState.means2D,
        precomputed_colors,
        features,
        geomState.conic_opacity,
        imgState.accum_alpha,
        imgState.n_contrib,
        background,
        featureCount,
        out_colors,
        out_observe,
        out_buffer);

    return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
        const int P, const int D, const int M, const int R,
        const float* background,
        const int width, const int height,
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
        char* img_buffer,
        const int featureCount,
        const float* grad_colors,
        const float* grad_buffer,
        float* dL_dmeans2D,
        float* dL_dconics,
        float* dL_dopacities,
        float* dL_dcolors,
        float* dL_dmeans3D,
        float* dL_dcov3D,
        float* dL_dshs,
        float* dL_dscales,
        float* dL_drots,
        float* dL_dfeatures) {
    GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
    BinningState binningState = BinningState::fromChunk(binning_buffer, R);
    ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

    if (radii == nullptr) {
        radii = geomState.internal_radii;
    }

    const float focal_y = height / (2.0f * tan_fovy);
    const float focal_x = width / (2.0f * tan_fovx);

    const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    const dim3 block(BLOCK_X, BLOCK_Y, 1);

    // Compute loss gradients w.r.t. 2D mean position, conic matrix,
    // opacity and RGB of Gaussians from per-pixel loss gradients.
    // If we were given precomputed colors and not SHs, use them.
    const float* precomputed_colors = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
    BACKWARD::render(
        tile_grid,
        block,
        imgState.ranges,
        binningState.point_list,
        width, height,
        focal_x, focal_y,
        background,
        featureCount,
        geomState.means2D,
        geomState.conic_opacity,
        precomputed_colors,
        features,
        imgState.accum_alpha,
        imgState.n_contrib,
        buffer,
        grad_colors,
        grad_buffer,
        (float4*)dL_dmeans2D,
        (float4*)dL_dconics,
        dL_dopacities,
        dL_dcolors,
        dL_dfeatures);

    // Take care of the rest of preprocessing. Was the precomputed covariance
    // given to us or a scales/rot pair? If precomputed, pass that. If not,
    // use the one we computed ourselves.
    const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
    BACKWARD::preprocess(
        P, D, M,
        (float3*)means3D,
        radii,
        shs,
        geomState.clamped,
        (glm::vec3*)scales,
        (glm::vec4*)rotations,
        scale_modifier,
        cov3D_ptr,
        viewmatrix,
        projmatrix,
        focal_x, focal_y,
        tan_fovx, tan_fovy,
        (glm::vec3*)campos,
        (float4*)dL_dmeans2D,
        dL_dconics,
        dL_dcolors,
        (glm::vec3*)dL_dmeans3D,
        dL_dcov3D,
        dL_dshs,
        (glm::vec3*)dL_dscales,
        (glm::vec4*)dL_drots);
}