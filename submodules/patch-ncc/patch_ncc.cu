#include <tuple>
#include "cuda_patch_ncc/patch_ncc_impl.h"
#include "patch_ncc.h"

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
    const float cx_n, const float cy_n)
{
    if (normals.ndimension() != 2 || normals.size(1) != 3) {
        AT_ERROR("normals must have dimensions (num_points, 3)");
    }
    const int P     = depths.size(0);
    auto float_opts = depths.options().dtype(torch::kFloat32);

    torch::Tensor ncc          = torch::zeros({P}, float_opts);
    torch::Tensor grad_depths  = torch::zeros_like(depths, float_opts);
    torch::Tensor grad_normals = torch::zeros({P, 3}, float_opts);
    torch::Tensor valid        = torch::zeros({P}, depths.options().dtype(torch::kBool));
    const int image_height_r   = image_r.size(0);
    const int image_width_r    = image_r.size(1);
    const int image_height_n   = image_n.size(0);
    const int image_width_n    = image_n.size(1);

    // Simultaneously calculate output and gradients to eliminate the time-consumption of repeatly fetching data from the global memory.
    forward_mode_differentiation(
        P,
        depths.contiguous().data_ptr<float>(),
        normals.contiguous().data_ptr<float>(),
        uvs.contiguous().data_ptr<int>(),
        R.contiguous().data_ptr<float>(),
        T.contiguous().data_ptr<float>(),
        image_r.contiguous().data_ptr<float>(),
        image_n.contiguous().data_ptr<float>(),
        fx_r, fy_r, cx_r, cy_r,
        fx_n, fy_n, cx_n, cy_n,
        image_height_r, image_width_r,
        image_height_n, image_width_n,
        ncc.contiguous().data_ptr<float>(),
        grad_depths.contiguous().data_ptr<float>(),
        grad_normals.contiguous().data_ptr<float>(),
        valid.contiguous().data_ptr<bool>());
    return std::make_tuple(ncc, grad_depths, grad_normals, valid);
}
