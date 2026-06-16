from typing import NamedTuple
import torch
from . import _C


class WarpParams(NamedTuple):
    R: torch.Tensor
    T: torch.Tensor
    fx_r: float
    fy_r: float
    cx_r: float
    cy_r: float
    fx_n: float
    fy_n: float
    cx_n: float
    cy_n: float


def warp_patch_ncc(
    depths: torch.Tensor,
    normals: torch.Tensor,
    uvs: torch.Tensor,
    R: torch.Tensor,
    T: torch.Tensor,
    image_r: torch.Tensor,
    image_n: torch.Tensor,
    fx_r: float,
    fy_r: float,
    cx_r: float,
    cy_r: float,
    fx_n: float,
    fy_n: float,
    cx_n: float,
    cy_n: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    params = WarpParams(
        R.contiguous(),
        T.contiguous(),
        fx_r,
        fy_r,
        cx_r,
        cy_r,
        fx_n,
        fy_n,
        cx_n,
        cy_n,
    )
    ncc, valid = _WarpPatchNCC.apply(depths, normals, uvs, image_r, image_n, params)
    return ncc, valid


class _WarpPatchNCC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, depths, normals, uvs, image_r, image_n, params):
        ncc, grad_depths, grad_normals, valid = _C.warp_patch_ncc(
            depths,
            normals,
            uvs,
            params.R,
            params.T,
            image_r,
            image_n,
            params.fx_r,
            params.fy_r,
            params.cx_r,
            params.cy_r,
            params.fx_n,
            params.fy_n,
            params.cx_n,
            params.cy_n,
        )

        ctx.save_for_backward(grad_depths, grad_normals)

        return ncc, valid

    @staticmethod
    def backward(ctx, grad_ncc, grad_valid):
        grad_depths, grad_normals = ctx.saved_tensors
        return grad_ncc * grad_depths, grad_ncc.unsqueeze(-1) * grad_normals, None, None, None, None
