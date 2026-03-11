import sys
import os
sys.path += [os.path.dirname(os.path.abspath(__file__))]
from util import *

import torch

from fused_bilagrid_cuda import (
    bilagrid_sample_forward,
    bilagrid_sample_backward,
    bilagrid_uniform_sample_forward,
    bilagrid_uniform_sample_backward
)


def generic_coords(rgb):
    h, w, _ = rgb.shape[-3:]
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 1.0, h).cuda(),
        torch.linspace(0, 1.0, w).cuda(),
        indexing="ij",
    )
    coords = torch.stack([grid_x, grid_y], dim=-1)
    for i in range(rgb.ndim-coords.ndim-1, -1, -1):
        coords = coords.unsqueeze(0).repeat(rgb.shape[i], *([1]*coords.ndim))
    return coords


def bilagrid_sample_torch(bilagrid, coords, rgb):

    if coords is None:
        coords = generic_coords(rgb)

    rgb2gray = torch.Tensor([[0.299, 0.587, 0.114]]).T.to(rgb)
    coords = torch.cat([coords, rgb @ rgb2gray], dim=-1).contiguous()

    coords = (coords - 0.5) * 2
    affine_mats = torch.nn.functional.grid_sample(
        bilagrid, coords, mode="bilinear", align_corners=True, padding_mode="border"
    )  # (N, 12, m, h, w)
    affine_mats = affine_mats.permute(0, 2, 3, 4, 1)  # (N, m, h, w, 12)
    affine_mats = affine_mats.reshape(*affine_mats.shape[:-1], 3, 4)  # (N, m, h, w, 3, 4)
    return torch.matmul(affine_mats[..., :3], rgb.unsqueeze(-1)).squeeze(-1) + affine_mats[..., 3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_bilagrid_sample():

    N, m = 3, 2
    L, H, W = 5, 7, 15
    h, w = 234, 567

    torch.random.manual_seed(42)

    bilagrid = torch.randn((N, 12, L, H, W)).cuda()
    coords = 0.5+0.5*torch.randn((N, m, h, w, 2)).cuda()
    rgb = 0.5+0.5*torch.randn((N, m, h, w, 3)).cuda()

    bilagrid = torch.nn.Parameter(bilagrid)
    coords = torch.nn.Parameter(coords)
    rgb = torch.nn.Parameter(rgb)

    output = bilagrid_sample_torch(bilagrid, coords, rgb)
    output.retain_grad()
    output.requires_grad_(True)

    print("# Test sample forward")
    output1 = bilagrid_sample_forward(bilagrid, coords, rgb)
    assert_close(output1, output, 1e-5, "output")
    print()

    weights = torch.randn_like(output)
    loss = (weights*output).mean()
    loss.backward()

    print("# Test sample backward")
    v_bilagrid, v_coord, v_rgb = bilagrid_sample_backward(bilagrid, coords, rgb, output.grad, False)
    assert_close(v_bilagrid, bilagrid.grad, 1e-8, "bilagrid.grad")
    assert v_coord is None
    assert_close(v_rgb, rgb.grad, 1e-8, "rgb.grad")
    print()

    print("# Test sample backward, with coords.grad")
    v_bilagrid, v_coord, v_rgb = bilagrid_sample_backward(bilagrid, coords, rgb, output.grad, True)
    assert_close(v_bilagrid, bilagrid.grad, 1e-8, "bilagrid.grad")
    assert_close(v_coord, coords.grad, 1e-8, "coords.grad")
    assert_close(v_rgb, rgb.grad, 1e-8, "rgb.grad")
    print()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_bilagrid_uniform_sample():

    # N, m = 1, 1
    # L, H, W = 4, 8, 8
    # h, w = 45, 59

    N, m = 3, 2
    L, H, W = 5, 7, 15
    h, w = 234, 567

    torch.random.manual_seed(42)

    bilagrid = torch.randn((N, 12, L, H, W)).cuda()
    rgb = 0.5+0.5*torch.randn((N, m, h, w, 3)).cuda()

    bilagrid = torch.nn.Parameter(bilagrid)
    rgb = torch.nn.Parameter(rgb)

    output = bilagrid_sample_torch(bilagrid, None, rgb)
    output.retain_grad()
    output.requires_grad_(True)

    print("# Test uniform sample forward")
    output1 = bilagrid_uniform_sample_forward(bilagrid, rgb)
    assert_close(output1, output, 1e-4, "output")
    print()

    weights = torch.randn_like(output)
    loss = (weights*output).mean()
    loss.backward()

    print("# Test uniform sample backward (v1)")
    v_bilagrid, v_rgb = bilagrid_uniform_sample_backward(bilagrid, rgb, output.grad, 1, 8, 8, 5)
    assert_close(v_bilagrid, bilagrid.grad, 1e-8, "bilagrid.grad")
    assert_close(v_rgb, rgb.grad, 1e-8, "rgb.grad")
    print()

    print("# Test uniform sample backward (v2)")
    v_bilagrid, v_rgb = bilagrid_uniform_sample_backward(bilagrid, rgb, output.grad, 2, 8, 8, 5)
    assert_close(v_bilagrid, bilagrid.grad, 1e-8, "bilagrid.grad")
    assert_close(v_rgb, rgb.grad, 1e-8, "rgb.grad")
    print()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def profile_bilagrid_sample():

    N, m = 1, 1
    L, H, W = 8, 16, 16
    h, w = 1080, 1440

    torch.random.manual_seed(42)

    print("# Profile sample")
    print()

    bilagrid = torch.randn((N, 12, L, H, W)).cuda()
    coords = torch.randn((N, m, h, w, 2)).cuda()
    rgb = torch.randn((N, m, h, w, 3)).cuda()

    timeits([
        (lambda: bilagrid_sample_torch(bilagrid, coords, rgb), "torch forward"),
        (lambda: bilagrid_sample_forward(bilagrid, coords, rgb), "fused forward"),
    ])
    print()

    bilagrid = torch.nn.Parameter(bilagrid)
    coords = torch.nn.Parameter(coords)
    rgb = torch.nn.Parameter(rgb)

    output = bilagrid_sample_torch(bilagrid, coords, rgb)
    output.retain_grad()
    output.requires_grad_(True)

    weight = torch.randn_like(output)
    loss = (weight*output).mean()

    timeits([
        (lambda: loss.backward(retain_graph=True), "torch backward"),
        (lambda: bilagrid_sample_backward(bilagrid, coords, rgb, output.grad, False), "fused backward"),
        (lambda: bilagrid_sample_backward(bilagrid, coords, rgb, output.grad, True), "fused backward with coords.grad"),
    ])
    print()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def profile_uniform_bilagrid_sample():

    N, m = 1, 1
    L, H, W = 8, 16, 16
    # L, H, W = 4, 8, 8
    # L, H, W = 3, 6, 6
    h, w = 1080, 1440
    # h, w = 3000, 4000

    torch.random.manual_seed(42)

    print("# Profile uniform sample")
    print()

    bilagrid = torch.randn((N, 12, L, H, W)).cuda()
    rgb = torch.randn((N, m, h, w, 3)).cuda()

    timeits([
        (lambda: bilagrid_sample_torch(bilagrid, None, rgb), "torch forward"),
        (lambda: bilagrid_uniform_sample_forward(bilagrid, rgb), "fused forward"),
    ])
    print()

    bilagrid = torch.nn.Parameter(bilagrid)
    rgb = torch.nn.Parameter(rgb)

    output = bilagrid_sample_torch(bilagrid, None, rgb)
    output.retain_grad()
    output.requires_grad_(True)

    weight = torch.randn_like(output)
    loss = (weight*output).mean()

    timeits([
        (lambda: loss.backward(retain_graph=True), "torch backward"),
        (lambda: bilagrid_uniform_sample_backward(bilagrid, rgb, output.grad, 1, 8, 8, 5), "fused backward v1"),
        # (lambda: bilagrid_uniform_sample_backward(bilagrid, rgb, output.grad, 1, 8, 4, 4), "fused backward v1"),
        (lambda: bilagrid_uniform_sample_backward(bilagrid, rgb, output.grad, 2, 8, 8, 5), "fused backward v2"),
    ])


if __name__ == "__main__":

    test_bilagrid_sample()
    test_bilagrid_uniform_sample()
    print()

    profile_bilagrid_sample()
    profile_uniform_bilagrid_sample()
