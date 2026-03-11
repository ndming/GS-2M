import sys
import os
sys.path += [os.path.dirname(os.path.abspath(__file__))]
from util import *

import torch

from nerfstudio.model_components import lib_bilagrid
import fused_bilagrid

from collections import OrderedDict

from typing import List, Tuple


def profile_sample(grid_size: List[int]):

    print("# Profile sample", grid_size)
    print()

    times = OrderedDict()

    for (h, w) in [
        (600, 600),
        (1080, 1440),
        (3000, 4000)
    ][::-1]:
        torch.cuda.empty_cache()
        print(f"(h, w) = ({h}, {w})")

        idx = torch.tensor([0]).cuda()

        bilagrid0 = lib_bilagrid.BilateralGrid(1, *grid_size).cuda()
        bilagrid1 = fused_bilagrid.BilateralGrid(1, *grid_size).cuda()

        torch.random.manual_seed(42)
        grid_data = torch.randn_like(bilagrid0.grids.data)
        bilagrid0.grids.data = grid_data
        bilagrid1.grids.data = grid_data

        ni = len(idx)
        uv = 0.5 + 0.5 * torch.randn((ni, h, w, 2)).cuda()
        rgb = 0.5 + 0.5 * torch.randn((ni, h, w, 3)).cuda()
        uv0 = torch.nn.Parameter(uv.clone())
        rgb0 = torch.nn.Parameter(rgb.clone())
        uv1 = torch.nn.Parameter(uv.clone())
        rgb1 = torch.nn.Parameter(rgb.clone())

        forward0 = lambda: lib_bilagrid.slice(bilagrid0, uv0, rgb0, idx)['rgb']
        forward1 = lambda: fused_bilagrid.slice(bilagrid1, uv1, rgb1, idx, compute_coords_grad=True)['rgb']

        output0 = forward0()
        output1 = forward1()
        weights = torch.randn_like(output0)
        loss0 = (weights*output0).sum()
        loss1 = (weights*output1).sum()

        backward0 = lambda: loss0.backward(retain_graph=True)
        backward1 = lambda: loss1.backward(retain_graph=True)

        repeat = min(10000//w, 10)
        dt_fwd_torch, dt_fwd_fused, dt_bwd_torch, dt_bwd_fused = timeits([
            (forward0, "forward torch"),
            (forward1, "forward fused"),
            (backward0, "backward torch"),
            (backward1, "backward fused"),
        ], repeat)
        print(f"forward: {dt_fwd_torch/dt_fwd_fused:.1f}x")
        print(f"backward: {dt_bwd_torch/dt_bwd_fused:.1f}x")

        times[f"{w}×{h}"] = [dt_fwd_torch, dt_fwd_fused, dt_bwd_torch, dt_bwd_fused]

        print()

    times = OrderedDict(reversed(times.items()))
    return (f"Irregular Sample {grid_size}", times)


def profile_uniform_sample(grid_size: List[int]):

    print("# Profile uniform sample", grid_size)
    print()

    times = OrderedDict()

    for (h, w) in [
        (600, 600),
        (1080, 1440),
        (3000, 4000)
    ][::-1]:
        torch.cuda.empty_cache()
        print(f"(h, w) = ({h}, {w})")

        idx = torch.tensor([0]).cuda()

        bilagrid0 = lib_bilagrid.BilateralGrid(1, *grid_size).cuda()
        bilagrid1 = fused_bilagrid.BilateralGrid(1, *grid_size).cuda()

        torch.random.manual_seed(42)
        grid_data = torch.randn_like(bilagrid0.grids.data)
        bilagrid0.grids.data = grid_data
        bilagrid1.grids.data = grid_data

        ni = len(idx)
        rgb = 0.5 + 0.5 * torch.randn((ni, h, w, 3)).cuda()
        rgb0 = torch.nn.Parameter(rgb.clone())
        rgb1 = torch.nn.Parameter(rgb.clone())

        def get_uv():
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, 1.0, h).cuda(),
                torch.linspace(0, 1.0, w).cuda(),
                indexing="ij",
            )
            return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(ni, 1, 1, 1)

        forward0 = lambda: lib_bilagrid.slice(bilagrid0, get_uv(), rgb0, idx)['rgb']
        forward1 = lambda: fused_bilagrid.slice(bilagrid1, None, rgb1, idx)['rgb']

        output0 = forward0()
        output1 = forward1()
        weights = torch.randn_like(output0)
        loss0 = (weights*output0).sum()
        loss1 = (weights*output1).sum()

        backward0 = lambda: loss0.backward(retain_graph=True)
        backward1 = lambda: loss1.backward(retain_graph=True)

        repeat = min(10000//w, 10)
        dt_fwd_torch, dt_fwd_fused, dt_bwd_torch, dt_bwd_fused = timeits([
            (forward0, "forward torch"),
            (forward1, "forward fused"),
            (backward0, "backward torch"),
            (backward1, "backward fused"),
        ], repeat)
        print(f"forward: {dt_fwd_torch/dt_fwd_fused:.1f}x")
        print(f"backward: {dt_bwd_torch/dt_bwd_fused:.1f}x")

        times[f"{w}×{h}"] = [dt_fwd_torch, dt_fwd_fused, dt_bwd_torch, dt_bwd_fused]

        print()

    times = OrderedDict(reversed(times.items()))
    return (f"Uniform Sample {grid_size}", times)


def profile_tv_loss():

    print("# Profile total variation loss")
    print()

    W, H, L = 16, 16, 8

    times = OrderedDict()

    for N in [250, 600, 2000][::-1]:
        torch.cuda.empty_cache()
        print(f"N = {N}")

        bilagrid0 = lib_bilagrid.BilateralGrid(N, W, H, L).cuda()
        bilagrid1 = fused_bilagrid.BilateralGrid(N, W, H, L).cuda()

        torch.random.manual_seed(42)
        grid_data = torch.randn_like(bilagrid0.grids.data)
        bilagrid0.grids.data = grid_data
        bilagrid1.grids.data = grid_data

        forward0 = lambda: bilagrid0.tv_loss()
        forward1 = lambda: bilagrid1.tv_loss()

        output0 = forward0()
        output1 = forward1()
        weights = torch.randn_like(output0)
        loss0 = (weights*output0).sum()
        loss1 = (weights*output1).sum()

        backward0 = lambda: loss0.backward(retain_graph=True)
        backward1 = lambda: loss1.backward(retain_graph=True)

        repeat = min(5000//N, 10)
        dt_fwd_torch, dt_fwd_fused, dt_bwd_torch, dt_bwd_fused = timeits([
            (forward0, "forward torch"),
            (forward1, "forward fused"),
            (backward0, "backward torch"),
            (backward1, "backward fused"),
        ], repeat)
        print(f"forward: {dt_fwd_torch/dt_fwd_fused:.1f}x")
        print(f"backward: {dt_bwd_torch/dt_bwd_fused:.1f}x")

        times[f"{N} images"] = [dt_fwd_torch, dt_fwd_fused, dt_bwd_torch, dt_bwd_fused]

        print()

    times = OrderedDict(reversed(times.items()))
    return ("Total Variation Loss", times)



if __name__ == "__main__":

    results = OrderedDict()

    for key, value in [
        profile_sample([16, 16, 8]),
        profile_uniform_sample([16, 16, 8]),
        profile_sample([8, 8, 4]),
        profile_uniform_sample([8, 8, 4]),
        profile_tv_loss()
    ]:
        results[key] = value

    import json
    with open(os.path.join(os.path.dirname(__file__), "timings.json"), 'w') as fp:
        json.dump(results, fp, indent=4)
