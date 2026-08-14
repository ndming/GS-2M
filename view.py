import math
import os
import threading
import time
import yaml

import torch
import torch.nn.functional as F
import viser

from argparse import ArgumentParser
from pathlib import Path

from gsplat.distributed import cli
from gsplat.rendering import rasterization

from nerfview import CameraState, RenderTabState, apply_float_colormap
from scene import Config, GsplatViewer, GsplatRenderTabState, get_ckpt_files


def main(local_rank: int, world_rank, world_size: int, args):
    cfg_file = Path(args.cfg_file)
    assert cfg_file.exists(), cfg_file

    with open(cfg_file, "r") as f:
        cfg_dict = yaml.unsafe_load(f)
    cfg = Config(**cfg_dict)

    torch.manual_seed(42)
    device = torch.device("cuda", local_rank)
    distributed = world_size > 1

    ckpt_dir = Path(cfg.result_dir) / "ckpts"
    ckpt_files, ckpt_step = get_ckpt_files(ckpt_dir, args.ckpt_step)

    # Shard the Gaussians across GPUs so each rank only holds a subset of the scene,
    # mirroring how training shards the model. When the number of per-rank checkpoints
    # matches the world size, each rank loads exactly its own shard; otherwise we load
    # everything and take a strided slice.
    if distributed and len(ckpt_files) == world_size:
        my_files = [ckpt_files[world_rank]]
    else:
        my_files = ckpt_files

    def load_splats(files):
        means, quats, scales, opacities, sh0, shN = [], [], [], [], [], []
        for ckpt_path in files:
            ckpt = torch.load(ckpt_path, map_location=device)["splats"]
            means.append(ckpt["means"])
            quats.append(F.normalize(ckpt["quats"], p=2, dim=-1))
            scales.append(torch.exp(ckpt["scales"]))
            opacities.append(torch.sigmoid(ckpt["opacities"]))
            sh0.append(ckpt["sh0"])
            shN.append(ckpt["shN"])
        means = torch.cat(means, dim=0)
        quats = torch.cat(quats, dim=0)
        scales = torch.cat(scales, dim=0)
        opacities = torch.cat(opacities, dim=0)
        colors = torch.cat([torch.cat(sh0, dim=0), torch.cat(shN, dim=0)], dim=-2)
        return means, quats, scales, opacities, colors

    means, quats, scales, opacities, colors = load_splats(my_files)
    if distributed and len(ckpt_files) != world_size:
        sl = slice(world_rank, None, world_size)
        means, quats, scales, opacities, colors = (
            means[sl], quats[sl], scales[sl], opacities[sl], colors[sl]
        )
    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)

    # Total (global) Gaussian count for display, summed across all shards
    if distributed:
        total_gs_tensor = torch.tensor([len(means)], device=device)
        torch.distributed.all_reduce(total_gs_tensor)
        total_gs = int(total_gs_tensor.item())
    else:
        total_gs = len(means)

    if world_rank == 0:
        print(f"[>] Viewing from {len(ckpt_files)} checkpoint(s) at step {ckpt_step}")
        print(f"[>] Number of Gaussians: {total_gs} ({world_size} GPU shard(s))")

    # Collective render shared by rank 0 (server) and worker ranks. In distributed mode
    # this is an all-ranks operation: gsplat gathers each camera's visible Gaussians from
    # every shard, so all ranks must call it in lockstep with the same camera and options.
    def rasterize_view(c2w, K, width, height, opts):
        viewmat = c2w.inverse()
        render_colors, render_alphas, info = rasterization(
            means,  # [N, 3]
            quats,  # [N, 4]
            scales,  # [N, 3]
            opacities,  # [N]
            colors,  # [N, S, 3]
            viewmat[None],  # [1, 4, 4]
            K[None],  # [1, 3, 3]
            width,
            height,
            sh_degree=opts["sh_degree"],
            near_plane=opts["near_plane"],
            far_plane=opts["far_plane"],
            radius_clip=opts["radius_clip"],
            eps2d=opts["eps2d"],
            backgrounds=torch.tensor([opts["backgrounds"]], device=device) / 255.0,
            render_mode=opts["render_mode"],
            rasterize_mode=opts["rasterize_mode"],
            camera_model=opts["camera_model"],
            packed=False,
            distributed=distributed,
            with_ut=cfg.with_ut,
            with_eval3d=cfg.with_eval3d,
        )
        # Count Gaussians projected to a positive radius. In distributed mode each rank
        # sees only its shard, so all-reduce to get the global visible count. This is a
        # collective, matched by every worker's own rasterize_view call.
        rendered_count = (info["radii"] > 0).all(-1).sum()
        if distributed:
            torch.distributed.all_reduce(rendered_count)
        return render_colors, render_alphas, info, int(rendered_count.item())

    # Worker ranks host no viewer: they wait for rank 0 to broadcast a camera, then
    # participate in the collective render (discarding the result) until told to stop.
    if world_rank != 0:
        while True:
            box = [None]
            torch.distributed.broadcast_object_list(box, src=0)
            msg = box[0]
            if not msg["alive"]:
                break
            rasterize_view(
                msg["c2w"].to(device), msg["K"].to(device),
                msg["width"], msg["height"], msg["opts"],
            )
        return

    RENDER_MODE_MAP = {
        "rgb": "RGB",
        "alpha": "RGB",
        "depth": "RGB+PD",
        "normal": "RGB+PD",
    }

    # Serialize renders so per-frame broadcasts from concurrent viser callbacks
    # never interleave (each broadcast must be matched 1:1 by a worker render).
    render_lock = threading.Lock()

    # Register and open viewer
    @torch.no_grad()
    def viewer_render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)

        opts = dict(
            sh_degree=(
                min(render_tab_state.max_sh_degree, sh_degree)
                if sh_degree is not None
                else None
            ),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=render_tab_state.backgrounds,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
        )

        with render_lock:
            if distributed:
                # Wake the worker ranks with the exact camera + options for this frame
                torch.distributed.broadcast_object_list(
                    [dict(
                        alive=True, c2w=c2w.cpu(), K=K.cpu(),
                        width=width, height=height, opts=opts,
                    )],
                    src=0,
                )
            render_colors, render_alphas, info, rendered_count = rasterize_view(
                c2w, K, width, height, opts
            )

        render_tab_state.total_gs_count = total_gs
        render_tab_state.rendered_gs_count = rendered_count

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["ed"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders

    # Where to dump viewer data (e.g., camera poses, rendered images, etc.)
    viewer_dir = Path(cfg.result_dir) / "viewer"
    os.makedirs(viewer_dir, exist_ok=True)

    # Run the viewer server
    server = viser.ViserServer(port=args.port, verbose=False)
    _ = GsplatViewer(
        server=server,
        render_fn=viewer_render_fn,
        output_dir=viewer_dir,
        mode="rendering",
    )
    print("Viewer running... Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down viewer...")
        if distributed:
            # Release the worker ranks from their render loop so they can reach the
            # barrier in gsplat's distributed teardown instead of hanging on broadcast.
            torch.distributed.broadcast_object_list([dict(alive=False)], src=0)
        server.stop()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg_file", type=str, required=True,
        help="Path to the cfg.yml file saved during training"
    )
    parser.add_argument(
        "--ckpt_step", type=int, default=-1,
        help="The specific training step to load checkpoint, default to the latest step if < 0"
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="Port for the viewer server"
    )
    parser.add_argument(
        "--scene_grid", type=int, default=1,
        help="Repeat the scene into a grid of NxN"
    )
    parser.add_argument(
        "--with_ut", action="store_true",
        help="Use unscented transform"
    )
    parser.add_argument(
        "--with_eval3d", action="store_true", help="Use eval 3D"
    )
    args = parser.parse_args()
    assert args.scene_grid % 2 == 1, "scene_grid must be odd"

    cli(main, args, verbose=True)
