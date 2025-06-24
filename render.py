#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import json
import torch
import torchvision

import numpy as np
import torch.nn.functional as F
from pbr import pbr_shading

from tqdm import tqdm
from pathlib import Path

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args

from scene import Scene
from gaussian_renderer import GaussianModel
from gaussian_renderer import render

from utils.general_utils import safe_state
from utils.image_utils import convert_normal_for_save, save_depth_map
from utils.mesh_utils import fuse_depths, write_mesh, post_process_mesh

def render_views(model, split, iteration, views, scene, pipeline, background, args, bounds=None):
    if (len(views) == 0):
        print(f"[!] No views to render in {split} set")
        return

    model_dir = Path(model.model_path)
    rgb_render_dir = model_dir / split / f"{args.label}_{iteration}" / "rgb_renders"
    pbr_render_dir = model_dir / split / f"{args.label}_{iteration}" / "pbr_renders"
    gt_dir     = model_dir / split / f"{args.label}_{iteration}" / "gts"
    normal_dir = model_dir / split / f"{args.label}_{iteration}" / "normals"
    depth_dir  = model_dir / split / f"{args.label}_{iteration}" / "depths"

    os.makedirs(rgb_render_dir, exist_ok=True)
    os.makedirs(pbr_render_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    scene.cubemap.build_mips()
    canonical_rays = scene.get_canonical_rays()

    # Environment map
    envmap = scene.cubemap.export_envmap(return_img=True).permute(2, 0, 1).clamp(0.0, 1.0) # (3, H, W)
    torchvision.utils.save_image(envmap, model_dir / split / f"{args.label}_{iteration}" / "envmap.png")

    fusion_depths = []
    for view in tqdm(views, desc="[>] Rendering", ncols=80):
        render_pkg = render(
            view, scene.gaussians, pipeline, background, material_stage=True,
            sobel_normal=args.filter_depth, inference=True, pad_normal=True)
        image_stem = view.image_name.rsplit('.', 1)[0]

        # Render and GT
        rgb_render = torch.clamp(render_pkg["render"], 0.0, 1.0)
        gt_image = torch.clamp(view.gt_image[0:3, :, :], 0.0, 1.0)

        torchvision.utils.save_image(rgb_render, rgb_render_dir / f"{image_stem}.png")
        torchvision.utils.save_image(gt_image, gt_dir / f"{image_stem}.png")

        # Normals
        normal = convert_normal_for_save(render_pkg["normal_map"], view).cpu() # (3, H, W)
        torchvision.utils.save_image(normal, normal_dir / f"{image_stem}.png")

        # Depth
        depth = render_pkg["depth_map"].squeeze() # (H, W)
        save_depth_map(depth.cpu().numpy(), depth_dir / f"{image_stem}.png")

        tsdf_depth = depth.clone()
        if args.filter_depth:
            view_dirs = F.normalize(view.get_rays(), p=2, dim=-1)
            sobel_map = render_pkg["sobel_normal_map"].permute(1,2,0) # (H, W, 3)
            sobel_map = F.normalize(sobel_map, p=2, dim=-1) # (H, W, 3)
            dots = torch.sum(view_dirs * sobel_map, dim=-1) # (H, W)
            angles = torch.acos(dots.abs())
            remove = angles > (100.0 / 180 * 3.14159)
            tsdf_depth[remove] = 0.0
        fusion_depths.append(tsdf_depth.cpu())

        # PBR rendering
        H, W = view.image_height, view.image_width
        c2w = view.world_view_transform[:3, :3]
        view_dirs = -(F.normalize(canonical_rays, p=2, dim=-1) @ c2w.T).reshape(H, W, 3)

        normals = render_pkg["normal_map"].permute(1, 2, 0).reshape(-1, 3) # (H * W, 3)
        normals = normals @ c2w.T # (H * W, 3)
        normal_map = normals.reshape(H, W, 3).permute(2, 0, 1) # (3, H, W)
        normal_map = torch.where(
            torch.norm(normal_map, dim=0, keepdim=True) > 0,
            F.normalize(normal_map, dim=0, p=2), normal_map)

        normal_mask = render_pkg["normal_mask"] # (1, H, W)

        albedo_map = render_pkg["albedo_map"] # (3, H, W)
        metallic_map = render_pkg["metallic_map"] # (1, H, W)
        roughness_map = render_pkg["roughness_map"] # (1, H, W)

        pbr_pkg = pbr_shading(
            light=scene.cubemap,
            normals=normal_map.permute(1, 2, 0), # (H, W, 3)
            view_dirs=view_dirs,
            mask=normal_mask.permute(1, 2, 0), # (H, W, 1)
            albedo=albedo_map.permute(1, 2, 0), # (H, W, 3)
            roughness=roughness_map.permute(1, 2, 0), # (H, W, 1)
            metallic=metallic_map.permute(1, 2, 0) if model.metallic else None, # (H, W, 1)
            occlusion=torch.ones_like(roughness_map).permute(1, 2, 0),
            irradiance=torch.zeros_like(roughness_map).permute(1, 2, 0),
            brdf_lut=scene.brdf_lut,
            gamma=model.gamma)

        pbr_render = pbr_pkg["render_rgb"].clamp(0.0, 1.0).permute(2, 0, 1) # (3, H, W)
        pbr_render = torch.where(normal_mask, pbr_render, background[:, None, None])
        torchvision.utils.save_image(pbr_render, pbr_render_dir / f"{image_stem}.png")

    if args.extract_mesh:
        mesh_dir = model_dir / split / f"{args.label}_{iteration}" / "meshes"
        os.makedirs(mesh_dir, exist_ok=True)

        max_depth = args.max_depth if args.max_depth > 0 else 2.0 * scene.cameras_extent
        voxel_size = args.voxel_size if args.voxel_size > 0 else max_depth / 1024.0
        sdf_trunc = args.sdf_trunc if args.sdf_trunc > 0 else 4.0 * voxel_size

        config = {
            "max_depth": max_depth,
            "voxel_size": voxel_size,
            "sdf_trunc": sdf_trunc,
        }
        with open(mesh_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4)

        tsdf_depths = torch.stack(fusion_depths, dim=0) # (N, H, W)
        tsdf_rgb_dir = rgb_render_dir if args.rgb_color else pbr_render_dir
        volume = fuse_depths(tsdf_depths, views, tsdf_rgb_dir, max_depth, voxel_size, sdf_trunc, bounds)

        # Raw mesh
        print(f"[>] Extracting mesh from TSDF volume...")
        mesh = volume.extract_triangle_mesh()
        write_mesh(str(mesh_dir / f"tsdf_mesh.ply"), mesh)
        print(f"[>] Mesh written to: {mesh_dir / f'tsdf_mesh.ply'}")

        # Post-process mesh
        post = post_process_mesh(mesh, args.num_clusters)
        print(f"[>] Num vertices mesh: {len(mesh.vertices)}")
        print(f"[>] Num vertices post: {len(post.vertices)}")
        write_mesh(str(mesh_dir / f"tsdf_post.ply"), post)
        print(f"[>] Post-processed mesh written to: {mesh_dir / f'tsdf_post.ply'}")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Render script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--label", default="ours", type=str)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--extract_mesh", action="store_true")
    parser.add_argument("--max_depth", default=-1, type=float)
    parser.add_argument("--voxel_size", default=-1, type=float)
    parser.add_argument("--sdf_trunc", default=-1, type=float)
    parser.add_argument("--num_clusters", default=1, type=int)
    parser.add_argument("--filter_depth", action="store_true", help="Filter depth maps before TSDF fusion")
    parser.add_argument("--rgb_color", action="store_true", help="Use RGB renderings for mesh colors, default to PBR")
    parser.add_argument("--dtu", action="store_true", help="Tailor the rendering for the DTU dataset")
    parser.add_argument("--tnt", action="store_true", help="Tailor the rendering for the TnT dataset")

    args = get_combined_args(parser)
    print(f"[>] Rendering {args.model_path}")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    model = model.extract(args)
    pipeline = pipeline.extract(args)
    bounds = None

    if args.dtu and args.tnt:
        raise ValueError("[!] Cannot set both --dtu and --tnt flags at the same time. Choose one.")

    if args.dtu:
        args.max_depth = 5.0
        args.voxel_size = 0.002
        args.sdf_trunc = 4.0 * args.voxel_size
        args.num_clusters = 1
        args.filter_depth = False
        args.extract_mesh = True
        args.skip_test = True
        args.rgb_color = True

    if args.tnt:
        tnt_360_scenes = ['barn', 'caterpillar', 'ignatius', 'truck']
        tnt_scene = Path(args.model_path).name.lower()
        args.max_depth = 3.0 if tnt_scene in tnt_360_scenes else 4.5
        print(f"[>] Using max_depth {args.max_depth} for TnT scene {tnt_scene}")

        args.num_clusters = 1
        args.filter_depth = True
        args.extract_mesh = True
        args.skip_test = True
        args.rgb_color = True

        voxel_size = 0.002

        transform_file = Path(args.source_path) / "transforms.json"
        if transform_file.exists():
            with open(transform_file, "r") as f:
                transforms = json.load(f)
            if "aabb_range" in transforms:
                bounds = (np.array(transforms["aabb_range"]))
            else:
                print("[!] No aabb_range found in transforms.json, using default 0.002")
        else:
            print("[!] No transforms.json found, using default voxel size 0.002")
        if bounds is not None:
            max_dis = np.max(bounds[:, 1] - bounds[:, 0])
            voxel_size = max_dis / 2048
            print(f"[>] Using voxel size based on bounding box: {voxel_size:4f}")

        args.voxel_size = voxel_size
        args.sdf_trunc = 4.0 * voxel_size

    with torch.no_grad():
        gaussians = GaussianModel(model.sh_degree)
        scene = Scene(model, gaussians, load_iteration=args.iteration, shuffle=False)
        scene.cubemap.eval()

        bg_color = [1, 1, 1] if model.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not args.skip_train:
            trainCameras = scene.getTrainCameras()
            render_views(model, "train", scene.loaded_iter, trainCameras, scene, pipeline, background, args, bounds)

        if not args.skip_test:
            testCameras = scene.getTestCameras()
            render_views(model, "test", scene.loaded_iter, testCameras, scene, pipeline, background, args)