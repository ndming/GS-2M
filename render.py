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
from pbr import pbr_render, linear_to_srgb

from tqdm import tqdm
from pathlib import Path

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args

from scene import Scene
from gaussian_renderer import GaussianModel
from gaussian_renderer import render

from utils.general_utils import safe_state
from utils.image_utils import convert_normal_for_save, save_depth_map, map_to_rgba
from utils.mesh_utils import fuse_depths, write_mesh, post_process_mesh

def render_views(model, split, iteration, views, scene, pipeline, background, args, bounds=None):
    if (len(views) == 0):
        print(f"[!] No views to render in {split} set")
        return

    model_dir = Path(model.model_path)
    render_dir = model_dir / split / f"{args.label}_{iteration}" / "render"
    gt_dir     = model_dir / split / f"{args.label}_{iteration}" / "gt"
    normal_dir = model_dir / split / f"{args.label}_{iteration}" / "normal"
    depth_dir  = model_dir / split / f"{args.label}_{iteration}" / "depth"

    os.makedirs(render_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    scene.cubemap.build_mips()
    canonical_rays = scene.get_canonical_rays()

    # Environment map
    if model.material:
        envmap = scene.cubemap.export_envmap(return_img=True).permute(2, 0, 1).clamp(0.0, 1.0) # (3, H, W)
        torchvision.utils.save_image(envmap, model_dir / split / f"{args.label}_{iteration}" / "envmap.png")

    # Save number of points
    points = {}
    point_file = model_dir / "points.json"
    if point_file.exists():
        with open(point_file, "r") as f:
            points = json.load(f)
    points[f"{args.label}_{iteration}"] = scene.gaussians.get_xyz.shape[0]
    with open(point_file, "w") as f:
        json.dump(points, f, indent=4)

    fusion_depths = []
    for view in tqdm(views, desc="[>] Rendering", ncols=80):
        render_pkg = render(view, scene.gaussians, pipeline, background, material_stage=True, sobel_normal=args.filter_depth)
        image_stem = view.image_name.rsplit('.', 1)[0]

        # GT image
        gt_image = torch.clamp(view.gt_image[0:3, :, :], 0.0, 1.0)
        if model.white_background:
            gt_image = torch.where(view.alpha_mask > 0.5, gt_image, background[:, None, None])
        torchvision.utils.save_image(gt_image, gt_dir / f"{image_stem}.png")

        # Normals
        normal = convert_normal_for_save(render_pkg["normal_map"], view, args.normal_world).cpu() # (3, H, W)
        if model.white_background:
            map_to_rgba(normal, view.alpha_mask).save(normal_dir / f"{image_stem}.png")
        else:
            torchvision.utils.save_image(normal, normal_dir / f"{image_stem}.png")

        # Depth
        depth = render_pkg["depth_map"].squeeze() # (H, W)
        save_depth_map(depth.cpu().numpy(), depth_dir / f"{image_stem}.png")

        tsdf_depth = depth.clone()
        if args.filter_depth:
            view_dirs = F.normalize(view.get_rays(), p=2, dim=-1)
            sobel_map = render_pkg["sobel_map"].permute(1,2,0) # (H, W, 3)
            sobel_map = F.normalize(sobel_map, p=2, dim=-1) # (H, W, 3)
            dots = torch.sum(view_dirs * sobel_map, dim=-1) # (H, W)
            angles = torch.acos(dots.abs())
            remove = angles > (100.0 / 180 * 3.14159)
            tsdf_depth[remove] = 0.0
        fusion_depths.append(tsdf_depth.cpu())

        if not model.material:
            rgb_image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            torchvision.utils.save_image(rgb_image, render_dir / f"{image_stem}.png")
        else:
            # PBR render
            pbr_pkg = pbr_render(scene, view, canonical_rays, render_pkg, model.metallic, model.gamma)
            pbr_image = pbr_pkg["render_rgb"].clamp(0.0, 1.0).permute(2, 0, 1) # (3, H, W)
            pbr_mask = view.alpha_mask.cuda() > 0.5 if model.mask_gt or model.white_background else pbr_pkg["normal_mask"]
            bg_color = 0.0 if model.mask_gt else background[:, None, None]
            pbr_image = torch.where(pbr_mask, pbr_image, bg_color)
            torchvision.utils.save_image(pbr_image, render_dir / f"{image_stem}.png")

            # BRDF maps
            albedo_map = render_pkg["albedo_map"].clamp(0.0, 1.0) # (3, H, W)
            roughness_map = pbr_pkg["roughness_map"] # (1, H, W)
            metallic_map = pbr_pkg["metallic_map"] # (1, H, W)

            # Component maps
            diffuse_map = linear_to_srgb(pbr_pkg["diffuse_rgb"]) if model.gamma else pbr_pkg["diffuse_rgb"]
            diffuse_map = diffuse_map.clamp(0.0, 1.0).permute(2, 0, 1) # (3, H, W)
            specular_map = linear_to_srgb(pbr_pkg["specular_rgb"]) if model.gamma else pbr_pkg["specular_rgb"]
            specular_map = specular_map.clamp(0.0, 1.0).permute(2, 0, 1) # (3, H, W)

            albedo_dir = model_dir / split / f"{args.label}_{iteration}" / "albedo"
            roughness_dir = model_dir / split / f"{args.label}_{iteration}" / "roughness"
            metallic_dir = model_dir / split / f"{args.label}_{iteration}" / "metallic"
            diffuse_dir = model_dir / split / f"{args.label}_{iteration}" / "diffuse"
            specular_dir = model_dir / split / f"{args.label}_{iteration}" / "specular"

            os.makedirs(albedo_dir, exist_ok=True)
            os.makedirs(roughness_dir, exist_ok=True)
            os.makedirs(metallic_dir, exist_ok=True)

            if model.white_background:
                map_to_rgba(albedo_map, view.alpha_mask).save(albedo_dir / f"{image_stem}.png")
                map_to_rgba(roughness_map, view.alpha_mask).save(roughness_dir / f"{image_stem}.png")
                map_to_rgba(metallic_map, view.alpha_mask).save(metallic_dir / f"{image_stem}.png")
                map_to_rgba(diffuse_map, view.alpha_mask).save(diffuse_dir / f"{image_stem}.png")
                map_to_rgba(specular_map, view.alpha_mask).save(specular_dir / f"{image_stem}.png")
            else:
                torchvision.utils.save_image(albedo_map, albedo_dir / f"{image_stem}.png")
                torchvision.utils.save_image(roughness_map, roughness_dir / f"{image_stem}.png")
                torchvision.utils.save_image(metallic_map, metallic_dir / f"{image_stem}.png")
                torchvision.utils.save_image(diffuse_map, diffuse_dir / f"{image_stem}.png")
                torchvision.utils.save_image(specular_map, specular_dir / f"{image_stem}.png")

    if args.extract_mesh:
        mesh_dir = model_dir / split / f"{args.label}_{iteration}" / "mesh"
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
        volume = fuse_depths(tsdf_depths, views, render_dir, max_depth, voxel_size, sdf_trunc, bounds)

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
    parser.add_argument("--dtu", action="store_true", help="Tailor the rendering for the DTU dataset")
    parser.add_argument("--tnt", action="store_true", help="Tailor the rendering for the TnT dataset")
    parser.add_argument("--blender", action="store_true", help="Tailor the rendering for Blender scenes")
    parser.add_argument("--normal_world", action="store_true", help="Save normals in world space, defaults to camera space")

    args = get_combined_args(parser)
    print(f"[>] Rendering {args.model_path}")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    model = model.extract(args)
    pipeline = pipeline.extract(args)
    bounds = None

    if args.dtu and args.tnt and args.blender:
        raise ValueError("[!] Please choose only one of: --dtu, --tnt, or --blender")

    if args.dtu:
        args.max_depth = 5.0
        args.voxel_size = 0.002
        args.sdf_trunc = 4.0 * args.voxel_size
        args.num_clusters = 1
        args.filter_depth = False
        args.extract_mesh = True
        args.skip_test = True

    if args.tnt:
        tnt_360_scenes = ['barn', 'caterpillar', 'ignatius', 'truck']
        tnt_scene = Path(args.model_path).name.lower()
        args.max_depth = 3.0 if tnt_scene in tnt_360_scenes else 4.5
        print(f"[>] Using max_depth {args.max_depth} for TnT scene {tnt_scene}")

        args.num_clusters = 1
        args.filter_depth = True
        args.extract_mesh = True
        args.skip_test = True

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

    if args.blender:
        args.extract_mesh = False
        args.skip_train = True
        args.skip_test = False
        args.normal_world = True

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