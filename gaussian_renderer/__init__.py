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

import torch
import math
import torch.nn.functional as F

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel

from utils.sh_utils import eval_sh
from utils.normal_utils import normal_from_depth_image

def render(
        viewpoint_camera,
        pc: GaussianModel,
        pipe,
        bg_color: torch.Tensor,
        geometry_stage=False,
        material_stage=False,
        sobel_normal=False):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # These view-space positional gradients are needed for densification
    tensor_shape = (pc.get_xyz.shape[0], 4) # the fist 2 retain the grads as in 3DGS, while the last 2 track abs grads
    screenspace_points = torch.zeros(tensor_shape, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    albedo = pc.get_albedo
    roughness = pc.get_roughness
    metallic = pc.get_metallic

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance()
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if pipe.convert_SHs_python:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        shs = pc.get_features

     # Blend normals in camera space
    normals = pc.get_normals(viewpoint_camera.camera_center) # normals in world space
    cam_normals = normals @ viewpoint_camera.world_view_transform[:3, :3]
    # Remember that view matrix is stored transposed
    cam_points = means3D @ viewpoint_camera.world_view_transform[:3, :3] + viewpoint_camera.world_view_transform[3, :3]

    feature_count = 10 if material_stage else 5 if geometry_stage else 1
    features = torch.zeros((means3D.shape[0], 10), dtype=torch.float32, device="cuda")
    features[:, 0] = 1.0 # alpha
    features[:, 1] = cam_points[:, 2] if pipe.z_depth else (cam_normals * cam_points).sum(dim=-1).abs() # distance
    features[:, 2:5] = cam_normals
    features[:, 5:8] = albedo
    features[:, 8:9] = roughness
    features[:, 9:10] = metallic

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        feature_count=feature_count)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, observe, buffer, depth = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=opacity,
        shs=shs,
        colors_precomp=colors_precomp,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        features=features)

    normal_map = buffer[2:5, ...] # (3, H, W)
    normal_mask = (normal_map != 0).all(0, keepdim=True).detach()

    # alpha_map = buffer[0:1, ...] # (1, H, W)
    # if pad_normal:
    #     alpha_map = torch.where(alpha_map < 0.004, torch.zeros_like(alpha_map), alpha_map)
    #     alpha_map = torch.where(alpha_map > 1.0 - 0.004, torch.ones_like(alpha_map), alpha_map)
    #     normal_bg = torch.tensor([0.0, 0.0, 0.0], device=normal_map.device)
    #     normal_map = normal_map * alpha_map + (1.0 - alpha_map) * normal_bg[:, None, None]

    out = {
        "render": rendered_image, # (3, H, W)
        "viewspace_points": screenspace_points, # (N, 4)
        "visibility_filter": radii > 0, # (N,)
        "radii": radii, # (N,)
        "observe": observe, # (N,)
        "alpha_map": buffer[0:1, ...], # (1, H, W)
        "distance_map": buffer[1:2, ...] if not pipe.z_depth else None, # (1, H, W)
        "depth_map": buffer[1:2, ...] if pipe.z_depth else depth, # (1, H, W)
        "normal_map": normal_map, # (3, H, W)
        "albedo_map": buffer[5:8, ...], # (3, H, W)
        "roughness_map": buffer[8:9, ...], # (1, H, W)
        "metallic_map": buffer[9:10, ...], # (1, H, W)
        "normal_mask": normal_mask, # (1, H, W)
    }

    if sobel_normal:
        depth_map = out["depth_map"].squeeze(0) # (H, W)
        sobel_map = render_normal_from_depth_map(viewpoint_camera, depth_map, bg_color, out["alpha_map"][0])
        out["sobel_map"] = sobel_map # (3, H, W)

    return out

def render_normal_from_depth_map(viewpoint_cam, depth, bg_color, alpha_map):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic, extrinsic = viewpoint_cam.get_calib_matrix_nerf()
    normal_ref = normal_from_depth_image(depth, intrinsic.to(depth.device), extrinsic.to(depth.device), view_space=True)
    background = bg_color[None, None, ...]
    normal_ref = normal_ref * alpha_map[..., None] + background * (1. - alpha_map[..., None])
    normal_ref = normal_ref.permute(2, 0, 1) # (3, H, W)
    return normal_ref