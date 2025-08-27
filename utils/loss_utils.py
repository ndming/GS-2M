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

import cv2
import random
import torch

import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable
from math import exp

from gaussian_renderer import render

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = _create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def _create_window(window_size, channel):
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def plane_loss(visibility_filter, gaussians):
    if visibility_filter.sum() == 0:
        return 0.0
    scale = gaussians.get_scaling[visibility_filter]
    sorted_scale, _ = torch.sort(scale, dim=-1)
    min_scale_loss = sorted_scale[..., 0]
    return min_scale_loss.mean()

def sparse_loss(alpha_map):
    zero_epsilon = 1e-3
    val = torch.clamp(alpha_map, zero_epsilon, 1 - zero_epsilon)
    loss = torch.mean(torch.log(val) + torch.log(1 - val))
    return loss

def delta_normal_loss(dn_norm_map, alpha_map):
    # delta_normal_norm: (1, H, W), alpha: (1, H, W)
    device = alpha_map.device
    weight = alpha_map.detach().cpu().numpy()[0]  # (H, W)
    weight = (weight * 255).astype(np.uint8)

    weight = _erode_cv(weight, erode_size=4)  # Output: (H, W), dtype=uint8

    weight = torch.from_numpy(weight.astype(np.float32) / 255.0)  # (H, W)
    weight = weight[None, ...].to(device)  # (1, H, W)

    # Compute weighted mean loss
    w = weight.reshape(-1)
    l = dn_norm_map.reshape(-1)
    loss = (w * l).mean()

    return loss

def _erode_cv(img_in, erode_size=4):
    img_out = np.copy(img_in)
    kernel = np.ones((erode_size, erode_size), np.uint8)
    img_out = cv2.erode(img_out, kernel, iterations=1)

    return img_out

def depth_normal_loss(normal_map, sobel_map, gt_image, weight_map=None):
    # normal_map, sobel_map, gt_image: (3, H, W), weight_map: (1, H, W)
    weights = (1.0 - _get_img_grad_weight(gt_image)).clamp(0, 1).detach() ** 2
    if weight_map is not None:
        weights *= weight_map.squeeze()
    loss = (weights * (sobel_map - normal_map).abs().sum(dim=0)).mean()
    return loss

def _get_img_grad_weight(img):
    _, hd, wd = img.shape 
    bottom_point = img[..., 2:hd,   1:wd-1]
    top_point    = img[..., 0:hd-2, 1:wd-1]
    right_point  = img[..., 1:hd-1, 2:wd]
    left_point   = img[..., 1:hd-1, 0:wd-2]
    grad_img_x = torch.mean(torch.abs(right_point - left_point), 0, keepdim=True)
    grad_img_y = torch.mean(torch.abs(top_point - bottom_point), 0, keepdim=True)
    grad_img = torch.cat((grad_img_x, grad_img_y), dim=0)
    grad_img, _ = torch.max(grad_img, dim=0)
    grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min())
    grad_img = torch.nn.functional.pad(grad_img[None, None], (1, 1, 1, 1), mode='constant', value=0.0).squeeze()
    return grad_img

def alpha_loss(alpha_map, roughness_map, normal_mask):
    mask = (roughness_map < 0.01) & normal_mask
    loss = (1.0 - alpha_map[mask.detach()]).mean() if mask.sum() > 0 else 0.0
    return loss

def roughness_loss(scene, viewpoint_cam, opt, render_pkg, pipe, bg_color):
    if pipe.z_depth:
        return 0.0

    nearby_indices = viewpoint_cam.nearby_indices
    nearby_cam = None if len(nearby_indices) == 0 else scene.getTrainCameras()[random.sample(nearby_indices, 1)[0]]
    if nearby_cam is None:
        return 0.0
    
    with torch.no_grad():
        pts = _get_points_from_depth(viewpoint_cam, render_pkg['depth_map'])
        pts_in_nearby_cam = pts @ nearby_cam.world_view_transform[:3, :3] + nearby_cam.world_view_transform[3, :3]

        nearby_render_pkg = render(
            nearby_cam, scene.gaussians, pipe, bg_color, geometry_stage=True, material_stage=False, sobel_normal=False)
        map_z, _, valid = _sample_depth_normal(pts_in_nearby_cam, nearby_cam, nearby_render_pkg)
        valid = valid & (pts_in_nearby_cam[:, 2] - map_z <= opt.mv_occlusion_threshold)

        ncc_scale = scene.ncc_scale
        mask = valid.reshape(-1)
        valid_indices = torch.arange(mask.shape[0], device=mask.device)[mask]
        if mask.sum() > opt.multi_view_sample_num:
            index = np.random.choice(mask.sum().cpu().numpy(), opt.multi_view_sample_num, replace=False)
            valid_indices = valid_indices[index]

        if valid_indices.sum() == 0:
            return 0.0

        pixels = scene.pixels.reshape(-1, 2)[valid_indices]
        offsets = _patch_offsets(opt.multi_view_patch_size, pixels.device)
        ori_pixels_patch = pixels.reshape(-1, 1, 2) / ncc_scale + offsets.float()

        gt_image_gray = viewpoint_cam.gray_image.cuda() # (1, H, W)
        h, w = gt_image_gray.squeeze().shape
        pixels_patch = ori_pixels_patch.clone()
        pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (w - 1) - 1.0
        pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (h - 1) - 1.0
        ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
        total_patch_size = (opt.multi_view_patch_size * 2 + 1) ** 2
        ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

        rn_R = nearby_cam.world_view_transform[:3, :3].transpose(-1, -2) @ viewpoint_cam.world_view_transform[:3, :3]
        rn_t = -rn_R @ viewpoint_cam.world_view_transform[3, :3] + nearby_cam.world_view_transform[3, :3]
    
        ref_local_n = render_pkg["local_normal_map"].permute(1, 2, 0) # (H, W, 3)
        ref_local_n = ref_local_n.reshape(-1, 3)[valid_indices] # (N, 3)
        ref_local_d = render_pkg["distance_map"].squeeze() # (H, W)
        ref_local_d = ref_local_d.reshape(-1)[valid_indices] # (N,)

        H_rn = rn_R[None] - torch.matmul(
            rn_t[None, :, None].expand(ref_local_d.shape[0], 3, 1),
            ref_local_n[:, :, None].expand(ref_local_d.shape[0], 3, 1).permute(0, 2, 1)) / ref_local_d[..., None, None]
        H_rn = torch.matmul(nearby_cam.get_K(ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3), H_rn)
        H_rn = H_rn @ viewpoint_cam.get_inv_K(ncc_scale)

        grid = _patch_warp(H_rn.reshape(-1, 3, 3), ori_pixels_patch)
        grid[:, :, 0] = 2 * grid[:, :, 0] / (w - 1) - 1.0
        grid[:, :, 1] = 2 * grid[:, :, 1] / (h - 1) - 1.0
        nearest_image_gray = nearby_cam.gray_image.cuda()
        sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
        sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)

        patch_size = opt.multi_view_patch_size * 2 + 1
        ref_grad = _patch_gradient(ref_gray_val, patch_size)
        nea_grad = _patch_gradient(sampled_gray_val, patch_size)

        ncc_grad, _ = _loss_ncc(ref_grad.view(-1, total_patch_size), nea_grad.view(-1, total_patch_size))
        ncc_gray, std_mask = _loss_ncc(ref_gray_val, sampled_gray_val, std_mask=True)
        ncc_error = torch.where(std_mask, ncc_grad, ncc_gray)
        # ncc_error = torch.exp(-ncc_gray * 3.0) * ncc_gray + ncc_grad
        ncc_error = ncc_error.reshape(-1).detach() # (N,) valid_indices
        ncc_error = torch.tanh(8.0 * (ncc_error - opt.reflection_threshold))
        # ncc_error = (1.0 - _sigmoid(ncc_error, 4, 0.4)) * ncc_error.sqrt() + _sigmoid(ncc_error, 4, 0.4) * (ncc_error ** 2.5)

    rough_map = render_pkg["roughness_map"] # (1, H, W)
    h_map, w_map = rough_map.squeeze().shape
    grid_map = pixels.reshape(-1, 1, 2)
    grid_map = grid_map.clone()
    grid_map[:, :, 0] = 2 * grid_map[:, :, 0] / (w_map - 1.0) - 1.0
    grid_map[:, :, 1] = 2 * grid_map[:, :, 1] / (h_map - 1.0) - 1.0
    rough_vals = F.grid_sample(rough_map.unsqueeze(1), grid_map.view(1, -1, 1, 2), align_corners=True)
    rough_vals = rough_vals.squeeze() # (N,) valid_indices

    # consistent_error = ncc_error.detach().pow(0.5)
    # reflection_threshold = opt.reflection_threshold
    increase_mask = (ncc_error < 0.0) & (rough_vals <= 0.8).detach()
    decrease_mask = (ncc_error > 0.0) & (rough_vals > 0.08).detach()

    rough_mask = increase_mask | decrease_mask
    rough_loss = (ncc_error * rough_vals)[rough_mask].mean() if rough_mask.sum() > 0 else 0.0
    # if increase_mask.sum() > 0:
    #     rough_loss -= rough_vals[increase_mask].mean()
    # if decrease_mask.sum() > 0:
    #     rough_loss += rough_vals[decrease_mask].mean()
    return rough_loss

def _patch_gradient(patch, patch_size):
    patch = patch.view(-1, 1, patch_size, patch_size)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=patch.dtype, device=patch.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    grad_x = F.conv2d(patch, sobel_x, padding=1)
    grad_y = F.conv2d(patch, sobel_y, padding=1)
    return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

def _sigmoid(x, k, t):
    return 1.0 / (1.0 + torch.exp(-k * (x - t)))

def multi_view_loss(scene, viewpoint_cam, opt, render_pkg, pipe, bg_color, material_stage):
    train_cams = scene.getTrainCameras()
    nearest_indices = viewpoint_cam.nearest_indices
    nearest_cam = None if len(nearest_indices) == 0 else train_cams[random.sample(nearest_indices, 1)[0]]
    if nearest_cam is None:
        return 0.0

    # Render the depth map and normal map from the nearest camera
    nearest_render_pkg = render(
        nearest_cam, scene.gaussians, pipe, bg_color, geometry_stage=True, material_stage=False, sobel_normal=False)
    # Use depth map to back-project pixel points in reference view to 3D world points
    pts = _get_points_from_depth(viewpoint_cam, render_pkg['depth_map']) # (N, 3)
    # Find the coordinates of those points in neighbor camera's view
    pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3, :3] + nearest_cam.world_view_transform[3, :3]
    # Sample depth values of those points from the depth map rendered from the neighbor camera
    # Also return a mask indicating which points are valid and sampled normals in world coordinates
    map_z, map_n, valid = _sample_depth_normal(pts_in_nearest_cam, nearest_cam, nearest_render_pkg)
    # Discard samples failing occlussion check
    valid = valid & (pts_in_nearest_cam[:, 2] - map_z <= opt.mv_occlusion_threshold)

    # Use the sampled depth values to reproject the points in the neighbor camera's view to reference camera's view
    re_projections = _reproject_points(nearest_cam, viewpoint_cam, pts_in_nearest_cam, map_z)
    # Calculates the Euclidean distance (pixel_noise) between the original pixel locations and the reprojected ones
    pixel_noise = torch.norm(re_projections - scene.pixels.reshape(*re_projections.shape), dim=-1)

    # Sample normals at the pixel locations in the reference camera's view
    normals = _sample_normal_map(scene.pixels, render_pkg['normal_map']) # (N, 3)
    # normals = normals @ viewpoint_cam.world_view_transform[:3, :3].T
    normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-8)
    # Compute cosine similarity between normals in ref view and sampled normals in neighbor view
    cos_sim = torch.sum(normals * map_n, dim=1) # (N,)
    angle_error_rad = torch.acos(cos_sim.clamp(-1 + 1e-6, 1 - 1e-6)) # [0, pi]
    angle_threshold = opt.mv_angle_threshold * torch.pi / 180.0 # to radians
    # Mask for variations within threshold
    angle_valid = valid & (angle_error_rad < angle_threshold)
    angle_noise = opt.mv_angle_factor * angle_error_rad

    pixel_valid = valid & (pixel_noise < 1.0)
    geo_weights = torch.exp(-pixel_noise * opt.mv_geo_weight_decay).detach()
    geo_weights[~pixel_valid] = 0

    pixel_loss = (geo_weights * pixel_noise)[pixel_valid.detach()].mean() if pixel_valid.sum() > 0 else 0.0
    angle_loss = (geo_weights * angle_noise)[angle_valid.detach()].mean() if angle_valid.sum() > 0 else 0.0
    geo_loss = pixel_loss + angle_loss

    if pipe.z_depth:
        return opt.multi_view_geo_weight * geo_loss

    ncc_scale = scene.ncc_scale
    with torch.no_grad():
        mask = pixel_valid.reshape(-1)
        valid_indices = torch.arange(mask.shape[0], device=mask.device)[mask]
        if mask.sum() > opt.multi_view_sample_num:
            index = np.random.choice(mask.sum().cpu().numpy(), opt.multi_view_sample_num, replace=False)
            valid_indices = valid_indices[index].detach()
        
        ncc_weights = torch.exp(-pixel_noise).detach()
        ncc_weights[~pixel_valid] = 0
        ncc_weights = ncc_weights.reshape(-1)[valid_indices]
        if material_stage:
            rough_weights = render_pkg["roughness_map"].squeeze().clamp(0, 1).detach() ** 2.0
            ncc_weights *= rough_weights.reshape(-1)[valid_indices]

        pixels = scene.pixels.reshape(-1, 2)[valid_indices]
        offsets = _patch_offsets(opt.multi_view_patch_size, pixels.device)
        ori_pixels_patch = pixels.reshape(-1, 1, 2) / ncc_scale + offsets.float()

        gt_image_gray = viewpoint_cam.gray_image.cuda() # (1, H, W)
        h, w = gt_image_gray.squeeze().shape
        pixels_patch = ori_pixels_patch.clone()
        pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (w - 1) - 1.0
        pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (h - 1) - 1.0
        ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
        total_patch_size = (opt.multi_view_patch_size * 2 + 1) ** 2
        ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

        rn_R = nearest_cam.world_view_transform[:3, :3].transpose(-1, -2) @ viewpoint_cam.world_view_transform[:3, :3]
        rn_t = -rn_R @ viewpoint_cam.world_view_transform[3, :3] + nearest_cam.world_view_transform[3, :3]

    ref_local_n = render_pkg["local_normal_map"].permute(1, 2, 0) # (H, W, 3)
    ref_local_n = ref_local_n.reshape(-1, 3)[valid_indices] # (N, 3)
    ref_local_d = render_pkg["distance_map"].squeeze() # (H, W)
    ref_local_d = ref_local_d.reshape(-1)[valid_indices] # (N,)

    H_rn = rn_R[None] - torch.matmul(
        rn_t[None, :, None].expand(ref_local_d.shape[0], 3, 1),
        ref_local_n[:, :, None].expand(ref_local_d.shape[0], 3, 1).permute(0, 2, 1)) / ref_local_d[..., None, None]
    H_rn = torch.matmul(nearest_cam.get_K(ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3), H_rn)
    H_rn = H_rn @ viewpoint_cam.get_inv_K(ncc_scale)

    grid = _patch_warp(H_rn.reshape(-1, 3, 3), ori_pixels_patch)
    grid[:, :, 0] = 2 * grid[:, :, 0] / (w - 1) - 1.0
    grid[:, :, 1] = 2 * grid[:, :, 1] / (h - 1) - 1.0
    nearest_image_gray = nearest_cam.gray_image.cuda()
    sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
    sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)

    ncc, ncc_mask = _loss_ncc(ref_gray_val, sampled_gray_val)
    ncc_mask = ncc_mask.reshape(-1)
    Lp = ncc.reshape(-1) * ncc_weights
    Lp = Lp[ncc_mask].squeeze()
    ncc_loss = Lp.mean() if ncc_mask.sum() > 0 else 0.0

    w_geo = opt.multi_view_geo_weight
    w_ncc = opt.multi_view_ncc_weight
    return w_geo * geo_loss + w_ncc * ncc_loss

def _get_points_from_depth(camera, depth_map, scale=1):
    # depth_map: (1, H, W)
    # Downsample the depth map if scale is not 1.0
    st = int(max(int(scale / 2) - 1, 0))
    depth_view = depth_map.squeeze()[st::scale, st::scale]
    # Scale ray points by depth and transform them to world coordinates
    rays_d = camera.get_rays(scale=scale)
    depth_view = depth_view[:rays_d.shape[0], :rays_d.shape[1]]
    pts = (rays_d * depth_view[..., None]).reshape(-1, 3)
    R = torch.tensor(camera.R).float().cuda()
    T = torch.tensor(camera.T).float().cuda()
    pts = (pts - T) @ R.transpose(-1, -2)
    return pts

def _sample_depth_normal(cam_points, camera, render_pkg, scale=1):
    depth_map = render_pkg['depth_map'] # (1, H, W)
    normal_map = render_pkg['normal_map'] # (3, H, W)

    st = max(int(scale / 2) - 1, 0)
    # Prepare depth and normal maps for grid_sample
    depth_view = depth_map[None, :, st::scale, st::scale] # (1, 1, H', W')
    normal_view = normal_map[None, :, st::scale, st::scale] # (1, 3, H', W')

    W, H = int(camera.image_width / scale), int(camera.image_height / scale)
    depth_view = depth_view[:, :, :H, :W]
    normal_view = normal_view[:, :, :H, :W]

    # Find pixel coordinates of cam_points (N, 2)
    pts_projections = torch.stack([
        cam_points[:, 0] * camera.Fx / cam_points[:, 2] + camera.Cx,
        cam_points[:, 1] * camera.Fy / cam_points[:, 2] + camera.Cy], dim=-1).float() / scale

    valid = (pts_projections[:, 0] > 0) & (pts_projections[:, 0] < W) & \
            (pts_projections[:, 1] > 0) & (pts_projections[:, 1] < H) & \
            (cam_points[:, 2] > 0.1) # (N,)

    # Normalize and prepare points for grid_sample
    pts_projections[..., 0] /= ((W - 1) / 2)
    pts_projections[..., 1] /= ((H - 1) / 2)
    pts_projections -= 1
    pts_projections = pts_projections.view(1, -1, 1, 2)

    # Sample depth (N,)
    map_z = F.grid_sample(
        input=depth_view,
        grid=pts_projections,
        mode='bilinear',
        padding_mode='border',
        align_corners=True)[0, 0, :, 0]

    # Sample normals: (N, 3)
    map_n = F.grid_sample(
        input=normal_view,
        grid=pts_projections,
        mode='bilinear',
        padding_mode='border',
        align_corners=True)[0, :, :, 0].permute(1, 0)
    
    # Rotate normals to world coordinates
    # map_n = map_n @ camera.world_view_transform[:3, :3].T
    map_n = map_n / (map_n.norm(dim=1, keepdim=True) + 1e-8)

    return map_z, map_n, valid

def _reproject_points(from_camera, to_camera, points, sampled_depth):
    # cam_points: (N, 3)
    # sampled_depth: (N,)

    pts = points / (points[:, 2:3])
    pts = pts * sampled_depth[..., None]
    R = torch.tensor(from_camera.R).float().cuda()
    T = torch.tensor(from_camera.T).float().cuda()
    pts = (pts - T) @ R.transpose(-1, -2)
    pts = pts @ to_camera.world_view_transform[:3, :3] + to_camera.world_view_transform[3, :3]

    pts_projections = torch.stack([
        pts[:, 0] * to_camera.Fx / pts[:, 2] + to_camera.Cx,
        pts[:, 1] * to_camera.Fy / pts[:, 2] + to_camera.Cy], dim=-1).float()
    return pts_projections

def _sample_normal_map(pixels, normal_map):
    # pixels: (H, W, 2)
    # normal_map: (3, H, W)
    H, W = pixels.shape[:2]
    pixels_flat = pixels.view(-1, 2) # (N, 2)

    # Normalize for grid_sample
    norm_pixels = pixels_flat.clone()
    norm_pixels[:, 0] = norm_pixels[:, 0] / ((W - 1) / 2) - 1
    norm_pixels[:, 1] = norm_pixels[:, 1] / ((H - 1) / 2) - 1
    norm_pixels = norm_pixels.view(1, -1, 1, 2)

    # Prepare normal map for grid_sample: (1, 3, H, W)
    ref_normal_map = normal_map.unsqueeze(0)
    normals = F.grid_sample(
        input=ref_normal_map,
        grid=norm_pixels,
        mode='bilinear',
        padding_mode='border',
        align_corners=True)[0, :, :, 0].permute(1, 0) # (N, 3)

    return normals

def _patch_offsets(h_patch_size, device):
    offsets = torch.arange(-h_patch_size, h_patch_size + 1, device=device)
    return torch.stack(torch.meshgrid(offsets, offsets, indexing='xy')[::-1], dim=-1).view(1, -1, 2)

def _patch_warp(H, uv):
    B, P = uv.shape[:2]
    H = H.view(B, 3, 3)
    ones = torch.ones((B, P, 1), device=uv.device)
    homo_uv = torch.cat((uv, ones), dim=-1)

    grid_tmp = torch.einsum("bik,bpk->bpi", H, homo_uv)
    grid_tmp = grid_tmp.reshape(B, P, 3)
    grid = grid_tmp[..., :2] / (grid_tmp[..., 2:] + 1e-10)
    return grid

def _loss_ncc(ref, nea, std_mask=False):
    # ref_gray: [batch_size, total_patch_size]
    # nea_grays: [batch_size, total_patch_size]
    bs, tps = nea.shape
    patch_size = int(np.sqrt(tps))

    ref_nea = ref * nea
    ref_nea = ref_nea.view(bs, 1, patch_size, patch_size)
    ref = ref.view(bs, 1, patch_size, patch_size)
    nea = nea.view(bs, 1, patch_size, patch_size)
    ref2 = ref.pow(2)
    nea2 = nea.pow(2)

    # sum over kernel
    filters = torch.ones(1, 1, patch_size, patch_size, device=ref.device)
    padding = patch_size // 2
    ref_sum = F.conv2d(ref, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea_sum = F.conv2d(nea, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref2_sum = F.conv2d(ref2, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea2_sum = F.conv2d(nea2, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref_nea_sum = F.conv2d(ref_nea, filters, stride=1, padding=padding)[:, :, padding, padding]

    # average over kernel
    ref_avg = ref_sum / tps
    nea_avg = nea_sum / tps

    cross = ref_nea_sum - nea_avg * ref_sum
    ref_var = ref2_sum - ref_avg * ref_sum
    nea_var = nea2_sum - nea_avg * nea_sum

    cc = cross * cross / (ref_var * nea_var + 1e-8)
    ncc = 1 - cc
    ncc = torch.clamp(ncc, 0.0, 2.0)
    ncc = torch.mean(ncc, dim=1, keepdim=True)
    mask = (ncc < 0.9)

    if not std_mask:
        return ncc, mask
    else:
        return ncc, torch.sqrt(ref_var) < 0.01

def tv_loss2(gt_image: torch.Tensor, prediction: torch.Tensor, pad=1, step=1):
    # gt_image: (3, H, W), prediction: (C, H, W)
    if pad > 1:
        gt_image = F.avg_pool2d(gt_image, pad, pad)
        prediction = F.avg_pool2d(prediction, pad, pad)
    rgb_grad_h = torch.exp(-(gt_image[:, 1:, :] - gt_image[:, :-1, :]).abs().mean(dim=0, keepdim=True)) # (1, H-1, W)
    rgb_grad_w = torch.exp(-(gt_image[:, :, 1:] - gt_image[:, :, :-1]).abs().mean(dim=0, keepdim=True)) # (1, H-1, W)
    # tv_h = torch.pow(prediction[:, 1:, :] - prediction[:, :-1, :], 2) # (C, H-1, W)
    # tv_w = torch.pow(prediction[:, :, 1:] - prediction[:, :, :-1], 2) # (C, H, W-1)
    tv_h = (prediction[:, 1:, :] - prediction[:, :-1, :]).abs() # (C, H-1, W)
    tv_w = (prediction[:, :, 1:] - prediction[:, :, :-1]).abs() # (C, H, W-1)
    tv_loss = (tv_h * rgb_grad_h).mean() + (tv_w * rgb_grad_w).mean()

    if step > 1:
        for s in range(2, step + 1):
            rgb_grad_h = torch.exp(-(gt_image[:, s:, :] - gt_image[:, :-s, :]).abs().mean(dim=0, keepdim=True)) # (1, H-1, W)
            rgb_grad_w = torch.exp(-(gt_image[:, :, s:] - gt_image[:, :, :-s]).abs().mean(dim=0, keepdim=True)) # (1, H-1, W)
            # tv_h = torch.pow(prediction[:, s:, :] - prediction[:, :-s, :], 2) # (C, H-1, W)
            # tv_w = torch.pow(prediction[:, :, s:] - prediction[:, :, :-s], 2) # (C, H, W-1)
            tv_h = (prediction[:, s:, :] - prediction[:, :-s, :]).abs() # (C, H-1, W)
            tv_w = (prediction[:, :, s:] - prediction[:, :, :-s]).abs() # (C, H, W-1)
            tv_loss += (tv_h * rgb_grad_h).mean() + (tv_w * rgb_grad_w).mean()

    return tv_loss

def tv_loss(gt_image: torch.Tensor, pred: torch.Tensor, norm1=True, weight_map: torch.Tensor=None) -> torch.Tensor:
    rgb_grad_h = torch.exp(-(gt_image[:, 1:, :] - gt_image[:, :-1, :]).abs().mean(dim=0, keepdim=True)) # (1, H-1, W)
    rgb_grad_w = torch.exp(-(gt_image[:, :, 1:] - gt_image[:, :, :-1]).abs().mean(dim=0, keepdim=True)) # (1, H, W-1)
    if norm1:
        tv_h = (pred[:, 1:, :] - pred[:, :-1, :]).abs() # (C, H-1, W)
        tv_w = (pred[:, :, 1:] - pred[:, :, :-1]).abs() # (C, H, W-1)
    else:
        tv_h = torch.pow(pred[:, 1:, :] - pred[:, :-1, :], 2) # (C, H-1, W)
        tv_w = torch.pow(pred[:, :, 1:] - pred[:, :, :-1], 2) # (C, H, W-1)

    loss_h = tv_h * rgb_grad_h
    loss_w = tv_w * rgb_grad_w

    # if erosion:
    #     kernel = mask.new_ones([7, 7])
    #     mask = kornia.morphology.erosion(mask[None, ...], kernel)[0]
    if weight_map is not None:
        loss_h *= ((weight_map[:, 1:, :] + weight_map[:, :-1, :]) / 2.0) # (1, H-1, W)
        loss_w *= ((weight_map[:, :, 1:] + weight_map[:, :, :-1]) / 2.0) # (1, H, W-1)

    tv_loss = loss_h.mean() + loss_w.mean()
    return tv_loss

def weighted_tv_loss2(gt_image: torch.Tensor, pred: torch.Tensor, weight_map: torch.Tensor=None, pad=1, step=1):
    # gt_image: (3, H, W), pred: (C, H, W), weight_map: (1, H, W) with values in [0, 1]
    if pad > 1:
        gt_image = F.avg_pool2d(gt_image, pad, pad)
        pred = F.avg_pool2d(pred, pad, pad)
        if weight_map is not None:
            weight_map = F.avg_pool2d(weight_map, pad, pad)

    rgb_grad_h = 1.0 # torch.exp(-(gt_image[:, 1:, :] - gt_image[:, :-1, :]).abs().mean(dim=0, keepdim=True)) # (1, H-1, W)
    rgb_grad_w = 1.0 # torch.exp(-(gt_image[:, :, 1:] - gt_image[:, :, :-1]).abs().mean(dim=0, keepdim=True)) # (1, H, W-1)
    tv_h = torch.pow(pred[:, 1:, :] - pred[:, :-1, :], 2) # (C, H-1, W)
    tv_w = torch.pow(pred[:, :, 1:] - pred[:, :, :-1], 2) # (C, H, W-1)
    # tv_h = (pred[:, 1:, :] - pred[:, :-1, :]).abs() # (C, H-1, W)
    # tv_w = (pred[:, :, 1:] - pred[:, :, :-1]).abs() # (C, H, W-1)

    if weight_map is not None:
        # Average weights for adjacent pixels along height and width gradients
        weight_h = (weight_map[:, 1:, :] + weight_map[:, :-1, :]) / 2 # (1, H-1, W)
        weight_w = (weight_map[:, :, 1:] + weight_map[:, :, :-1]) / 2 # (1, H, W-1)
        tv_loss = (tv_h * rgb_grad_h * weight_h).mean() + (tv_w * rgb_grad_w * weight_w).mean()
    else:
        tv_loss = (tv_h * rgb_grad_h).mean() + (tv_w * rgb_grad_w).mean()

    if step > 1:
        for s in range(2, step + 1):
            rgb_grad_h = torch.exp(-(gt_image[:, s:, :] - gt_image[:, :-s, :]).abs().mean(dim=0, keepdim=True)) # (1, H-s, W)
            rgb_grad_w = torch.exp(-(gt_image[:, :, s:] - gt_image[:, :, :-s]).abs().mean(dim=0, keepdim=True)) # (1, H, W-s)

            # tv_h = torch.pow(pred[:, s:, :] - pred[:, :-s, :], 2) # (C, H-s, W)
            # tv_w = torch.pow(pred[:, :, s:] - pred[:, :, :-s], 2) # (C, H, W-s)
            tv_h = (pred[:, s:, :] - pred[:, :-s, :]).abs() # (C, H-s, W)
            tv_w = (pred[:, :, s:] - pred[:, :, :-s]).abs() # (C, H, W-s)

            if weight_map is not None:
                weight_h = (weight_map[:, s:, :] + weight_map[:, :-s, :]) / 2
                weight_w = (weight_map[:, :, s:] + weight_map[:, :, :-s]) / 2
                tv_loss += (tv_h * rgb_grad_h * weight_h).mean() + (tv_w * rgb_grad_w * weight_w).mean()
            else:
                tv_loss += (tv_h * rgb_grad_h).mean() + (tv_w * rgb_grad_w).mean()

    return tv_loss

def laplacian_loss(pred: torch.Tensor, gt_image: torch.Tensor, weight_map: torch.Tensor=None) -> torch.Tensor:
    """
    Edge-aware Laplacian-based smoothness loss.
    
    Args:
        pred: Input tensor of shape (C, H, W) for smoothness.
        gt_image: Ground truth image of shape (C, H, W) for edge-aware weighting.
        weight_map: Optional mask tensor of shape (1, H, W).
        
    Returns:
        Computed edge-aware Laplacian loss as a scalar tensor.
    """

    lap = -4 * pred + torch.roll(pred, 1, 1) + torch.roll(pred, -1, 1) + torch.roll(pred, 1, 2) + torch.roll(pred, -1, 2)

    weights = (1.0 - _get_img_grad_weight(gt_image)).clamp(0, 1).detach() ** 2
    if weight_map is not None:
        weights *= weight_map.squeeze()

    loss = (lap.abs() * weights).mean()
    return loss

