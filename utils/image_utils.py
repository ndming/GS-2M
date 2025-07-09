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
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def dilate(bin_img, ksize=5):
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out

def erode(bin_img, ksize=5):
    out = 1 - dilate(1 - bin_img, ksize)
    return out

def convert_background_color(image: Image.Image, bg_color):
    im_data = np.array(image.convert("RGBA")) / 255.0
    rgb = im_data[:, :, :3]
    alpha = im_data[:, :, 3:4]

    blended_rgb = rgb * alpha + bg_color * (1 - alpha)
    rgba = np.concatenate((blended_rgb, alpha), axis=2)
    rgba_uint8 = (rgba * 255).astype(np.uint8)

    converted_image = Image.fromarray(rgba_uint8, "RGBA")
    return converted_image

def process_input_image(pil_image: Image.Image, resolution, mask_gt=False, pil_mask=None):
    image = pil_image
    alpha = pil_mask

    if pil_image.mode == 'RGBA':
        r, g, b, a = pil_image.split()
        image = Image.merge("RGB", (r, g, b))
        alpha = a if alpha is None else alpha # prioritize provided mask over alpha channel

    # Mask GT images before resizing
    if mask_gt and alpha is not None:
        image_np = np.array(image)[..., :3].astype(np.float32)
        alpha_np = np.expand_dims(np.array(alpha), axis=-1).astype(np.float32)
        rgb_masked = (image_np / 255.) * (alpha_np / np.max(alpha_np))
        rgb_masked = np.clip(rgb_masked, 0., 1.)
        image = Image.fromarray((rgb_masked * 255.).astype(np.uint8))

    image = image.resize(resolution)
    image = torch.from_numpy(np.array(image)) / 255.
    if len(image.shape) < 3:
        image = image.unsqueeze(dim=-1)
    image = image.permute(2, 0, 1)

    if alpha is not None:
        alpha = alpha.resize(resolution)
        alpha = torch.from_numpy(np.array(alpha)).float()
        alpha = alpha / torch.max(alpha)
        alpha = alpha.unsqueeze(dim=-1).permute(2, 0, 1)

    return image, alpha

def save_depth_map(depth, path):
    # depth is a numpy array of (H, W)
    lower_bound = np.percentile(depth, 1)
    upper_bound = np.percentile(depth, 99)
    depth_clipped = np.clip(depth, lower_bound, upper_bound)
    normed = (depth_clipped - lower_bound) / (upper_bound - lower_bound + 1e-8)
    colored = plt.cm.magma(normed) # returns RGBA array in [0, 1] range
    map = (colored[..., :3] * 255).astype(np.uint8)
    plt.imsave(str(path), map)

def convert_depth_for_save(depth_map, max_depth=None):
    # depth_map is a tensor of shape (1, H, W)
    # normalize depth to [0, 1] range
    lower_bound = torch.quantile(depth_map, 0.01)
    upper_bound = torch.quantile(depth_map, 0.99) if max_depth is None else min(torch.quantile(depth_map, 0.99), max_depth)
    depth_clipped = torch.clamp(depth_map, min=lower_bound, max=upper_bound)
    min_val = depth_clipped.amin(dim=(1, 2), keepdim=True)
    max_val = depth_clipped.amax(dim=(1, 2), keepdim=True)
    depth_normalized = (depth_clipped - min_val) / (max_val - min_val + 1e-8)
    return depth_normalized

def convert_normal_for_save(normal_map, viewpoint, world_space=False):
    # normal_map is a tensor of shape (3, H, W)

    # Flatten and normalize normals
    normals = normal_map.permute(1, 2, 0).view(-1, 3).clone() # (H * W, 3)
    # valid_mask = ~(normals == 0).all(dim=1)
    normals = torch.nn.functional.normalize(normals, dim=1, p=2)

    # Apply Y-up and Z-back coordinate fix
    # T = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=normals.dtype, device=normals.device)
    # normals[~valid_mask] = normals[~valid_mask] @ T.T

    # if world_space:
    #     normals[valid_mask] = normals[valid_mask] @ viewpoint.world_view_transform[:3, :3]
    # else:
    #     normals[valid_mask] = normals[valid_mask] @ T.T

    # Adjust range
    normals = normals * 0.5 + 0.5 # [-1, 1] -> [0, 1]
    H, W = viewpoint.image_height, viewpoint.image_width
    return normals.reshape(H, W, 3).permute(2, 0, 1)


def map_to_rgba(map, alpha):
    tensor_map = (map * 255).byte()
    alpha_map = (alpha * 255).byte()

    tensor_np = tensor_map.cpu().numpy()
    alpha_np = alpha_map.cpu().numpy()

    if tensor_np.shape[0] == 3:
        rgba_np = np.concatenate((tensor_np, alpha_np), axis=0)
    else:
        rgba_np = np.concatenate((tensor_np, tensor_np, tensor_np, alpha_np), axis=0)

    rgba_np = np.transpose(rgba_np, (1, 2, 0))
    image = Image.fromarray(rgba_np, 'RGBA')
    return image