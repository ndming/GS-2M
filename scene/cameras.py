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
import numpy as np

from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal
from utils.image_utils import process_input_image

class Camera(torch.nn.Module):
    def __init__(
            self, uid, R, T, K, FoVx, FoVy, image_name, resolution,
            image, mask, use_mask, depth,
            trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda"):
        super(Camera, self).__init__()

        self.uid = uid
        self.R = R # transposed rotation in w2c (basically c2w rotation)
        self.T = T # translation in w2c
        self.K = K # corrected intrinsic matrix scaled to resolution
        self.nearest_indices = [] # nearest neighbors indices for this camera, populated by Scene
        self.image_name = image_name # name with suffix
        self.image_width  = resolution[0]
        self.image_height = resolution[1]
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.Fx = fov2focal(FoVx, self.image_width)
        self.Fy = fov2focal(FoVy, self.image_height)
        self.Cx = 0.5 * self.image_width
        self.Cy = 0.5 * self.image_height
        self.gray_images = {} # cached grayscale GT at each scale 

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # Get the resized GT and mask images. If use_mask is True, GT is masked and resized. The provided mask
        # can be None, in which case the alpha channel of the image is used if available.
        rgb, alpha = process_input_image(image, resolution, use_mask, mask) # resize and transpose to (C, H, W)
        self.gt_image = rgb[:3, ...].to(self.data_device)
        self.alpha_mask = alpha if alpha is not None else torch.ones_like(rgb[0:1, ...])
        self.alpha_mask = self.alpha_mask.to(self.data_device)

        self.pil_image = image
        self.use_mask = use_mask
        self.pil_mask = mask

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def get_rays(self, scale=1.0):
        h, w = int(self.image_height / scale), int(self.image_width / scale)
        u, v = torch.meshgrid(
            torch.arange(w, device="cuda", dtype=torch.float32),
            torch.arange(h, device="cuda", dtype=torch.float32),
            indexing='xy')
        rx = (scale * u - self.Cx / scale) / self.Fx
        ry = (scale * v - self.Cy / scale) / self.Fy
        rays = torch.stack((rx, ry, torch.ones_like(rx)), dim=-1) # (H, W, 3)
        return rays

    def get_calib_matrix_nerf(self, scale=1.0):
        intrinsic_matrix = torch.tensor([[self.Fx/scale, 0, self.Cx/scale], [0, self.Fy/scale, self.Cy/scale], [0, 0, 1]]).float()
        extrinsic_matrix = self.world_view_transform.transpose(0, 1).contiguous() # w2c
        return intrinsic_matrix, extrinsic_matrix
    
    def get_gray_image(self, scale=1.0):
        if scale in self.gray_images:
            return self.gray_images[scale].cuda()
        
        rgb = self.gt_image
        if scale != 1.0:
            res = int(self.image_width / scale), int(self.image_height / scale)
            rgb, _ = process_input_image(self.pil_image, res, self.use_mask, self.pil_mask) # (C, H, W)
            rgb = rgb[:3, ...].to(self.data_device)

        gray = rgb[0:1, ...] * 0.299 + rgb[1:2, ...] * 0.587 + rgb[2:3, ...] * 0.114 # (1, H, W)
        self.gray_images[scale] = gray
        return gray.cuda()

    def get_K(self, scale=1.0):
        K = torch.tensor([
            [self.Fx / scale, 0.0, self.Cx / scale],
            [0.0, self.Fy / scale, self.Cy / scale],
            [0.0, 0.0, 1.0]]).cuda()
        return K
    
    def get_inv_K(self, scale=1.0):
        K_T = torch.tensor([
            [scale / self.Fx, 0.0, -self.Cx / self.Fx],
            [0.0, scale / self.Fy, -self.Cy / self.Fy],
            [0.0, 0.0, 1.0]]).cuda()
        return K_T

        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
