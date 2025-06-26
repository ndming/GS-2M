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
import random
import json
import torch

import numpy as np
import torch.nn.functional as F

from PIL import Image
from pathlib import Path
from arguments import ModelParams
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from pbr import CubemapLight, get_brdf_lut

from utils.graphics_utils import get_envmap_dirs
from utils.system_utils import searchForMaxIteration
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.image_utils import process_input_image, convert_background_color

class Scene:
    gaussians : GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        self.brdf_lut = get_brdf_lut().cuda()
        self.envmap_dirs = get_envmap_dirs()
        self.cubemap = CubemapLight(base_res=256).cuda()

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print(f"[>] Using trained model at iteration {self.loaded_iter}")

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            print("[>] Found sparse directory, assuming COLMAP dataset!")
            params = (args.source_path, args.images, args.masks, args.depths, args.eval, args.white_background)
            scene_info = sceneLoadTypeCallbacks["Colmap"](*params)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("[>] Found transforms_train.json file, assuming Blender dataset!")
            params = (args.source_path, args.white_background, args.depths, args.eval)
            scene_info = sceneLoadTypeCallbacks["Blender"](*params)
        else:
            assert False, "Could not recognize scene type!"

        # Write initial point cloud (sparse SfM or random) to input.ply and train/test views to cameras.json at model path
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras) # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print(f"[>] Scene half extent: {self.cameras_extent:.4f}")

        n_train_cameras = len(scene_info.train_cameras)
        n_test_cameras  = len(scene_info.test_cameras)
        print(f"[>] Found {n_train_cameras + n_test_cameras} cameras: {n_train_cameras} train, {n_test_cameras} test")

        for scale in resolution_scales:
            self.train_cameras[scale] = cameraList_from_camInfos(scene_info.train_cameras, scale, args, False)
            self.test_cameras[scale] = cameraList_from_camInfos(scene_info.test_cameras, scale, args, True)

        if self.loaded_iter:
            load_dir = Path(self.model_path) / f"point_cloud/iteration_{self.loaded_iter}"
            lighting = torch.load(str(load_dir / "lighting.pth"))

            self.gaussians.load_ply(str(load_dir / "point_cloud.ply"))
            self.cubemap.load_state_dict(lighting['cubemap'])
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        # point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        save_dir = Path(self.model_path) / f"point_cloud/iteration_{iteration}"
        self.gaussians.save_ply(str(save_dir / "point_cloud.ply"))

        light_dict = {
            'cubemap': self.cubemap.state_dict(),
        }
        torch.save(light_dict, str(save_dir / "lighting.pth"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def training_setup(self, opt, model, resolution_scale=1.0):
        print("[>] Populating camera neighbors")
        self._populate_neareast_cameras(opt, resolution_scale)

        if opt.multi_view_ncc_scale > 0:
            self.ncc_scale = opt.multi_view_ncc_scale
        elif model.resolution in [1, 2, 4, 8]:
            self.ncc_scale = 1.0 / model.resolution
        else:
            self.ncc_scale = 1.0
        print(f"[>] Using NCC scale: {self.ncc_scale:.2f}")

        print("[>] Populating gray images")
        self._populate_gray_images(model.mask_gt, model.white_background, resolution_scale)

        self.cubemap.train()
        param_groups = [
            { "name": "cubemap", "params": self.cubemap.parameters(), "lr": opt.opacity_lr },
        ]
        self.light_optimizer = torch.optim.Adam(param_groups, lr=opt.opacity_lr)

    def _populate_neareast_cameras(self, opt, resolution_scale):
        world_view_transforms = []
        camera_centers = []
        center_rays = []
        for cam in self.train_cameras[resolution_scale]:
            world_view_transforms.append(cam.world_view_transform)
            camera_centers.append(cam.camera_center)
            R = torch.tensor(cam.R).float().cuda()
            center_rays.append(R[:3, 2])

        world_view_transforms = torch.stack(world_view_transforms, dim=0)
        camera_centers = torch.stack(camera_centers, dim=0)
        center_rays = torch.stack(center_rays, dim=0)
        center_rays = torch.nn.functional.normalize(center_rays, dim=-1)
        distances = torch.norm(camera_centers[:, None] - camera_centers[None], dim=-1).detach().cpu().numpy()
        tmp = torch.sum(center_rays[:, None] * center_rays[None], dim=-1)
        angles = torch.arccos(tmp) * 180 / 3.14159
        angles = angles.detach().cpu().numpy()

        for id, cur_cam in enumerate(self.train_cameras[resolution_scale]):
            sorted_indices = np.lexsort((angles[id], distances[id]))
            mask = (angles[id][sorted_indices] <= opt.multi_view_max_angle) & \
                   (distances[id][sorted_indices] > opt.multi_view_min_dist) & \
                   (distances[id][sorted_indices] < opt.multi_view_max_dist)
            sorted_indices = sorted_indices[mask]
            multi_view_num = min(opt.multi_view_num, len(sorted_indices))
            for index in sorted_indices[:multi_view_num]:
                cur_cam.nearest_indices.append(index)

    def _populate_gray_images(self, mask_gt, white_bg, resolution_scale):
        for cam in self.train_cameras[resolution_scale]:
            rgb = cam.gt_image
            ncc_scale = self.ncc_scale
            if ncc_scale != 1.0:
                pil_image = Image.open(cam.image_path)
                bg_color = np.array([1, 1, 1]) if white_bg else np.array([0, 0, 0])
                image = convert_background_color(pil_image, bg_color)
                pil_mask = None if cam.mask_path is None else Image.open(cam.mask_path).convert("L")
                res = int(cam.image_width / ncc_scale), int(cam.image_height / ncc_scale)
                rgb, _ = process_input_image(image, res, mask_gt, pil_mask) # (C, H, W)
                rgb = rgb[:3, ...].to(cam.data_device)
            gray = rgb[0:1, ...] * 0.299 + rgb[1:2, ...] * 0.587 + rgb[2:3, ...] * 0.114 # (1, H, W)
            cam.gray_image = gray.to(cam.data_device)

    def get_canonical_rays(self, resolution_scale=1.0) -> torch.Tensor:
        ref_camera = self.train_cameras[resolution_scale][0]
        H, W = ref_camera.image_height, ref_camera.image_width
        x, y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
        x = x.flatten() # [H * W]
        y = y.flatten() # [H * W]
        camera_dirs = F.pad(
            torch.stack([(x - ref_camera.Cx + 0.5) / ref_camera.Fx, (y - ref_camera.Cy + 0.5) / ref_camera.Fy], dim=-1),
            (0, 1), value=1.0)
        return camera_dirs.cuda()