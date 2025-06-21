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

from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import focal2fov
from tqdm import tqdm

WARNED = False

def loadCam(args, cam_info, resolution_scale):
    FovY = focal2fov(cam_info.Fy, cam_info.height)
    FovX = focal2fov(cam_info.Fx, cam_info.width)

    orig_w, orig_h = cam_info.image.size
    if args.resolution in [1, 2, 4, 8]:
        scale = resolution_scale * args.resolution
        resolution = round(orig_w / scale), round(orig_h/ scale)
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    return Camera(
        uid=cam_info.uid, R=cam_info.R, T=cam_info.T, FoVx=FovX, FoVy=FovY,
        image_name=cam_info.image_name, resolution=resolution,
        image=cam_info.image, mask=cam_info.mask, mask_gt=args.mask_gt,
        depth=cam_info.depth, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_test_cams):
    camera_list = []
    total_cameras = len(cam_infos)

    split = "test" if is_test_cams else "train"
    description = f"[>] Loading {split} cameras"

    with tqdm(desc=description, total=total_cameras, bar_format='{l_bar}{r_bar}', leave=False) as pbar:
        for cam_info in cam_infos:
            camera_list.append(loadCam(args, cam_info, resolution_scale))
            pbar.update(1)

    return camera_list

def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose() # rotation was stored transposed, revert it
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    c2w = np.linalg.inv(Rt)
    pos = c2w[:3, 3]
    rot = c2w[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : camera.Fy,
        'fx' : camera.Fx,
    }
    return camera_entry