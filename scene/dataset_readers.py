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

import numpy as np
import json
import os

from pathlib import Path
from plyfile import PlyData, PlyElement
from PIL import Image
from typing import NamedTuple

from scene.gaussian_model import BasicPointCloud
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text

from utils.graphics_utils import getWorld2View2, fov2focal
from utils.image_utils import convert_background_color
from utils.sh_utils import SH2RGB

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    Fx: np.array
    Fy: np.array
    image_name: str
    image_path: str
    image: Image.Image # image in original size, preprocessed if --white_background is set
    mask: Image.Image  # foreground mask, None if not provided (via --masks)
    depth: Image.Image # depth map, None if not provided (via --depths)
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, image_dir, mask_dir, depth_dir, white_bg):
    cam_infos = []
    for key in cam_extrinsics:
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec)) # store transposed due to 'glm' in CUDA code
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = focal_length_x
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
        else:
            assert False, "Unsupported COLMAP camera model!"

        image_path = os.path.join(image_dir, extr.name)
        image_name = extr.name
        image_stem = Path(image_path).stem

        image = Image.open(image_path)
        if white_bg:
            bg_color = np.array([1, 1, 1])
            image = convert_background_color(image, bg_color)

        mask_path  = os.path.join(mask_dir,  f"{image_stem}.png") if mask_dir  != "" else ""
        depth_path = os.path.join(depth_dir, f"{image_stem}.png") if depth_dir != "" else ""

        mask = Image.open(mask_path) if mask_path != "" else None
        depth = Image.open(depth_path) if depth_path != "" else None

        cam_info = CameraInfo(
            uid=uid, R=R, T=T, Fx=focal_length_x, Fy=focal_length_y, image_name=image_name, image_path=image_path,
            image=image, mask=mask, depth=depth, width=width, height=height)
        cam_infos.append(cam_info)
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        print("[>] Load Ply colors and normals failed, random init...")
        colors = np.random.rand(*positions.shape) / 255.0
        normals = np.random.rand(*positions.shape)
        normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, masks, depths, eval, white_background, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    image_dir = os.path.join(path, images)
    depth_dir = os.path.join(path, depths) if depths != "" else ""
    mask_dir = os.path.join(path, masks) if masks != "" else ""
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
        image_dir=image_dir, mask_dir=mask_dir, depth_dir=depth_dir, white_bg=white_background)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("[>] Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, depth_dir, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3]) # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_stem = Path(cam_name).stem
            image_name = Path(cam_name).name

            pil_image = Image.open(image_path)
            bg_color = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            image = convert_background_color(pil_image, bg_color)
            image.filename = pil_image.filename
            focal = fov2focal(fovx, image.size[0])

            depth_path = os.path.join(depth_dir, f"{image_stem}.png") if depth_dir != "" else ""
            depth = Image.open(depth_path) if depth_path != "" else None

            cam_info = CameraInfo(
                uid=idx, R=R, T=T, Fx=focal, Fy=focal, image=image, image_name=image_name, image_path=image_path,
                mask=None, depth=depth, width=image.size[0], height=image.size[1])
            cam_infos.append(cam_info)

    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):
    depth_dir = os.path.join(path, depths) if depths != "" else ""
    print("[>] Reading training transforms...")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depth_dir, white_background, extension)
    print("[>] Reading test transforms...")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depth_dir, white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"[>] Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
}