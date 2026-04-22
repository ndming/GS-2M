import json
import math

from pathlib import Path
from typing import List, Optional

import imageio.v2 as imageio
import open3d as o3d
import numpy as np

from pycolmap.rotation import Quaternion
from tqdm import tqdm
from .utils import process_input_images, process_input_depths


def _read_keyframe_poses(pose_file, stride=1, max=0, offset=0):
    poses = []
    times = []

    with open(pose_file, "r") as f:
        for id, line in enumerate(f):
            if id / stride < offset:
                continue
            if id % stride != 0:
                continue
            if max > 0 and id / stride > max:
                break 

            line = line.strip()

            # Skip empty lines or comments
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            assert len(parts) == 8, f"Invalid line: {line}"
            
            times.append(int(parts[0]))

            tx, ty, tz = map(float, parts[1:4])
            qw, qx, qy, qz = map(float, parts[4:8]) # check the actual file for scalar order

            t = np.array([tx, ty, tz])
            R = Quaternion(np.array([qw, qx, qy, qz])).ToR()

            T = np.eye(4)
            T[:3, :3] = R
            T[:3,  3] = t
            poses.append(T)

    return times, poses


def _read_image_names(rgb_dir, kf_stamps):
    names = []
    
    for stamp in kf_stamps:
        image_file = rgb_dir / f"{stamp}.png"
        if not image_file.exists():
            raise ValueError(f"Missing image name ({image_file.name})")
        names.append(image_file.name)

    assert len(names) == len(kf_stamps), f"Inconsistent images and KF poses"
    return names


def _read_intrinsics(intrinsics_file, kf_stamps, factor=1):
    Ks_dict = dict()
    imsize_dict = dict() # width, height
    params_dict = dict()
    mask_dict = dict()

    with open(intrinsics_file, "r") as f:
        for id, line in enumerate(f):
            line = line.strip()

            # Skip empty lines or comments
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            assert len(parts) == 7, f"Invalid line: {line}"

            stamp = int(parts[0])
            if not stamp in kf_stamps:
                continue

            fx, fy, cx, cy = map(float, parts[1:5])
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[id] = K

            width, height = map(int, parts[5:7])
            imsize_dict[id] = (width // factor, height // factor)

            params = np.empty(0, dtype=np.float32)
            params_dict[id] = params
            mask_dict[id] = None  # distorted images only
    
    assert len(Ks_dict) == len(kf_stamps)
    return Ks_dict, imsize_dict, params_dict, mask_dict


def _read_calibration(calib_file, factor):
    Ks_dict = dict()
    imsize_dict = dict() # width, height
    params_dict = dict()

    with open(calib_file, "r") as f:
       data = json.load(f)

    cam_data = data["value0"]["intrinsics"][0]
    assert cam_data["camera_type"] == "pinhole"

    intrinsics = cam_data["intrinsics"]
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    K[:2, :] /= factor
    Ks_dict[0] = K

    resolution = data["value0"]["resolution"][0]
    width, height = resolution
    imsize_dict[0] = (width // factor, height // factor)

    params = np.empty(0, dtype=np.float32)
    params_dict[0] = params

    return Ks_dict, imsize_dict, params_dict


def _read_points(pcd_dir, image_paths, kf_poses, camera_ids, Ks_dict):
    points = []
    points_rgb = []
    point_indices = {}

    global_offset = 0 # tracks index in concatenated array

    for id, (path, pose) in tqdm(enumerate(zip(image_paths, kf_poses)), desc="[>] Reading points", ncols=128):
        image_file = Path(path)
        name = image_file.name
        stem = image_file.stem
        pcd_file = pcd_dir / f"{stem}.pcd"

        if not pcd_file.exists():
            point_indices[name] = np.empty((0,), dtype=np.int32)
            continue

        # Load PCD
        pcd = o3d.io.read_point_cloud(str(pcd_file))
        pts = np.asarray(pcd.points) # [N, 3] 

        if pts.shape[0] == 0:
            point_indices[name] = np.empty((0,), dtype=np.int32)
            continue

        # Load and project points to images to sample colors
        K = Ks_dict[camera_ids[id]]                            # [3, 3]
        img = imageio.imread(image_file)[..., :3]  # [H, W, 3]
        H, W = img.shape[:2]

        uvw = (K @ pts.T)  # [3, N]
        z   = uvw[2]       # [N] depth in camera frame
        u = uvw[0] / np.clip(z, 1e-6, None)  # [N]
        v = uvw[1] / np.clip(z, 1e-6, None)  # [N]

        valid = (
            (z > 0) &
            (u >= 0) & (u < W) &
            (v >= 0) & (v < H)
        )

        # Sample colors for valid points (nearest-neighbour)
        colors = np.random.uniform(0.0, 0.5, size=(pts.shape[0], 3)).astype(np.float32)
        if valid.any():
            px = np.round(u[valid]).astype(np.int32).clip(0, W - 1)
            py = np.round(v[valid]).astype(np.int32).clip(0, H - 1)
            sampled = img[py, px]  # [M, 3]  uint8
            colors[valid] = sampled.astype(np.float32) / 255.0

        # Transform to world, pose is c2w [4, 4]
        R = pose[:3, :3] # [3, 3]
        t = pose[:3, 3]  # [3]

        pts_world = (R @ pts.T).T + t  # [N, 3]
        N = pts_world.shape[0]

        inds = np.arange(global_offset, global_offset + N, dtype=np.int32)
        point_indices[name] = inds

        points.append(pts_world)
        points_rgb.append(colors)

        global_offset += N

    if len(points) == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            point_indices,
        )
    
    points = np.concatenate(points, axis=0).astype(np.float32)
    points_rgb = np.concatenate(points_rgb, axis=0).astype(np.float32)

    return points, points_rgb, point_indices


def _read_shutter_meta(times_file, kf_stamps):
    # Read all shutter times first
    stamp_to_time = {}
    stamp_to_gain = {}
    with open(times_file, "r") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines or comments
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) != 4:
                raise ValueError(f"Invalid line: {line}")
            
            stamp_to_time[int(parts[0])] = float(parts[2])
            stamp_to_gain[int(parts[0])] = float(parts[3])

    shutter_times = []
    shutter_gains = []
    for stamp in kf_stamps:
        if not stamp in stamp_to_time:
            print(f"[!] Warning: missing shutter time for frame at timestamp {stamp}")
            shutter_times.append(None)
        else:
            # Convert milliseconds to seconds
            shutter_times.append(stamp_to_time[stamp] * 1e-3)

        if not stamp in stamp_to_gain:
            print(f"[!] Warning: missing shutter gain for frame at timestamp {stamp}")
            shutter_gains.append(None)
        else:
            # Conver mDB -> DB -> linear gain
            gain_db = stamp_to_gain[stamp] * 1e-3
            gain_linear = 10 ** (gain_db / 20.0)
            shutter_gains.append(gain_linear)
    
    assert len(shutter_times) == len(kf_stamps)
    assert len(shutter_gains) == len(kf_stamps)

    return shutter_times, shutter_gains


class Parser:
    """DSO-formatted parser"""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        load_exposure: bool = False,
        mask_gt_image: bool = False,
        reuse_processed_images: bool = False,
        **kwargs,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        self.load_exposure = load_exposure

        pose_file = Path(data_dir) / "output" / "poses.txt"
        assert pose_file.exists(), f"{pose_file} not found"

        stride = kwargs.get("num_stride_frames", 1)
        offset = kwargs.get("num_offset_frames", 0)
        print(f"[>] DSO parsers: load frames from offset {offset} with stride {stride}")
        kf_stamps, kf_poses = _read_keyframe_poses(pose_file, stride=stride, offset=offset)
        assert len(kf_stamps) == len(kf_poses)

        if len(kf_stamps) == 0:
            raise ValueError(f"No images found in {data_dir}")
        
        Ks_dict, imsize_dict, params_dict = _read_calibration(Path(data_dir) / "calibration.json", factor)
        mask_dict = { 0: None } # distorted images only
        # intrinsics_file = Path(data_dir) / "output" / "intrinsics.txt"
        # Ks_dict, imsize_dict, params_dict, mask_dict = _read_intrinsics(intrinsics_file, kf_stamps, factor)
        # camera_ids = list(Ks_dict.keys())
        print(f"[>] Parser: {len(kf_stamps)} images, taken by {len(Ks_dict)} camera")

        dso_image_dir = Path(data_dir) / "dso" / "rgb"
        dso_depth_dir = Path(data_dir) / "ffs"
        
        # These two arrays match 1-to-1 (image and pose)
        c2w_mats = np.stack(kf_poses, axis=0)
        image_names = _read_image_names(dso_image_dir, kf_stamps)
        camera_ids = [0] * len(image_names) # only use one camera at the moment

        # Preprocess input images and depths
        processed_image_dir = Path(data_dir) / f"processed_images_{factor}"
        processed_depth_dir = Path(data_dir) / f"processed_depths_{factor}"
        
        image_paths = process_input_images(
            str(dso_image_dir), str(processed_image_dir), image_names, factor, reuse=reuse_processed_images,
            mask_image=mask_gt_image
        )
        depth_paths = process_input_depths(
            str(dso_depth_dir), str(processed_depth_dir), image_names, factor, reuse=reuse_processed_images,
        )
        
        pcd_dir = Path(data_dir) / "output" / "clouds"
        points, points_rgb, point_indices = _read_points(pcd_dir, image_paths, kf_poses, camera_ids, Ks_dict)
        points_err = None

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = Path(data_dir) / "ext_metadata.json"
        if extconf_file.exists():
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])
        pose_file = Path(data_dir) / "poses_bounds.npy"
        if pose_file.exists():
            self.bounds = np.load(pose_file)[:, -2:]

        transform = np.eye(4)
        if normalize:
            # We're not supporting scene normalization for DSO dataset because of c2w convention
            raise ValueError(f"Scene normalization is currently not supported for DSO dataset")

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = c2w_mats     # np.ndarray, (num_images, 4, 4)
        self.camera_ids  = camera_ids   # List[int], (num_images,)
        self.Ks_dict     = Ks_dict      # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict   = mask_dict    # Dict of camera_id -> mask
        # Sparse SfM points
        self.points = points                # np.ndarray, (num_points, 3) in world space
        self.points_err = points_err        # np.ndarray, (num_points,)
        self.points_rgb = points_rgb        # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform          # np.ndarray, (4, 4)
        # GT depth maps
        self.depth_paths = depth_paths
        self.depth_scale = 1.0 / 1000.0

        # Create 0-based contiguous camera indices from camera_ids.
        # This is useful for camera-based embeddings/modules.
        unique_camera_ids = sorted(set(camera_ids))
        self.camera_id_to_idx = { cid: idx for idx, cid in enumerate(unique_camera_ids) }
        self.camera_indices = [self.camera_id_to_idx[cid] for cid in camera_ids]
        self.num_cameras = len(unique_camera_ids)

        # Load exposure data if requested, we read from the exported shutter times
        self.exposure_values = [None] * len(image_paths)
        if load_exposure:
            times_file = Path(data_dir) / "dso" / "times.txt"
            shutter_times, shutter_gains = _read_shutter_meta(times_file, kf_stamps)
            exposure_values = [
                math.log2((t / (2.2 ** 2)) * g * 100.0) # Using hardcoded aperture f/2.2
                if t is not None and t > 0.0 and g is not None else None 
                for t, g in zip(shutter_times, shutter_gains)
            ]

            valid_exposures = [e for e in exposure_values if e is not None]
            if valid_exposures:
                exposure_mean = sum(valid_exposures) / len(valid_exposures)
                self.exposure_values: List[Optional[float]] = [
                    (e - exposure_mean) if e is not None else None
                    for e in exposure_values
                ]
                print(
                    f"[>] Parser: loaded exposure for {len(valid_exposures)}/{len(exposure_values)} images "
                    f"(mean={exposure_mean:.3f} EV)"
                )

            else:
                self.exposure_values = [None] * len(exposure_values)
                print("[>] Parser: no valid EXIF exposure data found in any image")

        # Load one image to check the size. In the case of tanksandtemples dataset,
        # the intrinsics stored in COLMAP corresponds to 2x upsampled images.
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        calib_width, calib_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / calib_height, actual_width / calib_width
        if s_height != 1.0 or s_width != 1.0:
            print(f"[!] Warning: actual image size ({actual_width}x{actual_height}) does not match scaled intrinsics ({calib_width}x{calib_height})")
            print(f"[>] Rescaling intrinsics by ({s_width:.2f}x, {s_height:.2f}x)")
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # Undistortion, do nothing
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()

        # Size of the scene measured by cameras
        camera_locations = c2w_mats[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)
