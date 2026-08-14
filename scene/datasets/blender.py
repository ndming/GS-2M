import json

import imageio.v2 as imageio
import numpy as np

from pathlib import Path
from typing import List, Optional

from .utils import process_input_images


def _fov_to_focal(fov: float, pixels: int) -> float:
    return pixels / (2.0 * np.tan(fov / 2.0))


def _read_frames(data_path: Path, json_file: Path, extension: str):
    """Return (fovx, frames) where each frame carries an OpenCV c2w and its source path."""
    with open(json_file) as f:
        contents = json.load(f)
    fovx = contents["camera_angle_x"]

    ext = extension
    if ext[0] != ".":
        ext = f".{extension}"

    frames = []
    for frame in contents["frames"]:
        p = Path(frame["file_path"])
        if p.suffix == "":
            p = p.with_suffix(ext)
        source_path = (data_path / p).resolve()

        # NeRF 'transform_matrix' is a camera-to-world in OpenGL/Blender axes (Y up, Z back).
        # Flip Y and Z to get the OpenCV convention (Y down, Z forward) we use everywhere.
        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        c2w[:3, 1:3] *= -1
        frames.append({"c2w": c2w, "source_path": source_path, "basename": source_path.name})

    return fovx, frames


class Parser:
    """NeRF-synthetic (Blender) parser, matching ColmapParser interface."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        load_exposure: bool = False,
        mask_gt_image: bool = False,
        **kwargs,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        self.load_exposure = load_exposure

        if normalize:
            print("[!] Blender parser: scene normalization is not supported for Blender scenes, ignoring")
        if kwargs.get("center_world_space", False):
            print("[!] Blender parser: scene centering is not supported for Blender scenes, ignoring")

        data_path = Path(data_dir)
        extension = kwargs.get("blender_file_extension", "png")
        reuse = kwargs.get("reuse_processed_images", False)

        # Parse the canonical train/test transforms
        fovx, train_frames = _read_frames(data_path, data_path / "transforms_train.json", extension)
        _, test_frames = _read_frames(data_path, data_path / "transforms_test.json", extension)
        frames_by_split = {"train": train_frames, "test": test_frames}

        # Single shared intrinsic: constant FoV and image size across the scene.
        # Read one image to recover the full-resolution size.
        height, width = imageio.imread(str(train_frames[0]["source_path"])).shape[:2]
        focal = _fov_to_focal(fovx, width)
        K = np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])
        K[:2, :] /= factor

        image_names: List[str] = []
        image_paths: List[str] = []
        camtoworlds: List[np.ndarray] = []
        split_indices = {}
        proc_workers = kwargs.get("image_process_workers", None)
        for split in ("train", "test"):
            frames = frames_by_split[split]

            # All frames of a split are expected to live under a single directory
            source_dirs = {f["source_path"].parent for f in frames}
            assert len(source_dirs) == 1, f"Blender {split} frames span multiple dirs: {source_dirs}"
            source_dir = source_dirs.pop()
            basenames = [f["basename"] for f in frames]

            out_dir = data_path / f"images_{factor}x" / split
            paths = process_input_images(
                source_dir, out_dir, basenames, factor, reuse=reuse,
                mask_image=mask_gt_image, num_workers=proc_workers,
            )
            assert len(paths) == len(frames), (
                f"Processed {len(paths)} images but {split} split has {len(frames)} frames"
            )

            split_start = len(image_paths)
            image_paths.extend(paths)
            image_names.extend(f"{split}/{Path(b).stem}" for b in basenames)
            camtoworlds.extend(f["c2w"] for f in frames)
            split_indices[split] = np.arange(split_start, split_start + len(frames))

        print(f"[>] Blender parser: {len(image_paths)} images "
              f"({len(train_frames)} train, {len(test_frames)} test)")

        camtoworlds = np.stack(camtoworlds, axis=0)
        num_images = len(image_paths)

        self.image_names = image_names      # List[str], (num_images,)
        self.image_paths = image_paths      # List[str], (num_images,)
        self.camtoworlds = camtoworlds      # np.ndarray, (num_images, 4, 4), OpenCV c2w
        self.camera_ids = [0] * num_images  # single shared camera
        self.Ks_dict = {0: K}
        self.params_dict = {0: np.empty(0, dtype=np.float32)}  # pinhole, no distortion
        self.imsize_dict = {0: (width // factor, height // factor)}
        self.mask_dict = {0: None}
        self.transform = np.eye(4)
        self.split_indices = split_indices  # honored by Dataset in place of test_every

        # Init points for Blender scenes will be initialized by runner
        self.points = np.empty((0, 3), dtype=np.float32)
        self.points_rgb = np.empty((0, 3), dtype=np.float32)
        self.points_err = np.empty((0, 3), dtype=np.float32)
        self.point_indices = {}  # no per-image visibility for sparse points

        # Contiguous camera indexing (single camera)
        self.camera_id_to_idx = {0: 0}
        self.camera_indices = [0] * num_images
        self.num_cameras = 1

        # PNG carries no EXIF; exposure is unavailable for synthetic scenes
        if load_exposure:
            print("[>] Blender parser: EXIF exposure unavailable for synthetic scenes")
        self.exposure_values: List[Optional[float]] = [None] * num_images

        # Size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)
