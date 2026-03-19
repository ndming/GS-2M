from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import torch

import imageio.v2 as imageio
import numpy as np

from .colmap import Parser as ColmapParser
from .dso import Parser as DsoParser


def get_parser(data_dir):
    if (Path(data_dir) / "sparse").exists():
        print("[>] Detected sparse dir, assuming COLMAP scene")
        return ColmapParser
    if (Path(data_dir) / "dso").exists():
        print("[>] Detected dso dir, assuming DSO scene")
        return DsoParser


class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: ColmapParser | DsoParser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_point_depth: bool = False,
        load_image_depth: bool = False,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_point_depth = load_point_depth
        self.load_image_depth = load_image_depth
        
        indices = np.arange(len(self.parser.image_names))
        if self.parser.test_every <= 0 and split == "train":
            self.indices = indices # get all training images if testing is disabled
        elif self.parser.test_every <= 0 and split == "val":
            self.indices = indices[indices % 5 == 0] # sample images like the original 3DGS
        elif split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]


    def __len__(self):
        return len(self.indices)


    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        mask = self.parser.mask_dict[camera_id]

        if len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
            "camera_idx": self.parser.camera_indices[
                index
            ],  # 0-based contiguous camera index
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        # Add exposure if available for this image
        exposure = self.parser.exposure_values[index]
        if exposure is not None:
            data["exposure"] = torch.tensor(exposure, dtype=torch.float32)

        if self.load_point_depth:
            if not hasattr(self.parser, "points") or self.parser.points is None:
                raise ValueError(f"The current scene loader ({self.parser.__class__.__name__}) does not support point depth")

            # Projected points to image plane to get depths
            worldtocams = np.linalg.inv(camtoworlds)
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices[image_name]
            points_world = self.parser.points[point_indices]
            points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
            points_proj = (K @ points_cam.T).T
            points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
            depths = points_cam[:, 2]  # (M,)

            # Filter out points outside the image
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            points = points[selector]
            depths = depths[selector]
            data["depth_pixels"] = torch.from_numpy(points).float()
            data["depth_values"] = torch.from_numpy(depths).float()

        if self.load_image_depth:
            if not hasattr(self.parser, "depth_paths") or self.parser.depth_paths is None:
                raise ValueError(f"The current scene loader ({self.parser.__class__.__name__}) does not support image depth")
            
            depth = cv2.imread(self.parser.depth_paths[index], cv2.IMREAD_UNCHANGED)
            depth = depth.astype(np.float32)
            depth = depth * self.parser.depth_scale
            
            if len(params) > 0:
                # Undistort depth image, very unlikely
                depth = cv2.remap(depth, mapx, mapy, cv2.INTER_NEAREST)
                depth = depth[y : y + h, x : x + w]

            if self.patch_size is not None:
                depth = depth[y : y + self.patch_size, x : x + self.patch_size]
  
            data["depth_image"] = torch.from_numpy(depth).float().unsqueeze(-1) # [H, W, 1]

        return data
