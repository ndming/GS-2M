import json
import math
import os
import random
import sys
import time

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union
from typing_extensions import Literal, assert_never

import imageio
import viser
import yaml

import numpy as np
import torch
import torch.nn.functional as F

from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from gsplat import export_splats
from gsplat.color_correct import color_correct_affine, color_correct_quadratic
from gsplat.compression import PngCompression
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat.sampling import sample_geometry
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.utils import depth_to_points

from nerfview import CameraState, RenderTabState, apply_float_colormap
from fused_ssim import fused_ssim

from .datasets import Dataset, get_parser
from .viewer import GsplatViewer, GsplatRenderTabState
from .trajs import generate_ellipse_path_z, generate_interpolated_path, generate_spiral_path
from .utils import (
    AppearanceOptModule, CameraOptModule, PatchMatchModule, build_decoupled_appearance,
    knn, rgb_to_sh, set_random_seed, depths_to_camera_normals, apply_opengl_normals,
    image_grad_weight, fibonacci_sphere_points,
)


@dataclass
class Config:
    # Path to a scene directory
    data_dir: str = "data/scene"
    # Downsample factor for the dataset
    data_factor: int = 1
    # Directory to save results
    result_dir: str = "results"
    # Every N images there is a test image, test_every=0 will use all images for training
    test_every: int = 0
    # Random crop size for training (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to all scene-size parameters
    global_scale: float = 1.1
    # Normalize the world (poses, sparse points, etc.) to fit within a unit cube, mutually exclusive with center_world_space
    normalize_world_space: bool = False
    # Translate the world so that mean camera positions hits the world origin, mutually exclusive with normalize_world_space
    center_world_space: bool = False
    # Camera projection model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"
    # Load EXIF exposure metadata from images (if available)
    load_exposure: bool = False
    # Mask GT images (RGB, depth, normal, etc.) for foreground reconstruction using alpha channel or masks from mask_image_dir
    mask_gt_image: bool = False
    # The directory under data_dir containing masks for GT images, only effective if mask_gt_image is True
    mask_image_dir: str = "masks"
    # The directory under data_dir containing reference depth images, leave empty to skip depth loading
    depth_image_dir: str = ""
    # Depth scale factor to multiply with depth values when loading reference depths (e.g, 1e-3 for mm -> metric)
    depth_image_scale: float = 1e-3
    # If set, skip pre-processing of images and use the existing ones (if available)
    reuse_processed_images: bool = True
    # How many workers to process images in parallel, None to use all CPUs available
    image_process_workers: Optional[int] = None
    # Apply frame subsampling at this interval (useful for long-sequence scenes), ignored by Blender/IDR scenes
    num_stride_frames: int = 1

    # Use this option if the image directory of your COLMAP scene is not named "images"
    colmap_image_dir: str = "images"
    # Rotate COLMAP's native world up to +Z so the world is Z-up, mutually exclusive with normalize_world_space
    colmap_z_up: bool = False
    # The filename extension of image files in Blender datasets
    blender_file_extension: str = "png"
    # How many frames at the start to skip when loading egocentric scenes
    offset_start_frames: int = 0
    # How many frames at the end to skip when loading egocentric scenes
    offset_end_frames: int = 0

    # Initialization strategy for Gaussians
    init_type: Literal["sparse", "random", "sphere"] = "sparse"
    # Initial number of GSs, ignored if init_type is sparse
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiplier of the camera extent, ignored if init_type is sparse
    init_extent: float = 1.0
    # The maximum number of spherical harmonics degrees to use for training
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1_000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # LR for 3D point positions
    means_lr: float = 1.6e-4
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-2
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Path to checkpoint .pt files to pick up training from, all step-options will be offset by the checkpoint's step
    ckpt: Optional[List[str]] = None
    # Number of training steps, offset automatically if training from a checkpoint (via --ckpt)
    max_steps: int = 30_000
    # Steps to evaluate the model, relative to max_steps
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 15_000, 25_000, 30_000])
    # Steps to save the model checkpoint, relative to max_steps
    save_steps: List[int] = field(default_factory=lambda: [7_000, 15_000, 25_000, 30_000])
    # Whether to save .ply file for the training scene (storage size can be large)
    save_ply: bool = True
    # Steps to save the training scene as .ply, relative to max_steps
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 15_000, 25_000, 30_000])
    # Batch size for training, learning rates will be scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps, use this under distributed training or when batch_size > 1
    steps_scaler: float = 1.0

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(default_factory=DefaultStrategy)
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower
    packed: bool = False
    # Use sparse gradients for optimization (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS (experimental)
    visible_adam: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False
    # Use white background in visualization and evaluation
    white_bkgd: bool = False
    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 100.0

    # Disable nerfview viewer during training
    disable_viewer: bool = False
    # Port for the viewer server, only applies if disable_viewer is False
    viewer_port: int = 8080
    # Whether to disable video generation of novel views during evaluation
    disable_video: bool = True
    # The trajectory type along which novel views are rendered for video generation, only applies if disable_video is False
    traj_type: Literal["interp", "ellipse", "spiral"] = "interp"
    # If traj_type is interp, increase this factor to interpolate more frames between each pair of input views
    traj_interp_factor: int = 1
    # If traj_type is ellipse/spiral, the output trajectory will have exactly traj_num_frames frames
    traj_num_frames: int = 120

    # Which network to use for LPIPS metric
    lpips_net: Literal["vgg", "alex"] = "alex"
    # Name of compression strategy to use, None to disable
    compression: Optional[Literal["png"]] = None

    # Project 3D Gaussians using the unscented transform (3DGUT)
    with_ut: bool = False
    # Compute the Gaussian responses in the 3D world space instead of the 2D image space
    with_eval3d: bool = False

    # Enable Mip-Splatting 2D (screen-space) filter, please adjust mip_filter_2d_variance if set
    mip_filter_2d: bool = False
    # The variance of Mip-Splatting 2D filter if mip_filter_2d is True, otherwise the epsilon of EWA filter
    mip_filter_2d_variance:float = 0.3
    # Enable Mip-Splatting 3D (world-space) filter, can be combined with mip_filter_2d
    mip_filter_3d: bool = False
    # Recompute the 3D filter every this many steps, only applies if mip_filter_3d is True
    mip_filter_3d_update_every: int = 100
    # The variance of Mip-Splatting 3D smoothing filter, only applies if mip_filter_3d is True
    mip_filter_3d_variance: float = 0.2

    # Opacity regularization, 0 to disable
    opacity_reg: float = 0.0
    # Scale regularization, 0 to disable
    scale_reg: float = 0.0
    # Alpha regularization, 0 to disable
    alpha_reg: float = 0.0
    # Enforce disk-like 3D Gaussians, 0 to disable
    planar_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics, this is only to test the camera pose optimization
    pose_noise: float = 0.0

    # Enable appearance optimization (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6
    # Enable decoupled appearance to compensate for view-dependent effects, None to disable
    decoupled_appearance: Optional[Literal["pgsr"]] = None
    # Learning rate for the decoupled appearance parameters (only applies if decoupled_appearance)
    decoupled_appearance_lr: float = 1e-3

    # Post-processing method for appearance correction (experimental)
    post_processing: Optional[Literal["bilateral_grid", "ppisp"]] = None
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)
    # Enable PPISP controller
    ppisp_use_controller: bool = True
    # Use controller distillation in PPISP (only applies when post_processing="ppisp" and ppisp_use_controller=True)
    ppisp_controller_distillation: bool = True
    # Activate PPISP controller from this step (only applies when post_processing="ppisp" and ppisp_use_controller=True)
    ppisp_controller_from_step: int = 25_000
    # Color correction method for cc_* metrics (only applies when post_processing is set)
    color_correct_method: Literal["affine", "quadratic"] = "affine"
    # Compute color-corrected metrics (cc_psnr, cc_ssim, cc_lpips) during evaluation
    use_color_correction_metric: bool = False

    # How depths are rendered for regularizations
    depth_render_mode: Optional[Literal["ZD", "PD"]] = None
    # Enable depth loss between rendered depths and depths from sparse points (SfM/LiDAR), 0 to disable
    depth_point_lambda: float = 0.0
    # Start applying depth point loss from this step (only applies when depth_point_lambda > 0)
    depth_point_loss_from_step: int = 5_000
    # Enable depth loss between rendered depths and reference depth images, 0 to disable 
    depth_image_lambda: float = 0.0
    # Start applying depth image loss from this step (only applies when depth_image_lambda > 0)
    depth_image_loss_from_step: int = 5_000
    # The distance (m) beyond which depth values are considered invalid in depth image loss
    depth_image_max_distance: float = 10.0
    # Enforce depth normal consistency, requires a depth_render_mode with normal rendering support, 0 to disable
    depth_normal_lambda: float = 0.0
    # Start applying depth normal consistency loss from this step (only applies when depth_normal_lambda > 0)
    depth_normal_loss_from_step: int = 7_000
    # Use gradients derived from GT images to heuristically down weight depth-normal loss at silhouette
    depth_normal_loss_edge_aware: bool = False

    # Enable multi-view photometric consistency, requires a depth_render_mode with normal rendering support, 0 to disable
    multi_view_ncc_lambda: float = 0.0
    # Enable multi-view geometric consistency, requires a depth_render_mode with normal rendering support, 0 to disable
    multi_view_geo_lambda: float = 0.0
    # Start applying multi-view photometric and/or geometric consistency losses from this step
    multi_view_loss_from_step: int = 7_000
    # Compute and apply multi-view losses every this number, increase to cut down training time
    multi_view_loss_every: int = 1
    # The maximum number of nearest neighbors to consider for mutli-view losses
    multi_view_nearest_max_num: int = 8
    # Only neighbor views whose angular difference within this degree are valid for multi-view neighbor selection
    multi_view_nearest_max_angle: float = 30
    # Only neighbor views whose displacement greater than this value (m) are valid for multi-view neighbor selection
    multi_view_nearest_min_dis: float = 0.01
    # Only neighbor views whose displacement within this value (m) are valid for multi-view neighbor selection
    multi_view_nearest_max_dis: float = 1.5
    # The threshold beyond which reprojected pixel errors are considered noise, smaller numbers discard more pixels
    multi_view_pixel_noise_threshold: float = 1.0
    # The threshold beyond which sampled depths in neighbor views are considered occluded, smaller numbers discard more samples
    multi_view_occlusion_threshold: float = 1e-1
    # Multi-view normal consistency (part of the geo loss) between reference and neighbor normals, 0 to disable
    multi_view_angle_factor: float = 0.0
    # Only neighbor normals within this angular difference (degrees) contribute to the multi-view normal consistency loss
    multi_view_angle_noise_threshold: float = 30.0
    # The maximum number of depth values to sample from nearest views in mutli-view losses, 0 to use all
    multi_view_max_num_samples: int = 0
    # How rapidly multi-view geometrically consistent weights decay as a function of reprojection errors
    multi_view_geo_weights_decay_rate: float = 1.0
    # How rapidly multi-view photometrically consistent weights decay as a function of reprojection errors
    multi_view_ncc_weights_decay_rate: float = 1.0
    # Multi-view observe-trim: prune Gaussians observed in fewer than multi_view_trim_min_views while densification is active
    multi_view_trim: bool = False
    # Every this many steps perform multi_view_trim if enabled
    multi_view_trim_every: int = 1_000
    # How many views to count a Gaussian as observed in multi_view_trim
    multi_view_trim_min_views: int = 2
    # Whether to report during training how many Gaussians are trimmed due to multi_view_trim
    multi_view_trim_verbose: bool = False

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = True

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        self.mip_filter_3d_update_every = int(self.mip_filter_3d_update_every * factor)
        self.ppisp_controller_from_step = int(self.ppisp_controller_from_step * factor)

        self.depth_point_loss_from_step  = int(self.depth_point_loss_from_step  * factor)
        self.depth_image_loss_from_step  = int(self.depth_image_loss_from_step  * factor)
        self.depth_normal_loss_from_step = int(self.depth_normal_loss_from_step * factor)
        self.multi_view_loss_from_step   = int(self.multi_view_loss_from_step   * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.teleport_stop_iter = int(strategy.teleport_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
            if strategy.noise_injection_stop_iter >= 0:
                strategy.noise_injection_stop_iter = int(
                    strategy.noise_injection_stop_iter * factor
                )
        else:
            assert_never(strategy)


def create_splats_with_optimizers(
    parser,
    init_type: str = "sparse",
    init_num_pts: int = 100_000,
    init_extent: float = 1.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
    ckpt_splats: Optional[Dict[str, Tensor]] = None,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if ckpt_splats is not None:
        # LR map mirrors the params list below — means gets scene_scale factor,
        # sh0/features/colors share sh0_lr, everything else is direct.
        lr_map = {
            "means":      means_lr * scene_scale,
            "scales":     scales_lr,
            "quats":      quats_lr,
            "opacities":  opacities_lr,
            "sh0":        sh0_lr,
            "shN":        shN_lr,
            "features":   sh0_lr,
            "colors":     sh0_lr,
        }
        params = [
            (name, torch.nn.Parameter(tensor[world_rank::world_size]), lr_map[name])
            for name, tensor in ckpt_splats.items()
        ]
    else:
        # Init means and base colors based on init_type
        if init_type == "sparse":
            points = torch.from_numpy(parser.points).float()
            rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
        elif init_type == "random":
            points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
            rgbs = torch.rand((init_num_pts, 3))
        elif init_type == "sphere":
            radius = init_extent * scene_scale
            points = torch.from_numpy(fibonacci_sphere_points(init_num_pts, radius)).float()
            rgbs = torch.rand((init_num_pts, 3))
        else:
            raise ValueError("Please specify a correct init_type: sparse, random, or sphere")

        # Initialize the GS size to be the average dist of the 3 nearest neighbors
        dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        dist_avg = torch.sqrt(dist2_avg)
        scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

        # Distribute the GSs to different ranks (also works for single rank)
        points = points[world_rank::world_size]
        rgbs = rgbs[world_rank::world_size]
        scales = scales[world_rank::world_size]

        N = points.shape[0]
        quats = torch.rand((N, 4))  # [N, 4]
        opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

        params = [
            # name, value, lr
            ("means", torch.nn.Parameter(points), means_lr * scene_scale),
            ("scales", torch.nn.Parameter(scales), scales_lr),
            ("quats", torch.nn.Parameter(quats), quats_lr),
            ("opacities", torch.nn.Parameter(opacities), opacities_lr),
        ]

        if feature_dim is None:
            # color is SH coefficients.
            colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
            colors[:, 0, :] = rgb_to_sh(rgbs)
            params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
            params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
        else:
            # features will be used for appearance and view-dependent shading
            features = torch.rand(N, feature_dim)  # [N, feature_dim]
            params.append(("features", torch.nn.Parameter(features), sh0_lr))
            colors = torch.logit(rgbs)  # [N, 3]
            params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            fused=True,
        )
        for name, _, lr in params
    }
    return splats, optimizers


def compute_nearest_indices(
    dataset: Dataset,
    nearest_num: int,
    nearest_max_angle: float,
    nearest_min_dis: float,
    nearest_max_dis: float,
):
    print(f"[>] Populating nearest camera indices for {len(dataset)} views")
    c2w_mats = dataset.parser.camtoworlds  # [total_N, 4, 4], numpy
    centers  = c2w_mats[:, :3, 3]          # [total_N, 3]

    # Forward ray in cam space rotated to world space
    forward_rays = c2w_mats[:, :3, 2]      # [total_N, 3]
    norms = np.linalg.norm(forward_rays, axis=-1, keepdims=True)
    forward_rays = forward_rays / np.maximum(norms, 1e-8)  # [total_N, 3]

    # Map dataset item index to a list of parser-level indices
    nearest_ids: dict[int, list[int]] = {}
    nearst_cnt = 0
    for item_idx, parser_idx in enumerate(dataset.indices):
        # Pairwise distances from this camera to all others in the split
        center_i   = centers[parser_idx]       # [3,]
        ray_i      = forward_rays[parser_idx]  # [3,]

        split_centers = centers[dataset.indices]       # [S, 3]
        split_rays    = forward_rays[dataset.indices]  # [S, 3]

        dists  = np.linalg.norm(split_centers - center_i, axis=-1)  # [S,]
        dots   = np.clip(np.sum(split_rays * ray_i, axis=-1), -1.0, 1.0)
        angles = np.degrees(np.arccos(dots))                        # [S,]

        # Sort by distance primarily, angle as tiebreaker
        sorted_indices = np.lexsort((angles, dists))  # indices into self.indices

        mask = (
            (angles[sorted_indices] < nearest_max_angle) &
            (dists[sorted_indices]  > nearest_min_dis)   &
            (dists[sorted_indices]  < nearest_max_dis)   &
            (sorted_indices != item_idx)  # exclude self
        )
        sorted_indices = sorted_indices[mask]
        top_k = sorted_indices[:nearest_num]
        indices = top_k.tolist() # indices into dataset.indices
        nearest_ids[item_idx] = indices
        nearst_cnt += len(indices)

    print(f"[>] Average nearest cameras per-view: {nearst_cnt / len(nearest_ids):.1f}")
    return nearest_ids


class Runner:
    """Engine for training, testing, and meshing."""

    def __init__(self, local_rank: int, world_rank, world_size: int, cfg: Config) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        # Tensorboard, wipe old contents on a clean run
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")
        if cfg.ckpt is None:
            for file in Path(f"{cfg.result_dir}/tb").glob("events.out.tfevents.*"):
                file.unlink()

        # Data parser, interpret dataset dir and gather paths
        self.parser = get_parser(cfg.data_dir)(
            # Common options
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
            load_exposure=cfg.load_exposure,
            mask_gt_image=cfg.mask_gt_image,
            mask_image_dir=cfg.mask_image_dir,
            reuse_processed_images=cfg.reuse_processed_images,
            image_process_workers=cfg.image_process_workers,
            center_world_space=cfg.center_world_space,
            num_stride_frames=cfg.num_stride_frames,
            depth_image_dir=cfg.depth_image_dir,
            depth_image_scale=cfg.depth_image_scale,
            # Dataset-specific options
            colmap_image_dir=cfg.colmap_image_dir,
            colmap_z_up=cfg.colmap_z_up,
            blender_file_extension=cfg.blender_file_extension,
            offset_start_frames=cfg.offset_start_frames,
            offset_end_frames=cfg.offset_end_frames,
        )
        if self.parser.num_cameras > 1 and cfg.batch_size != 1:
            raise ValueError(
                f"When using multiple cameras ({self.parser.num_cameras} found), batch_size must be 1, "
                f"but got batch_size={cfg.batch_size}."
            )

        # Data splits
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_point_depth=cfg.depth_point_lambda > 0.0,
            load_image_depth=cfg.depth_image_lambda > 0.0,
            load_image_gray=cfg.multi_view_ncc_lambda > 0.0,
        )
        self.valset = Dataset(
            self.parser,
            split="val",
            load_image_depth=cfg.depth_image_lambda > 0.0,
        )

        # Per-frame indices of nearest cameras, only used for multi-view losses
        if cfg.multi_view_geo_lambda > 0.0 or cfg.multi_view_ncc_lambda > 0.0:
            self.trainset_nearest_indices = compute_nearest_indices(
                dataset=self.trainset,
                nearest_num=cfg.multi_view_nearest_max_num,
                nearest_max_angle=cfg.multi_view_nearest_max_angle,
                nearest_min_dis=cfg.multi_view_nearest_min_dis,
                nearest_max_dis=cfg.multi_view_nearest_max_dis,
            )

        # Scene half extent, matters for metric quantities
        self.scene_scale = self.parser.scene_scale * cfg.global_scale
        print("[>] Scene half extent:", self.scene_scale)

         # Load checkpoint splats, if any
        ckpt_splats = None
        ckpt_pose_adjust_state = None
        ckpt_app_state = None
        ckpt_post_processing_state = None
        ckpt_decoupled_app_state = None
        self.ckpt_step = -1
        if cfg.ckpt is not None:
            ckpts = [
                torch.load(file, map_location=self.device, weights_only=True)
                for file in cfg.ckpt
            ]
            # Reconstruct full (unsharded) tensors from per-rank checkpoints
            ckpt_splats = {
                k: torch.cat([ckpt["splats"][k] for ckpt in ckpts])
                for k in ckpts[0]["splats"].keys()  # use rank-0 to get keys
            }
            self.ckpt_step = ckpts[0]["step"]
            ckpt_app_state = ckpts[0].get("app_module")
            ckpt_pose_adjust_state = ckpts[0].get("pose_adjust")
            ckpt_post_processing_state = ckpts[0].get("post_processing")
            ckpt_decoupled_app_state = ckpts[0].get("decoupled_appearance")
            print(f"[>] Loaded model from step {self.ckpt_step}")

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
            ckpt_splats=ckpt_splats,
        )
        print("[>] Model initialized. Number of GS:", len(self.splats["means"]))

        # Mip-Splatting 3D filter buffer (per-Gaussian scale dilation size). Recomputed
        # from the training cameras periodically; None when mip_filter_3d is disabled.
        self.filter_3d = None
        if cfg.mip_filter_3d:
            self.compute_3d_filter()

        # Convert the chosen depth render mode to gsplat's render mode
        self.render_mode = f"RGB+{cfg.depth_render_mode}" if cfg.depth_render_mode is not None else "RGB"

        # Densification strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)
        self.cfg.strategy.refine_stop_iter += self.ckpt_step + 1
        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(scene_scale=self.scene_scale)
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Compression strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")
        
        # Enable pose optimization if requested
        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            if ckpt_pose_adjust_state is not None:
                self.pose_adjust.load_state_dict(ckpt_pose_adjust_state)
                print("[>] Loaded pose adjust state from checkpoint")
            else:
                self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)
        elif ckpt_pose_adjust_state is not None:
            print("[!] Found pose adjust state in checkpoint but pose_opt is off, "
                  " perhaps you forgot to enable pose_opt for this session?")
        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        # Enable per-frame latent appearance if requested
        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            if ckpt_app_state is not None:
                self.app_module.load_state_dict(ckpt_app_state)
                print("[>] Loaded per-frame latent appearance state from checkpoint")
            else:
                # Initialize the last layer to be zero so that the initial output is zero.
                torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
                torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)
        elif ckpt_app_state is not None:
            print("[!] Found per-frame latent appearance state in checkpoint but app_opt is off, "
                  " perhaps you forgot to enable app_opt for this session?")

        # Enable decoupled appearance (distinct from AppearanceOptModule) if requested
        self.decoupled_app_optimizers = []
        if cfg.decoupled_appearance is not None:
            self.appearance_adjust = build_decoupled_appearance(
                cfg.decoupled_appearance, len(self.trainset)
            ).to(self.device)
            if ckpt_decoupled_app_state is not None:
                self.appearance_adjust.load_state_dict(ckpt_decoupled_app_state)
                print("[>] Loaded decoupled appearance state from checkpoint")
            self.decoupled_app_optimizers = [
                torch.optim.Adam(
                    self.appearance_adjust.parameters(),
                    lr=cfg.decoupled_appearance_lr * math.sqrt(cfg.batch_size),
                    betas=(0.9, 0.99),  # matches reference implementation
                ),
            ]
            if world_size > 1:
                self.appearance_adjust = DDP(self.appearance_adjust)
        elif ckpt_decoupled_app_state is not None:
            print("[!] Found decoupled appearance state in checkpoint but decoupled_appearance is None, "
                  " perhaps you forgot to pass decoupled_appearance for this session?")

        # Import post-processing modules based on configuration
        # These imports must be here (not in __main__) for distributed workers
        if cfg.post_processing == "bilateral_grid":
            global BilateralGrid, slice, total_variation_loss
            from fused_bilagrid import BilateralGrid, slice, total_variation_loss
        elif cfg.post_processing == "ppisp":
            global PPISP, PPISPConfig, export_ppisp_report
            from ppisp import PPISP, PPISPConfig
            from ppisp.report import export_ppisp_report

        self.post_processing_module = None
        self._gaussians_frozen = False  # track if Gaussians are frozen for controller distillation
        if cfg.post_processing == "bilateral_grid":
            self.post_processing_module = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
        elif cfg.post_processing == "ppisp":
            ppisp_config = PPISPConfig(
                use_controller=cfg.ppisp_use_controller,
                controller_distillation=cfg.ppisp_controller_distillation,
                controller_activation_ratio=cfg.ppisp_controller_from_step / cfg.max_steps,
            )
            self.post_processing_module = PPISP(
                num_cameras=self.parser.num_cameras,
                num_frames=len(self.trainset),
                config=ppisp_config,
            ).to(self.device)

        self.post_processing_optimizers = []
        if cfg.post_processing == "bilateral_grid":
            self.post_processing_optimizers = [
                torch.optim.Adam(
                    self.post_processing_module.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]
        elif cfg.post_processing == "ppisp":
            self.post_processing_optimizers = (
                self.post_processing_module.create_optimizers()
            )

        # Load post-processing state if present in checkpoint
        if self.post_processing_module is not None and ckpt_post_processing_state is not None:
            if cfg.post_processing == "ppisp":
                module = self.post_processing_module
                ckpt_num_frames = ckpt_post_processing_state["exposure_params"].shape[0]

                if ckpt_num_frames == module.num_frames:
                    # Shapes match (maybe we're resuming training on the same dataset)
                    module.load_state_dict(ckpt_post_processing_state)
                    print("[>] Loaded PPISP state from checkpoint, assuming the same training set!")
                else:
                    # Per-frame shape mismatch (e.g. fine-tuning on a different dataset):
                    # load per-camera params directly and initialize per-frame params from
                    # checkpoint mean so PPISP starts from the average learned correction
                    module.vignetting_params.data.copy_(ckpt_post_processing_state["vignetting_params"])
                    module.crf_params.data.copy_(ckpt_post_processing_state["crf_params"])
        
                    # Per-frame params: initialize from checkpoint mean rather than zero,
                    # so PPISP starts from the average correction seen during pre-training
                    module.exposure_params.data.fill_(
                        ckpt_post_processing_state["exposure_params"].mean().item()
                    )
                    module.color_params.data.copy_(
                        ckpt_post_processing_state["color_params"]
                            .mean(dim=0, keepdim=True)
                            .expand_as(module.color_params)
                    )
        
                    # Controller weights: load directly (shape doesn't depend on num_frames)
                    if cfg.ppisp_use_controller and len(module.controllers) > 0:
                        controller_keys = {
                            k: v for k, v in ckpt_post_processing_state.items()
                            if k.startswith("controllers.")
                        }
                        module.load_state_dict(controller_keys, strict=False)

                    print(
                        f"[>] Loaded PPISP per-camera state from checkpoint "
                        f"(frame count changed {ckpt_num_frames} -> {module.num_frames}, "
                        f"per-frame params initialized from checkpoint mean)"
                    )
            else:
                # Bilateral grid or other: shape matches, load directly
                self.post_processing_module.load_state_dict(ckpt_post_processing_state)
                print("[>] Loaded post-processing state from checkpoint")
        
        # Losses & metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        # Lpips
        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")
        
        # Patch matching module for multi-view consistency losses
        self.pm = PatchMatchModule(
            pixel_noise_threshold=cfg.multi_view_pixel_noise_threshold,
            occlusion_threshold=cfg.multi_view_occlusion_threshold * self.scene_scale,
            max_num_samples=cfg.multi_view_max_num_samples,
            angle_factor=cfg.multi_view_angle_factor,
            angle_noise_threshold=cfg.multi_view_angle_noise_threshold,
            geo_weights_decay_rate=cfg.multi_view_geo_weights_decay_rate,
            ncc_weights_decay_rate=cfg.multi_view_ncc_weights_decay_rate,
            optimize_ncc=cfg.multi_view_ncc_lambda > 0.0,
            optimize_geo=cfg.multi_view_geo_lambda > 0.0,
            device=self.device,
        )

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.viewer_port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

    @torch.no_grad()
    def compute_3d_filter(self):
        """Mip-Splatting 3D filter: per-Gaussian world-space low-pass size from the
        maximum sampling rate across training views (reference: RaDe-GS).
        Sets self.filter_3d to [N, 1] (distance / focal * sqrt(0.2))."""

        means = self.splats["means"]  # [N, 3]
        N = means.shape[0]
        device = means.device
        distance = torch.full((N,), float("inf"), device=device)
        valid_points = torch.zeros(N, dtype=torch.bool, device=device)
        focal = 0.0

        for idx in self.trainset.indices:
            c2w = torch.from_numpy(self.parser.camtoworlds[idx]).float().to(device)
            w2c = torch.linalg.inv(c2w)
            R = w2c[:3, :3]  # world -> camera rotation
            t = w2c[:3, 3]   # world -> camera translation
            cid = self.parser.camera_ids[idx]
            K = self.parser.Ks_dict[cid]
            fx, fy = float(K[0, 0]), float(K[1, 1])
            W, H = self.parser.imsize_dict[cid]

            xyz_cam = means @ R.T + t  # [N, 3]
            z = xyz_cam[:, 2]
            uv = xyz_cam[:, :2] / z.unsqueeze(-1).clamp_min(1e-6)
            in_screen = (uv[:, 0].abs() <= W / fx * 0.575) & (
                uv[:, 1].abs() <= H / fy * 0.575
            )
            valid = (z > 0.2) & in_screen

            distance = torch.where(valid, torch.minimum(distance, z), distance)
            valid_points |= valid
            focal = max(focal, fx)

        if valid_points.any():
            distance[~valid_points] = distance[valid_points].max()
        else:
            distance.fill_(1.0)  # degenerate: no Gaussian seen by any camera
        self.filter_3d = (
            distance / focal * (self.cfg.mip_filter_3d_variance ** 0.5)
        ).unsqueeze(-1)  # [N, 1]

    def rasterize(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
        skip_post: bool = False,
        frame_idcs: Optional[Tensor] = None,
        camera_idcs: Optional[Tensor] = None,
        exposure: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        # Model params
        means: Tensor = self.splats["means"]  # [N, 3]
        quats = self.splats["quats"]  # [N, 4], will be normalized internally
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        # Mip-Splatting 3D filter: dilate the 3D scales by the per-Gaussian
        # filter size and compensate opacity for the added volume.
        if self.cfg.mip_filter_3d and self.filter_3d is not None:
            s2 = scales * scales                    # [N, 3]
            f2 = self.filter_3d * self.filter_3d    # [N, 1], broadcasts over the 3 axes
            scales = torch.sqrt(s2 + f2)            # [N, 3]
            coef = torch.sqrt(s2.prod(dim=-1) / (s2 + f2).prod(dim=-1))  # [N,]
            opacities = opacities * coef

        # Rasterization does not support image_ids, pop it from kwargs
        # to use it for appearance optimization if needed
        image_ids = kwargs.pop("image_ids", None)

        # Colors provided upstream has the most priority
        colors = kwargs.pop("colors", None)
        if colors is None:
            if self.cfg.app_opt:
                # Colors from appearance latents
                colors = self.app_module(
                    features=self.splats["features"],
                    embed_ids=image_ids,
                    dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                    sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
                )
                colors = colors + self.splats["colors"]
                colors = torch.sigmoid(colors)
            else:
                # Colors from SHs
                colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        # Default rasterize mode and camera model if omitted
        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.mip_filter_2d else "classic"
        if camera_model is None:
            camera_model = self.cfg.camera_model

        eps2d = kwargs.pop("eps2d", self.cfg.mip_filter_2d_variance)
        near_plane = kwargs.pop("near_plane", self.cfg.near_plane)
        far_plane = kwargs.pop("far_plane", self.cfg.far_plane)

        # Invoke gsplat's rasterization backend
        render_colors, render_alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            near_plane=near_plane,
            far_plane=far_plane,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            eps2d=eps2d,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=camera_model,
            with_ut=self.cfg.with_ut,
            with_eval3d=self.cfg.with_eval3d,
            **kwargs,
        )

        # Apply masks if provied
        if masks is not None:
            render_colors[~masks] = 0

        # Apply post-processing to rendered RGB if provided
        if not skip_post and self.cfg.post_processing is not None and "RGB" in kwargs["render_mode"]:
            # Create pixel coordinates [H, W, 2] with +0.5 center offset
            pixel_y, pixel_x = torch.meshgrid(
                torch.arange(height, device=self.device) + 0.5,
                torch.arange(width, device=self.device) + 0.5,
                indexing="ij",
            )
            pixel_coords = torch.stack([pixel_x, pixel_y], dim=-1)  # [H, W, 2]

            # Split RGB from extra channels (e.g. depth) for post-processing
            rgb = render_colors[..., :3]
            extra = render_colors[..., 3:] if render_colors.shape[-1] > 3 else None

            if self.cfg.post_processing == "bilateral_grid":
                if frame_idcs is not None:
                    grid_xy = (
                        pixel_coords / torch.tensor([width, height], device=self.device)
                    ).unsqueeze(0)
                    rgb = slice(
                        self.post_processing_module,
                        grid_xy.expand(rgb.shape[0], -1, -1, -1),
                        rgb,
                        frame_idcs.unsqueeze(-1),
                    )["rgb"]
            elif self.cfg.post_processing == "ppisp":
                camera_idx = camera_idcs.item() if camera_idcs is not None else None
                frame_idx = frame_idcs.item() if frame_idcs is not None else None
                rgb = self.post_processing_module(
                    rgb=rgb,
                    pixel_coords=pixel_coords,
                    resolution=(width, height),
                    camera_idx=camera_idx,
                    frame_idx=frame_idx,
                    exposure_prior=exposure,
                )

            render_colors = (torch.cat([rgb, extra], dim=-1) if extra is not None else rgb)

        return render_colors, render_alphas, meta

    def sample(self, points2d, camtoworlds, Ks, width, height, want_normals):
        """Sample this model's surface geometry at query pixels in a given camera."""

        # Params for sampling
        cfg = self.cfg
        means = self.splats["means"]
        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"])

        # Same 3D Mip-filter scale dilation + opacity compensation as rasterize().
        if cfg.mip_filter_3d and self.filter_3d is not None:
            s2 = scales * scales
            f2 = self.filter_3d * self.filter_3d
            scales = torch.sqrt(s2 + f2)
            opacities = opacities * torch.sqrt(s2.prod(dim=-1) / (s2 + f2).prod(dim=-1))

        return sample_geometry(
            means, quats, scales, opacities,
            torch.linalg.inv(camtoworlds), Ks, width, height, points2d,
            near_plane=cfg.near_plane, far_plane=cfg.far_plane,
            eps2d=cfg.mip_filter_2d_variance, sample_normals=want_normals,
            geometry_mode=2,  # plane depth
        )

    @torch.no_grad()
    def _observe_trim(self):
        """Multi-view observe-trim: prune Gaussians that contribute (T > 0.5)
        in fewer than  `multi_view_trim_min_views` train views.
        """
        cfg = self.cfg
        device = self.device
        means = self.splats["means"]
        N = means.shape[0]

        observe_cnt = torch.zeros(N, dtype=torch.int32, device=device)
        for data in self.trainset:
            c2w = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            if c2w.dim() == 2:
                c2w, Ks = c2w[None], Ks[None]
            H, W = int(data["image"].shape[0]), int(data["image"].shape[1])
            _, _, meta = self.rasterize(
                camtoworlds=c2w, Ks=Ks, width=W, height=H, skip_post=True,
                # Skip colors / SHs entirely, set colors to a dummy tensor
                # with correct shape to satisfy internal asserts
                colors = means.new_zeros(N, 1),  # [N, 1]
                sh_degree=None, render_mode="RGB", count_observe=True,
            )
            obs = meta.get("out_observe", None)
            if obs is not None and obs.numel() == N:
                observe_cnt += (obs > 0).to(torch.int32)

        prune_mask = observe_cnt < cfg.multi_view_trim_min_views
        n_prune = int(prune_mask.sum())
        # Never prune all Gaussians away.
        if 0 < n_prune < N:
            from gsplat.strategy.ops import remove
            remove(self.splats, self.optimizers, self.strategy_state, prune_mask)
        return n_prune

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        # Steps are offset if we continue training from a checkpoint
        max_steps = cfg.max_steps
        init_step = self.ckpt_step + 1
        last_step = self.ckpt_step + cfg.max_steps

        # GS means has a learning rate schedule, that end at 0.01 of the initial value
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        # Pose optimization has a learning rate schedule
        if cfg.pose_opt:
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        # Post-processing module has a learning rate schedule
        if cfg.post_processing == "bilateral_grid":
            # Linear warmup + exponential decay
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.post_processing_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.post_processing_optimizers[0],
                            gamma=0.01 ** (1.0 / max_steps),
                        ),
                    ]
                )
            )
        elif cfg.post_processing == "ppisp":
            ppisp_schedulers = self.post_processing_module.create_schedulers(
                self.post_processing_optimizers,
                max_optimization_iters=max_steps,
            )
            schedulers.extend(ppisp_schedulers)

        # Data loader
        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        train_loader_iter = iter(train_loader)

        # Timer and progress bar
        global_tic = time.time()
        pbar = tqdm(range(init_step, last_step + 1), desc="[>] Training", ncols=150)

        # Multi-view losses may only activate at intervals, so we hoist log
        # vars out of the training loop to prevent them from hitting zeros
        mv_loss, ncc_loss, geo_loss = 0.0, 0.0, 0.0

        # Optimization loop
        for step in pbar:
            # Pause training if viewer wants to
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            # Freeze Gaussians when PPISP controller distillation starts
            if (
                cfg.post_processing == "ppisp"
                and cfg.ppisp_use_controller
                and cfg.ppisp_controller_distillation
                and step >= cfg.ppisp_controller_from_step + self.ckpt_step + 1
                and not self._gaussians_frozen
            ):
                # Freeze all Gaussian parameters for controller distillation.
                # This prevents Gaussians from being updated by any loss (including regularization)
                # while the controller learns to predict per-frame corrections.
                for _, param in self.splats.items():
                    param.requires_grad = False

                self._gaussians_frozen = True
                tqdm.write("[>] Distillation: Gaussian parameters frozen")

            # Sample the next batch (B) of views (most of the time B=1)
            try:
                data = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(train_loader)
                data = next(train_loader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [B, 4, 4]
            Ks = data["K"].to(device)                    # [B, 3, 3]
            gt_image = data["image"].to(device) / 255.0  # [B, H, W, 3]
            gt_alpha = data["alpha"].to(device) / 255.0  # [B, H, W, 1]

            num_train_rays_per_step = gt_image.shape[0] * gt_image.shape[1] * gt_image.shape[2]  # B * H * W
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None             # [B, H, W]
            exposure = data["exposure"].to(device) if "exposure" in data else None  # [B,]

            height, width = gt_image.shape[1:3]

            # Inject noise and nudge poses to refine them
            pose_loss = 0.0
            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)
            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # Periodically increase SH degree
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # The rendered depth/normal channels are only consumed once a geometry loss kicks in.
            # Until then, render RGB-only to skip the geometry overhead.
            geometry_active = self.render_mode != "RGB" and (
                   (cfg.depth_normal_lambda > 0.0 and step >= cfg.depth_normal_loss_from_step)
                or (cfg.depth_point_lambda  > 0.0 and step >= cfg.depth_point_loss_from_step)
                or (cfg.depth_image_lambda  > 0.0 and step >= cfg.depth_image_loss_from_step)
                or ((cfg.multi_view_geo_lambda > 0.0 or cfg.multi_view_ncc_lambda > 0.0)
                    and step >= cfg.multi_view_loss_from_step)
            )
            render_mode_step = self.render_mode if geometry_active else "RGB"

            # Forward pass
            renders, alphas, meta = self.rasterize(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                image_ids=image_ids,
                render_mode=render_mode_step,
                masks=masks,
                frame_idcs=image_ids,
                camera_idcs=data["camera_idx"].to(device),
                exposure=exposure,
            )

            # Obtain the rendered maps
            if render_mode_step == "RGB":
                colors, depths, _, normals = renders[..., 0:3], None, None, None
            elif render_mode_step == "RGB+ZD":
                colors, depths, _, normals = renders[..., 0:3], renders[..., 3:4], renders[..., 4:5], None
            else:  # plane depth
                colors, depths, _, normals = renders[..., 0:3], renders[..., 3:4], None, renders[..., 5:8]

            # Set random background colors if requested
            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            # While Gaussians are frozen for controller distillation (PPISP) the
            # render output has requires_grad=False, so densification bookkeeping
            # (e.g. DefaultStrategy's retain_grad) is both invalid and unnecessary.
            if not self._gaussians_frozen:
                self.cfg.strategy.step_pre_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=meta,
                )

            # Photometric loss: L1 and Lssim between rendered and GT color images.
            # Decoupled appearance transforms the render for the L1 term only; its
            # module returns the (possibly cropped) appearance L1 directly.
            if cfg.decoupled_appearance is not None:
                Ll1 = self.appearance_adjust(colors, gt_image, image_ids)
            else:
                Ll1 = F.l1_loss(colors, gt_image)
            Lssim = 1.0 - fused_ssim(colors.permute(0, 3, 1, 2), gt_image.permute(0, 3, 1, 2), padding="valid")
            loss = (1.0 - cfg.ssim_lambda) * Ll1 + cfg.ssim_lambda * Lssim

            # Depth losses
            depth_loss = 0.0
            depth_point_kicked_in = cfg.depth_point_lambda > 0.0 and step >= cfg.depth_point_loss_from_step
            depth_image_kicked_in = cfg.depth_image_lambda > 0.0 and step >= cfg.depth_image_loss_from_step

            # Ld: supervise rendered depths with prior depth points
            if depth_point_kicked_in:
                # Pixels and corresponding depth values of this view batch
                depth_pixels = data["depth_pixels"].to(device) # [B, M, 2]
                depth_values = data["depth_values"].to(device) # [B, M]

                # Prepare depth pixels for grid sampling into rendered depth map
                depth_pixels = torch.stack(
                    [
                        depth_pixels[:, :, 0] / (width  - 1) * 2 - 1,
                        depth_pixels[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                ) # normalize to [-1, 1]
                grid = depth_pixels.unsqueeze(2)  # [B, M, 1, 2]
                disp_gt = 1.0 / depth_values  # [B, M]

                # Sample the rendered depth at the prior points; compare in disparity space
                sampled = F.grid_sample(depths.permute(0, 3, 1, 2), grid, align_corners=True)  # [B, 1, M, 1]
                sampled = sampled.squeeze(3).squeeze(1)  # [B, M]
                disp = torch.where(sampled > 0.0, 1.0 / sampled, torch.zeros_like(sampled))
                depth_point_loss = F.l1_loss(disp, disp_gt) * self.scene_scale

                depth_loss += depth_point_loss.item()
                loss += cfg.depth_point_lambda * depth_point_loss

            # Ld: supervise rendered depths with prior depth maps, assumming both have the same scale
            if depth_image_kicked_in:
                depth_image = data["depth_image"].to(device)  # [B, H, W, 1]

                # Per-variant validity mask (its own depth must be positive)
                m = (depth_image > 0) & (depth_image < cfg.depth_image_max_distance) & (depths > 0)
                depth_image_loss = F.l1_loss(depths[m], depth_image[m]) if m.any() else depths.new_zeros(())

                depth_loss += depth_image_loss.item()
                loss += cfg.depth_image_lambda * depth_image_loss

            # Ldn: enforce consistency between rendered normals and normals dervied from rendered depths
            dn_loss = 0.0
            depth_normal_kicked_in = cfg.depth_normal_lambda > 0.0 and step >= cfg.depth_normal_loss_from_step
            if depth_normal_kicked_in:
                # Optional edge weighting from the GT image gradient, precompute once here
                # in case we have to invoke dn loss two times for the two depth variants
                edge_w = None
                if cfg.depth_normal_loss_edge_aware:
                    edge_w = (1.0 - image_grad_weight(gt_image)).clamp(0, 1).detach() ** 2

                # Normals from this depth variant (camera space) vs the rendered normals
                depth_normals, valid = depths_to_camera_normals(depths, Ks)        # [B, H, W, 3], [B, H, W, 1]
                error = (normals - depth_normals * alphas.detach()).abs().sum(-1, keepdim=True)  # [B, H, W, 1]
                # Weight only pixels with a valid depth neighborhood the render covers;
                # optionally down-weight silhouettes/edges via the GT image gradient.
                weights = (valid & (normals.norm(dim=-1, keepdim=True) > 1e-6)).float()
                if edge_w is not None:
                    weights = weights * edge_w
                depth_normal_loss = (weights * error).sum() / (weights.sum() + 1e-6)  # weighted mean over valid

                dn_loss = depth_normal_loss.item()
                loss += cfg.depth_normal_lambda * depth_normal_loss

            # Multi-view losses
            multi_view_kicked_in = (cfg.multi_view_geo_lambda > 0.0 
                or cfg.multi_view_ncc_lambda > 0.0) and step >= cfg.multi_view_loss_from_step
            if step % cfg.multi_view_loss_every == 0 and multi_view_kicked_in:
                nearest_indices = self.trainset_nearest_indices[data["image_id"].item()]
                if len(nearest_indices) > 0:
                    # We need to create batch dim for data_nearest because we
                    # randomly index into the dataset class, not via data loader.
                    data_nea = self.trainset[random.choice(nearest_indices)]
                    c2w_nea  = data_nea["camtoworld"].unsqueeze(0).to(device)  # [1, 4, 4]
                    K_nea    = data_nea["K"].unsqueeze(0).to(device)           # [1, 3, 3]
                    H_nea, W_nea = data_nea["image"].shape[:2]

                    # Sample the neighbour view's surface geometry directly from the
                    # Gaussians at requested pixels, instead of rendering a full
                    # neighbour depth/normal map. Grad flows to the Gaussians only
                    # when optimizing geometric consistency.
                    def sample_fn(points2d, want_normals):
                        with torch.set_grad_enabled(cfg.multi_view_geo_lambda > 0.0):
                            return self.sample(points2d, c2w_nea, K_nea, W_nea, H_nea, want_normals)

                    Lncc, Lgeo = self.pm(data, data_nea, depths, normals, sample_fn)
                    multi_view_loss = cfg.multi_view_ncc_lambda * Lncc + cfg.multi_view_geo_lambda * Lgeo

                    mv_loss, ncc_loss, geo_loss = multi_view_loss.item(), Lncc.item(), Lgeo.item()
                    loss += multi_view_loss

            # Post-processing losses
            post_loss = 0.0
            if cfg.post_processing == "bilateral_grid":
                post_processing_reg_loss = 10 * total_variation_loss(self.post_processing_module.grids)
                post_loss += post_processing_reg_loss.item()
                loss += post_processing_reg_loss
            elif cfg.post_processing == "ppisp":
                post_processing_reg_loss = (self.post_processing_module.get_regularization_loss())
                post_loss += post_processing_reg_loss.item()
                loss += post_processing_reg_loss

            # Regularizations
            if cfg.opacity_reg > 0.0:
                opacity_loss = torch.sigmoid(self.splats["opacities"]).mean()
                loss += cfg.opacity_reg * opacity_loss
            if cfg.scale_reg > 0.0:
                scales = torch.exp(self.splats["scales"])  # [N, 3]
                if self.cfg.mip_filter_3d and self.filter_3d is not None:
                    scales = torch.sqrt(scales * scales + self.filter_3d * self.filter_3d)  # [N, 3]
                scale_loss = scales.mean()
                loss += cfg.scale_reg * scale_loss
            if cfg.alpha_reg > 0.0:
                alpha_loss = F.binary_cross_entropy(alphas, gt_alpha)
                loss += cfg.alpha_reg * alpha_loss
            if cfg.planar_reg > 0.0:
                radii = meta["radii"]                    # [..., B, N, 2]
                valid_per_cam = (radii > 0).any(dim=-1)  # [..., B, N]
                visibility_filter = valid_per_cam.any(dim=-2)  # [..., N]
                if visibility_filter.sum() > 0:
                    scales = torch.exp(self.splats["scales"])  # [N, 3]
                    if self.cfg.mip_filter_3d and self.filter_3d is not None:
                        scales = torch.sqrt(scales * scales + self.filter_3d * self.filter_3d)  # [N, 3]
                    sorted_scales, _ = torch.sort(scales[visibility_filter], dim=-1)
                    min_axes = sorted_scales[..., 0]
                    planar_loss = min_axes.mean()
                    loss += cfg.planar_reg * planar_loss

            # Propagate gradients backward
            loss.backward()

            # Save checkpoint before updating states
            if step in [i + self.ckpt_step for i in cfg.save_steps] or step == last_step:
                # Save runtime stats
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem_gb": mem,
                    "points": len(self.splats["means"]),
                    "ellapsed_time": time.time() - global_tic,
                }
                with open(f"{self.stats_dir}/train_{step + 1}_rank{self.world_rank}.json", "w") as f:
                    json.dump(stats, f)

                # Report runtime stats
                elapsed_minutes = stats["ellapsed_time"] / 60
                tqdm.write(f"[>] After {(step + 1):>5} steps: mem = {mem:.2f}GB, elapsed time = {elapsed_minutes:.2f}mins")

                def _get_module_state_dict(m):
                    return m.module.state_dict() if world_size > 1 else m.state_dict()

                # Save checkpoint
                model = { "step": step, "splats": self.splats.state_dict() }
                if cfg.pose_opt:
                    model["pose_adjust"] = _get_module_state_dict(self.pose_adjust)
                if cfg.app_opt:
                    model["app_module"] = _get_module_state_dict(self.app_module)
                if cfg.decoupled_appearance is not None:
                    model["decoupled_appearance"] = _get_module_state_dict(self.appearance_adjust)
                if self.post_processing_module is not None:
                    model["post_processing"] = self.post_processing_module.state_dict()
                torch.save(model, f"{self.ckpt_dir}/step{step}_rank{self.world_rank}.pt")

            # Save Gaussian point cloud if requested
            if cfg.save_ply and (step in [i + self.ckpt_step for i in cfg.ply_steps] or step == last_step):
                if self.cfg.app_opt:
                    # Eval at origin to bake the appeareance into the colors
                    rgb = self.app_module(
                        features=self.splats["features"],
                        embed_ids=None,
                        dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                        sh_degree=sh_degree_to_use,
                    )
                    rgb = rgb + self.splats["colors"]
                    rgb = torch.sigmoid(rgb).squeeze(0).unsqueeze(1)
                    sh0 = rgb_to_sh(rgb)
                    shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
                else:
                    sh0 = self.splats["sh0"]
                    shN = self.splats["shN"]
                
                means = self.splats["means"]
                scales = self.splats["scales"]
                quats = self.splats["quats"]
                opacities = self.splats["opacities"]

                export_splats(
                    means=means,
                    scales=scales,
                    quats=quats,
                    opacities=opacities,
                    sh0=sh0,
                    shN=shN,
                    format="ply",
                    save_to=f"{self.ply_dir}/point_cloud_{step + 1}.ply",
                )

            # Update progress bar with postfix losses
            if world_rank == 0 and step % 100 == 0:
                postfix_dict = {
                    "SH": f"{sh_degree_to_use}",
                    "Loss": f"{loss.item():.4f}",
                }
                if depth_point_kicked_in or depth_image_kicked_in:
                    postfix_dict["Ld"] = f"{depth_loss:.4f}"
                if depth_normal_kicked_in:
                    postfix_dict["Ldn"] = f"{dn_loss:.4f}"
                if multi_view_kicked_in:
                    postfix_dict["Lmv"] = f"{mv_loss:.4f}"
                if cfg.pose_opt and cfg.pose_noise:
                    pose_loss = F.l1_loss(camtoworlds_gt, camtoworlds).item()
                    postfix_dict["Lpose"] = f"{pose_loss:.4f}"
                if cfg.post_processing is not None:
                    postfix_dict["Lpost"] = f"{post_loss:.4f}"
                # Number of active Gaussians
                n_points = len(self.splats["means"])
                postfix_dict["N"] = f"{n_points}"
                pbar.set_postfix(postfix_dict)

            # Update tensorboard scalar values
            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024 ** 3
                self.writer.add_scalar("train/Mem_GB", mem, step)
                self.writer.add_scalar("train/Points", len(self.splats["means"]), step)
                self.writer.add_scalar("train/Loss", loss.item(), step)
                self.writer.add_scalar("train/Ll1", Ll1.item(), step)
                self.writer.add_scalar("train/Lssim", Lssim.item(), step)
                self.writer.add_scalar("train/Ld", depth_loss, step)
                self.writer.add_scalar("train/Ldn", dn_loss, step)
                self.writer.add_scalar("train/Lmv", mv_loss, step)
                self.writer.add_scalar("train/Lncc", ncc_loss, step)
                self.writer.add_scalar("train/Lgeo", geo_loss, step)
                self.writer.add_scalar("train/Lpose", pose_loss, step)
                self.writer.add_scalar("train/Lpost", post_loss, step)
                self.writer.flush()

            # Turn gradients into Sparse Tensor before running optimizer if requested
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = meta["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],   # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                # gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(self.splats["opacities"], dtype=bool)
                    visibility_mask.scatter_(0, meta["gaussian_ids"], 1)
                else:
                    visibility_mask = (meta["radii"] > 0).all(-1).any(0)

            # Step optimizers
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.decoupled_app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.post_processing_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Run post-backward steps after backward and optimizer
            if self._gaussians_frozen:
                # Skip structural updates while Gaussians are frozen for controller distillation (PPISP)
                pass
            elif isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=meta,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=meta,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # Keep the Mip-Splatting 3D filter in sync: densification may have changed the
            # Gaussian count (size guard), and it is refreshed periodically as means move.
            if cfg.mip_filter_3d and (
                self.filter_3d is None
                or self.filter_3d.shape[0] != self.splats["means"].shape[0]
                or step % cfg.mip_filter_3d_update_every == 0
            ):
                self.compute_3d_filter()

            # PGSR multi-view observe-trim: every multi_view_trim steps while densification is
            # active, prune Gaussians observed in fewer than the required number of train views.
            # Runs after the strategy's own densify/prune.
            if (
                cfg.multi_view_trim
                and not self._gaussians_frozen
                and step > 0
                and step % cfg.multi_view_trim_every == 0
                and step < self.cfg.strategy.refine_stop_iter
            ):
                n_trim = self._observe_trim()
                if cfg.multi_view_trim_verbose and world_rank == 0 and n_trim > 0:
                    tqdm.write(f"[>] Step {step}: observe-trim pruned {n_trim} Gaussians")
                if n_trim > 0 and cfg.mip_filter_3d and self.filter_3d is not None:
                    self.compute_3d_filter()  # refresh 3D filter when Gaussians are trimmed

            # Eval on train/val set and render trajectory if requested
            if step in [i + self.ckpt_step for i in cfg.eval_steps]:
                eval_stage = "val"
                if getattr(self.parser, "split_indices", None) is not None:
                    # We're using predefined train/test split (Blender scenes)
                    # Example datasets: nerf, shiny, glossy, etc.
                    eval_stage = "test"
                elif cfg.test_every <= 0:
                    # We're using all images for training
                    # Example datasets: DTU, TnT (not tandt), etc.
                    eval_stage = "train"
                
                # Evaluate
                tqdm.write(f"--- Running evaluation ({eval_stage})...")
                self.eval(step, stage=eval_stage)

                # Render trajectory
                if not self.cfg.disable_video:
                    # Save all frames to video
                    video_dir = Path(cfg.result_dir) / "videos"
                    os.makedirs(video_dir, exist_ok=True)

                    tqdm.write("--- Rendering trajectory...")
                    video_file = video_dir / f"traj_{step + 1}.mp4"
                    self.render_traj(video_file)
                    tqdm.write(f"--- Video saved to: {video_file}")

            # Run compression if requested
            if cfg.compression is not None and step in [i + self.ckpt_step for i in cfg.eval_steps]:
                tqdm.write("--- Running compression...")
                self.run_compression(step)

            # Update training states for viewer
            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = num_train_rays_per_step * num_train_steps_per_sec
                # Update the viewer state
                self.viewer.render_tab_state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene
                self.viewer.update(step, num_train_rays_per_step)

        # Export PPISP per-view statistics post-training
        if cfg.post_processing == "ppisp":
            self.export_ppisp_reports()

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size
        val_loader = torch.utils.data.DataLoader(self.valset, batch_size=1, shuffle=False, num_workers=2)

        ellapsed_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(val_loader):
            c2w_mats = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None

            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            pixels = torch.clamp(pixels, 0.0, 1.0)     # [1, H, W, 3]
            height, width = pixels.shape[1:3]

            # Exposure metadata is available for any image with EXIF data (train or val)
            exposure = data["exposure"].to(device) if "exposure" in data else None

            torch.cuda.synchronize()
            tic = time.time()
            renders, alphas, _ = self.rasterize(
                camtoworlds=c2w_mats,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                masks=masks,
                frame_idcs=None,  # For novel views, pass None (no per-frame parameters available)
                camera_idcs=data["camera_idx"].to(device),
                exposure=exposure,
                render_mode=self.render_mode,
            )  # [1, H, W, ...], [1, H, W, 1]
            torch.cuda.synchronize()

            ellapsed_time += max(time.time() - tic, 1e-10)
            colors = torch.clamp(renders[..., :3], 0.0, 1.0)  # [1, H, W, 3]

            if world_rank == 0:
                # Compute NVS metrics
                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]

                if cfg.white_bkgd:
                    gt_alpha = data["alpha"].to(device) / 255.0  # [1, H, W, 1]
                    alphas_p = gt_alpha.permute(0, 3, 1, 2)      # [1, 1, H, W]
                    pixels_p = pixels_p * alphas_p + (1.0 - alphas_p)
                    colors_p = colors_p * alphas_p + (1.0 - alphas_p)

                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))

                # Compute color-corrected metrics for fair comparison across methods
                if cfg.use_color_correction_metric:
                    if cfg.color_correct_method == "affine":
                        cc_colors = color_correct_affine(colors, pixels)
                    else:
                        cc_colors = color_correct_quadratic(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
                    metrics["cc_ssim"].append(self.ssim(cc_colors_p, pixels_p))
                    metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))

                # Save renders to tensorboard, if requested
                if not cfg.tb_save_image:
                    continue

                stem = self.parser.image_names[self.valset.indices[data["image_id"]]].rsplit(".", 1)[0]
                alphas_p = alphas.permute(0, 3, 1, 2)  # [1, 1, H, W]

                self.writer.add_images(f"{stage}_{stem}/image", pixels_p, global_step=step + 1)
                self.writer.add_images(f"{stage}_{stem}/color", colors_p, global_step=step + 1)
                self.writer.add_images(f"{stage}_{stem}/alpha", alphas_p, global_step=step + 1)

                # Better visualization of rendered depths on tensorboard
                def rescale_depth(depth_map):
                    d = depth_map.view(-1)
                    near = torch.quantile(d, 0.02)
                    far  = torch.quantile(d, 0.98)
                    scaled_depths = (depth_map - near) / (far - near + 1e-6)
                    scaled_depths = torch.clamp(scaled_depths, 0.0, 1.0)
                    return scaled_depths
                
                # Save rendered depths if enabled depth rendering, likely we're doing depth supervision
                if "D" in self.render_mode:
                    depths = renders[..., 3:4]  # [1, H, W, 1] metric depth
                    depth_normals, _ = depths_to_camera_normals(depths, Ks)              # [1, H, W, 3]
                    dn_render = apply_opengl_normals(depth_normals).permute(0, 3, 1, 2)  # [1, 3, H, W]

                    depth_render = rescale_depth(depths).permute(0, 3, 1, 2)  # [1, 1, H, W], viz only
                    self.writer.add_images(f"{stage}_{stem}/depth", depth_render, global_step=step + 1)
                    self.writer.add_images(f"{stage}_{stem}/normal_from_depth", dn_render, global_step=step + 1)

                # Also save reference depths if depth image loss enabled
                if cfg.depth_image_lambda > 0.0:
                    depth_image = rescale_depth(data["depth_image"])  # [1, H, W, 1]
                    depth_image = depth_image.permute(0, 3, 1, 2)     # [1, 1, H, W]
                    self.writer.add_images(f"{stage}_{stem}/depth_reference", depth_image, global_step=step + 1)

                # All depth render modes except ZD enable normal rendering, save that too.
                if "D" in self.render_mode and not "ZD" in self.render_mode:
                    normals = renders[..., 5:8]
                    normal_render = apply_opengl_normals(normals).permute(0, 3, 1, 2)  # [1, 3, H, W]
                    self.writer.add_images(f"{stage}_{stem}/normal", normal_render, global_step=step + 1)
        
        if world_rank == 0:
            ellapsed_time /= len(val_loader)

            stats = { k: torch.stack(v).mean().item() for k, v in metrics.items() }
            stats.update({ "time_per_image": ellapsed_time })
            
            if cfg.use_color_correction_metric:
                tqdm.write(
                    f"--- PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f}, "
                    f"PSNR-CC: {stats['cc_psnr']:.3f}, SSIM-CC: {stats['cc_ssim']:.4f}, LPIPS-CC: {stats['cc_lpips']:.3f}, "
                    f"Time: {stats['time_per_image']:.3f}s/image "
                )
            else:
                tqdm.write(
                    f"--- PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f}, "
                    f"Time: {stats['time_per_image']:.3f}s/image "
                )
            # Save stats as json
            with open(f"{self.stats_dir}/{stage}_{step + 1}_metrics.json", "w") as f:
                json.dump(stats, f)

            # Save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)

            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, video_file: Path, depth_cutoff_factor=1.0):
        cfg = self.cfg
        device = self.device

        # Generate rendering trajectory based on traj_type
        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if getattr(self.parser, "split_indices", None) is not None:
            # For Blender scenes, it's best to render the trajectory from the test split
            camtoworlds_all = self.parser.camtoworlds[self.parser.split_indices["test"]]

        if cfg.traj_type == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, n_interp=cfg.traj_interp_factor
            )  # [N, 3, 4]
        elif cfg.traj_type == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height, n_frames=cfg.traj_num_frames
            )  # [N, 3, 4]
        elif cfg.traj_type == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
                n_frames=cfg.traj_num_frames,
            )  # [N, 3, 4]
        else:
            raise ValueError(f"-!- Unsupported trajectory type: {cfg.traj_type}")

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        def append_canvas(canvas, writer):
            # Pad the frame to standard resolutions
            FRAME_BLOCK_SIZE=16
            h, w, _ = canvas.shape
            pad_h = (FRAME_BLOCK_SIZE - h % FRAME_BLOCK_SIZE) % FRAME_BLOCK_SIZE
            pad_w = (FRAME_BLOCK_SIZE - w % FRAME_BLOCK_SIZE) % FRAME_BLOCK_SIZE

            padded_canvas = canvas
            if pad_h > 0 or pad_w > 0:
                padded_canvas = np.pad(canvas, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
            
            # Append frames
            writer.append_data(padded_canvas)

        # Render novel-views following trajectory and write to video
        writer = imageio.get_writer(str(video_file), fps=30)
        for i in range(len(camtoworlds_all)):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, alphas, _ = self.rasterize(
                camtoworlds=camtoworlds, Ks=Ks, width=width, height=height,
                sh_degree=cfg.sh_degree, render_mode=self.render_mode,
            )
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]

            if not "D" in self.render_mode:
                if cfg.mask_gt_image and cfg.alpha_reg > 0.0:
                    alpha_mask = alphas < 0.5  # [1, H, W, 1]
                    colors = torch.where(alpha_mask, 1.0, colors)
                
                # Canvas is simply the rendered novel-view
                canvas = colors.squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)

                append_canvas(canvas, writer)
                continue
            
            # We have depths and other stuff, bundle extra images into a 2x2 canvas
            cutoff = self.scene_scale * depth_cutoff_factor
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - cfg.near_plane) / (cutoff - cfg.near_plane)
            depths = torch.clamp(depths, 0.0, 1.0)

            colored_alphas = apply_float_colormap(alphas.squeeze(0), "gray").unsqueeze(0)   # [1, H, W, 3]
            colored_depths = apply_float_colormap(depths.squeeze(0), "magma").unsqueeze(0)  # [1, H, W, 3]

            camera_normals = torch.zeros_like(colored_depths)             # [1, H, W, 3]
            if not "ZD" in self.render_mode:
                camera_normals = apply_opengl_normals(renders[..., 5:8])  # [1, H, W, 3]

            # Gaussians opacities are forced to match GT mask,
            # setting the bg to white in the visualization
            if cfg.mask_gt_image and cfg.alpha_reg > 0.0:
                alpha_mask = alphas < 0.5  # [1, H, W, 1]
                colors = torch.where(alpha_mask, 1.0, colors)
                colored_depths = torch.where(alpha_mask, 1.0, colored_depths)
                camera_normals = torch.where(alpha_mask, 1.0, camera_normals)

            # Buid canvas
            canvas_1st_row = torch.cat([colors, colored_alphas], dim=2)
            canvas_2nd_row = torch.cat([colored_depths, camera_normals], dim=2)
            canvas = torch.cat([canvas_1st_row, canvas_2nd_row], dim=1).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            append_canvas(canvas, writer)

        # Save video
        writer.close()

    @torch.no_grad()
    def render_traj_with_mesh(self, mesh_file: Path, video_file: Path, depth_cutoff_factor=1.0):
        if sys.platform != "win32":
            os.environ["PYOPENGL_PLATFORM"] = "egl"
        import pyrender
        import trimesh
        assert mesh_file.exists(), mesh_file

        cfg = self.cfg
        device = self.device

        # Trajectory, same as render_traj
        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if getattr(self.parser, "split_indices", None) is not None:
            # For Blender scenes, it's best to render the trajectory from the test split
            camtoworlds_all = self.parser.camtoworlds[self.parser.split_indices["test"]]

        if cfg.traj_type == "interp":
            camtoworlds_all = generate_interpolated_path(camtoworlds_all, n_interp=cfg.traj_interp_factor)
        elif cfg.traj_type == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(camtoworlds_all, height=height, n_frames=cfg.traj_num_frames)
        elif cfg.traj_type == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
                n_frames=cfg.traj_num_frames,
            )
        else:
            assert_never(cfg.traj_type)

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]
        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # Build pyrender scene once, first the mesh file
        tm = trimesh.load(str(mesh_file), force="mesh")
        tm.visual = trimesh.visual.ColorVisuals()  # drops vertex colors, textures
        py_mesh = pyrender.Mesh.from_trimesh(
            tm,
            smooth=False, # flat shading — every face normal is used as-is
            material=pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.5, 0.5, 0.5, 1.0],  # neutral mid-gray
                metallicFactor=0.0,                    # fully diffuse, zero metallic
                roughnessFactor=1.0,                   # maximum roughness = lambertian
            ),
        )

        # Then camera
        K_np = K.cpu().numpy()  # [3, 3]
        py_camera = pyrender.IntrinsicsCamera(
            fx=K_np[0, 0], fy=K_np[1, 1],
            cx=K_np[0, 2], cy=K_np[1, 2],
            znear=cfg.near_plane, zfar=cfg.far_plane,
        )

        # Set up scene with camera, mesh, and lighting
        scene = pyrender.Scene(bg_color=[0., 0., 0., 0.], ambient_light=[0.15, 0.15, 0.15])
        scene.add(py_mesh)
        camera_node = scene.add(py_camera, pose=np.eye(4))
        headlight = pyrender.DirectionalLight(color=np.ones(3), intensity=1.5)
        scene.add(headlight, parent_node=camera_node)

        # Render offscreen
        renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
        FLIP_YZ = np.diag([1., -1., -1., 1.])

        def append_canvas(canvas, writer):
            # Pad the frame to standard resolutions
            FRAME_BLOCK_SIZE=16
            h, w, _ = canvas.shape
            pad_h = (FRAME_BLOCK_SIZE - h % FRAME_BLOCK_SIZE) % FRAME_BLOCK_SIZE
            pad_w = (FRAME_BLOCK_SIZE - w % FRAME_BLOCK_SIZE) % FRAME_BLOCK_SIZE

            padded_canvas = canvas
            if pad_h > 0 or pad_w > 0:
                padded_canvas = np.pad(canvas, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
            
            # Append frames
            writer.append_data(padded_canvas)

        # Render novel views with gsplat and mesh file with pyrender
        writer = imageio.get_writer(str(video_file), fps=30)
        for i in tqdm(range(len(camtoworlds_all)), desc="[>] Rendering trajectory", ncols=80):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            # Render Gaussians
            renders, alphas, _ = self.rasterize(
                camtoworlds=camtoworlds, Ks=Ks, width=width, height=height,
                sh_degree=cfg.sh_degree, render_mode=self.render_mode,
            )
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]

            # Render mesh
            c2w_np  = camtoworlds.squeeze(0).cpu().numpy()  # [4, 4]
            c2w_gl  = c2w_np @ FLIP_YZ                      # [4, 4]
            scene.set_pose(camera_node, pose=c2w_gl)
            mesh_color_u8, _ = renderer.render(scene)                   # [H, W, 3] uint8
            mesh_color_u8 = np.ascontiguousarray(mesh_color_u8.copy())  # make writable copy
            meshes = torch.from_numpy(mesh_color_u8).float() / 255.0    # [H, W, 3]
            meshes = meshes.unsqueeze(0).to(device)                     # [1, H, W, 3]

            # We only have RGB from splats, export 1x2 canvas with RGB (left) and mesh (right)
            if not "D" in self.render_mode:
                if cfg.mask_gt_image and cfg.alpha_reg > 0.0:
                    alpha_mask = alphas < 0.5  # [1, H, W, 1]
                    colors = torch.where(alpha_mask, 1.0, colors)
                    meshes = torch.where(alpha_mask, 1.0, meshes)

                canvas = torch.cat([colors, meshes], dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)

                append_canvas(canvas, writer)
                continue

            # We also have depths and other stuff, put them into canvas
            cutoff = self.scene_scale * depth_cutoff_factor
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - cfg.near_plane) / (cutoff - cfg.near_plane)
            depths = torch.clamp(depths, 0.0, 1.0)
            colored_depths = apply_float_colormap(depths.squeeze(0), "magma").unsqueeze(0)  # [1, H, W, 3]

            camera_normals = torch.zeros_like(colored_depths)             # [1, H, W, 3]
            if not "ZD" in self.render_mode:
                camera_normals = apply_opengl_normals(renders[..., 5:8])  # [1, H, W, 3]

            # Set BG to white at regions with alpha < 0.5
            if cfg.mask_gt_image and cfg.alpha_reg > 0.0:
                alpha_mask = alphas < 0.5  # [1, H, W, 1]
                colors = torch.where(alpha_mask, 1.0, colors)
                meshes = torch.where(alpha_mask, 1.0, meshes)
                colored_depths = torch.where(alpha_mask, 1.0, colored_depths)
                camera_normals = torch.where(alpha_mask, 1.0, camera_normals)

            # Buid canvas 2x2 canvas
            canvas_1st_row = torch.cat([colors, meshes], dim=2)
            canvas_2nd_row = torch.cat([colored_depths, camera_normals], dim=2)
            canvas = torch.cat([canvas_1st_row, canvas_2nd_row], dim=1).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            append_canvas(canvas, writer)
        
        # Clean up
        renderer.delete()
        writer.close()

    @torch.no_grad()
    def run_compression(self, step: int):
        world_rank = self.world_rank

        compress_dir = f"{self.cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def export_ppisp_reports(self) -> None:
        """Export PPISP visualization reports (PDF) and parameter JSON."""
        if self.cfg.post_processing != "ppisp":
            return
        print("[>] Exporting PPISP reports for all cameras...")

        # Compute frames per camera from training dataset
        num_cameras = self.parser.num_cameras
        frames_per_camera = [0] * num_cameras
        for idx in self.trainset.indices:
            cam_idx = self.parser.camera_indices[idx]
            frames_per_camera[cam_idx] += 1

        # Generate camera names from COLMAP camera IDs
        # camera_id_to_idx maps COLMAP ID -> 0-based index
        idx_to_camera_id = {v: k for k, v in self.parser.camera_id_to_idx.items()}
        camera_names = [f"camera_{idx_to_camera_id[i]}" for i in range(num_cameras)]

        # Export reports
        output_dir = Path(self.cfg.result_dir) / "ppisp"
        _ = export_ppisp_report(
            self.post_processing_module,
            frames_per_camera,
            output_dir,
            camera_names=camera_names,
        )
        print(f"[>] PPISP reports saved to: {output_dir}")

    @torch.no_grad()
    def extract_tsdf_mesh(
        self,
        max_depth: float = 10.0,
        voxel_size: float = 0.02,
        trunc_voxels: float = 8.0,
        depth_filter: bool = False,
        bounds=None,
        backend: str = "scalable",
    ):
        """Single-level TSDF fusion"""
        import open3d as o3d

        assert self.world_size == 1, "TSDF fusion cannot be run in distributed mode"
        assert "D" in self.render_mode, "TSDF fusion requires depth_render_mode, did you miss it during training?"

        device = self.device
        loader = torch.utils.data.DataLoader(self.trainset, batch_size=1, shuffle=False, num_workers=1)
        if getattr(self.parser, "split_indices", None) is not None:
            # For Blender scenes, extract mesh from the test split for better coverage
            loader = torch.utils.data.DataLoader(self.valset, batch_size=1, shuffle=False, num_workers=1)

        use_vbg = backend == "vbg"
        if use_vbg:
            # Tensor VoxelBlockGrid: extract_triangle_mesh() ignores voxels observed in fewer than
            # 3 views (weight_threshold=3.0), producing cleaner meshes for bounded objects. The grid
            # preallocates `block_count` blocks (`block_resolution^3` voxels each). Since surfaces are
            # approximately 2D, the number of occupied blocks scales roughly with
            # (scene_diameter / block_size)^2. We therefore estimate `block_count` from the scene
            # extent, enforce a minimum of 50k blocks, and add ~4× headroom for layered surfaces.
            # The extractor may crash on very large scenes, so this backend is only used for small ones.
            o3d_device = o3d.core.Device("CPU:0")
            block_resolution = 16
            block_side = block_resolution * voxel_size
            blocks_per_axis = (2.0 * self.scene_scale) / max(block_side, 1e-8)
            block_count = max(50_000, int(4 * blocks_per_axis ** 2))
            print(
                f"[>] VoxelBlockGrid: voxel={voxel_size:.4f} | block_res={block_resolution} | "
                f"block_count={block_count:,} (scene_scale={self.scene_scale:.3f})"
            )
            volume = o3d.t.geometry.VoxelBlockGrid(
                attr_names=("tsdf", "weight", "color"),
                attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.float32),
                attr_channels=((1), (1), (3)),
                voxel_size=voxel_size,
                block_resolution=block_resolution,
                block_count=block_count,
                device=o3d_device,
            )
        else:
            # Legacy ScalableTSDFVolume: grows on demand (no fixed capacity to overflow) and its
            # marching-cubes extractor is robust on large scenes. This is an alternative for
            # VoxelBlockGrid whose extractor usually crashes on large scenes (Open3D 0.18 and 0.19).
            # The trade-off is that ScalableTSDFVolume does not have weight pruning, so it
            # keeps more low-observation surface than the VoxelBlockGrid path.
            sdf_trunc = trunc_voxels * voxel_size
            print(
                f"[>] ScalableTSDFVolume: voxel={voxel_size:.4f} | sdf_trunc={sdf_trunc:.4f} "
                f"(scene_scale={self.scene_scale:.3f})"
            )
            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=voxel_size,
                sdf_trunc=sdf_trunc,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            )

        for data in tqdm(loader, desc="[1/2] TSDF fusion", ncols=80):
            c2w = data["camtoworld"].to(device)  # [1, 4, 4]
            K = data["K"].to(device)             # [1, 3, 3]
            masks = data["mask"].to(device) if "mask" in data else None
            image = data["image"].to(device) / 255.0  # [1, H, W, 3]
            alpha = data["alpha"].to(device) / 255.0  # [1, H, W, 1]
            height, width = image.shape[1:3]
            exposure = data["exposure"].to(device) if "exposure" in data else None

            renders, _, _ = self.rasterize(
                camtoworlds=c2w,
                Ks=K,
                width=width,
                height=height,
                sh_degree=self.cfg.sh_degree,
                masks=masks,
                frame_idcs=None,  # novel views: no per-frame params
                camera_idcs=data["camera_idx"].to(device),
                exposure=exposure,
                render_mode=self.render_mode,
            )  # [1, H, W, D]

            colors = torch.clamp(renders[..., :3], 0.0, 1.0)      # [1, H, W, 3]
            depths = renders[..., 3:4].clone()                    # [1, H, W, 1]
            depths[alpha < 0.5]        = 0.0
            depths[depths > max_depth] = 0.0

            if depth_filter:
                # Discard grazing-incidence pixels: where the rendered camera-space
                # normal is near-perpendicular to the viewing ray, the fused depth is
                # unreliable. Cull where the angle exceeds ~80 deg, using rendered normals.
                normals = renders[..., 5:8]  # [1, H, W, 3], camera-space
                fx, fy, cx, cy = K[0, 0, 0], K[0, 1, 1], K[0, 0, 2], K[0, 1, 2]
                vv, uu = torch.meshgrid(
                    torch.arange(height, device=device) + 0.5,
                    torch.arange(width, device=device) + 0.5,
                    indexing="ij",
                )  # [H, W] pixel-centre row/col
                ray = torch.stack([(uu - cx) / fx, (vv - cy) / fy, torch.ones_like(uu)], dim=-1)  # [H, W, 3]
                ray = ray / ray.norm(dim=-1, keepdim=True)
                n = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)   # [1, H, W, 3]
                cos = (n * ray).sum(dim=-1, keepdim=True).abs()            # [1, H, W, 1]
                depths[cos < math.cos(math.radians(80.0))] = 0.0

            if bounds is not None:
                # bounds is [3, 2]; each row is the valid [min, max] along a world axis
                points = depth_to_points(depths, c2w, K, z_depth=True)  # [1, H, W, 3]
                erase = (points[..., 0] < bounds[0, 0]) | (points[..., 0] > bounds[0, 1]) |\
                        (points[..., 1] < bounds[1, 0]) | (points[..., 1] > bounds[1, 1]) |\
                        (points[..., 2] < bounds[2, 0]) | (points[..., 2] > bounds[2, 1])
                depths[erase] = 0.0

            # Fuse this view with float depth (no uint16 mm quantization)
            depth_np = np.ascontiguousarray(depths[0, :, :, 0].cpu().numpy().astype(np.float32))     # [H, W]
            w2c_np = np.linalg.inv(c2w[0].cpu().numpy()).astype(np.float64)                          # [4, 4]

            if use_vbg:
                color_np = np.ascontiguousarray(colors[0].cpu().numpy().astype(np.float32))          # [H, W, 3], 0-1
                K_np = K[0].cpu().numpy().astype(np.float64)                                         # [3, 3]
                depth_img = o3d.t.geometry.Image(depth_np).to(o3d_device)
                color_img = o3d.t.geometry.Image(color_np).to(o3d_device)
                intrinsic = o3d.core.Tensor(K_np)
                extrinsic = o3d.core.Tensor(w2c_np)
                frustum = volume.compute_unique_block_coordinates(depth_img, intrinsic, extrinsic, 1.0, max_depth)
                volume.integrate(frustum, depth_img, color_img, intrinsic, extrinsic, 1.0, max_depth, trunc_voxels)
            else:
                color_np = np.ascontiguousarray((colors[0] * 255).cpu().numpy().astype(np.uint8))    # [H, W, 3]
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(color_np),
                    o3d.geometry.Image(depth_np),
                    depth_scale=1.0,
                    depth_trunc=max_depth,
                    convert_rgb_to_intensity=False,
                )
                intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    int(width), int(height),
                    K[0, 0, 0].item(), K[0, 1, 1].item(), K[0, 0, 2].item(), K[0, 1, 2].item(),
                )
                volume.integrate(rgbd, intrinsic, w2c_np)

        print("[2/2] Extracting mesh from TSDF volume...")
        mesh = volume.extract_triangle_mesh()
        if use_vbg:
            mesh = mesh.to_legacy()
        mesh.compute_vertex_normals()
        print(f"[>] Done extraction, num vertices: {len(mesh.vertices):,}")

        return mesh
    
    @torch.no_grad()
    def _viewer_render_fn(self, camera_state: CameraState, render_tab_state: RenderTabState):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height

        c2w = camera_state.c2w
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = camera_state.get_K((width, height))
        K = torch.from_numpy(K).float().to(self.device)

        RENDER_MODE_MAP = {
            # Don't use render_mode for rgb/alpha because geometry rendering is expensive
            "rgb": "RGB", "alpha": "RGB",
            # No D in render mode means geometry is not the main concern, pick the cheapest option
            "depth": "ZD" if not "D" in self.render_mode else self.render_mode,
            "normal": "RGB+PD",
        }

        render_colors, render_alphas, info = self.rasterize(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device) / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
        )  # [1, H, W, D]
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        elif render_tab_state.render_mode == "depth":
            # normalize depth to [0, 1]
            depth_ch = 0 if not "D" in self.render_mode else 3
            depth = render_colors[0, ..., depth_ch:depth_ch+1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "normal":
            normals = render_colors[..., 5:8]
            cam_normals = apply_opengl_normals(normals)  # [1, H, W, 3]
            renders = cam_normals.squeeze(0).cpu().numpy()
        return renders
