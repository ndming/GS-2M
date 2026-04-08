import json
import math
import os
import time

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union
from typing_extensions import Literal, assert_never

import viser
import yaml
import imageio

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
from gsplat.strategy import DefaultStrategy, MCMCStrategy

from viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap
from fused_ssim import fused_ssim

from .datasets import Dataset, get_parser
from .datasets.traj import generate_ellipse_path_z, generate_interpolated_path, generate_spiral_path
from .utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed, fix_normal_coordinates


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"
    # If render_traj_path is interp, the output traj will have traj_num_interps * (n_poses - 1)
    traj_num_interps: int = 1

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 1
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image, 0 (default) will use all images for training
    test_every: int = 0
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"
    # Load EXIF exposure metadata from images (if available)
    load_exposure: bool = True
    # Mask GT RGB image during training for object-centric reconstruction
    mask_gt_image: bool = False
    # If set, don't perfrom pre-processing of GT RGB images and use the existing ones
    reuse_processed_images: bool = False

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 20_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 20_000, 30_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = False
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 20_000, 30_000])
    # Whether to disable video generation during training and evaluation
    disable_video: bool = False

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 100.0

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False
    # Use white background in visualization and evaluation
    white_bkgd: bool = False

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

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Post-processing method for appearance correction (experimental)
    post_processing: Optional[Literal["bilateral_grid", "ppisp"]] = None
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)
    # Enable PPISP controller
    ppisp_use_controller: bool = True
    # Use controller distillation in PPISP (only applies when post_processing="ppisp" and ppisp_use_controller=True)
    ppisp_controller_distillation: bool = True
    # Controller activation ratio for PPISP (only applies when post_processing="ppisp" and ppisp_use_controller=True)
    ppisp_controller_activation_num_steps: int = 25_000
    # Color correction method for cc_* metrics (only applies when post_processing is set)
    color_correct_method: Literal["affine", "quadratic"] = "affine"
    # Compute color-corrected metrics (cc_psnr, cc_ssim, cc_lpips) during evaluation
    use_color_correction_metric: bool = False

    # Enable depth loss between rendered depths and depths from sparse SfM points (experimental)
    depth_point_loss: bool = False
    # Enable depth loss between rendered depths and GT depth images (experimental)
    depth_image_loss: bool = False
    # Start applying depth image loss from this step (only applies when depth_image_loss=True)
    depth_image_loss_from: int = 7000
    # The distance beyond which GT depth values are considered invalid
    depth_image_max_distance: float = 10.0
    # Weight for depth loss
    depth_lambda: float = 1e-2
    # Let the pipeline know if poses, sparse points, and depth GT already in metric scale
    metric_scale: bool = False

    # Mutli-view observation trimming
    multi_view_observe_trim: bool = False
    # Enforce disk-like Gaussians (planar loss), 0 to disable
    planar_reg: float = 0.0
    # Enforce depth normal consistency, 0 to disable
    depth_normal_lambda: float = 0.0
    # Start applying depth normal consistency loss from this step (only applies when depth_normal_lambda > 0)
    depth_normal_loss_from: int = 7000

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = True

    lpips_net: Literal["vgg", "alex"] = "alex"

    # 3DGUT (uncented transform + eval 3D)
    with_ut: bool = False
    with_eval3d: bool = False

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
            if strategy.noise_injection_stop_iter >= 0:
                strategy.noise_injection_stop_iter = int(
                    strategy.noise_injection_stop_iter * factor
                )
        else:
            assert_never(strategy)


def create_splats_with_optimizers(
    parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
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
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

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


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
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

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")
        for file in Path(f"{cfg.result_dir}/tb").glob("events.out.tfevents.*"):
            file.unlink()

        # Load data: Training data should contain initial points and colors.
        self.parser = get_parser(cfg.data_dir)(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
            load_exposure=cfg.load_exposure,
            mask_gt_image=cfg.mask_gt_image,
            reuse_processed_images=cfg.reuse_processed_images,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_point_depth=cfg.depth_point_loss,
            load_image_depth=cfg.depth_image_loss,
        )
        self.valset = Dataset(self.parser, split="val", load_image_depth=cfg.depth_image_loss)
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("[>] Scene half extent:", self.scene_scale)

        if self.parser.num_cameras > 1 and cfg.batch_size != 1:
            raise ValueError(
                f"When using multiple cameras ({self.parser.num_cameras} found), batch_size must be 1, "
                f"but got batch_size={cfg.batch_size}."
            )
        if cfg.post_processing == "ppisp" and cfg.batch_size != 1:
            raise ValueError(
                f"PPISP post-processing requires batch_size=1, got batch_size={cfg.batch_size}"
            )
        if cfg.post_processing is not None and world_size > 1:
            raise ValueError(
                f"Post-processing ({cfg.post_processing}) requires single-GPU training, "
                f"but world_size={world_size}."
            )
        if cfg.post_processing == "ppisp" and isinstance(cfg.strategy, DefaultStrategy):
            raise ValueError(
                f"PPISP post-processing requires MCMCStrategy at the moment."
            )

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
        )
        print("[>] Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
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

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
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

        self.post_processing_module = None
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
                controller_activation_ratio=cfg.ppisp_controller_activation_num_steps / cfg.max_steps,
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

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

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

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

        # Track if Gaussians are frozen (for controller distillation)
        self._gaussians_frozen = False

        # Render mode
        with_plane_depth = cfg.depth_point_loss or cfg.depth_image_loss or cfg.depth_normal_lambda > 0.0
        self.render_mode = "RGB+PD" if with_plane_depth else "RGB"

    def freeze_gaussians(self):
        """Freeze all Gaussian parameters for controller distillation.

        This prevents Gaussians from being updated by any loss (including regularization)
        while the controller learns to predict per-frame corrections.
        """
        if self._gaussians_frozen:
            return

        for name, param in self.splats.items():
            param.requires_grad = False

        self._gaussians_frozen = True
        tqdm.write("[>] Distillation: Gaussian parameters frozen")

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
        frame_idcs: Optional[Tensor] = None,
        camera_idcs: Optional[Tensor] = None,
        exposure: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = self.cfg.camera_model
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            with_ut=self.cfg.with_ut,
            with_eval3d=self.cfg.with_eval3d,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0

        if self.cfg.post_processing is not None:
            # Create pixel coordinates [H, W, 2] with +0.5 center offset
            pixel_y, pixel_x = torch.meshgrid(
                torch.arange(height, device=self.device) + 0.5,
                torch.arange(width, device=self.device) + 0.5,
                indexing="ij",
            )
            pixel_coords = torch.stack([pixel_x, pixel_y], dim=-1)  # [H, W, 2]

            if "RGB" in kwargs["render_mode"]:
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

                render_colors = (
                    torch.cat([rgb, extra], dim=-1) if extra is not None else rgb
                )

        return render_colors, render_alphas, info

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
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

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm(range(init_step, max_steps), ncols=128, desc="[>] Training")
        for step in pbar:
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
                and step >= cfg.ppisp_controller_activation_num_steps
            ):
                self.freeze_gaussians()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = ( # B * H * W
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
            exposure = (
                data["exposure"].to(device) if "exposure" in data else None
            )  # [B,]

            if cfg.depth_point_loss:
                depth_pixels = data["depth_pixels"].to(device) # [1, M, 2]
                depth_values = data["depth_values"].to(device) # [1, M]
            if cfg.depth_image_loss:
                depth_image = data["depth_image"].to(device) # [1, H, W, 1]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode=self.render_mode,
                masks=masks,
                frame_idcs=image_ids,
                camera_idcs=data["camera_idx"].to(device),
                exposure=exposure,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

            # Supervise sampled depths from rendered depths with prior depth points
            if cfg.depth_point_loss:
                # Prepare depth pixels for grid sampling into rendered depth map
                depth_pixels = torch.stack(
                    [
                        depth_pixels[:, :, 0] / (width - 1) * 2 - 1,
                        depth_pixels[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                ) # normalize to [-1, 1]
                grid = depth_pixels.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(depths.permute(0, 3, 1, 2), grid, align_corners=True)  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depth_values  # [1, M]

                scale = 1.0 if cfg.metric_scale else self.scene_scale
                depth_loss = F.l1_loss(disp, disp_gt) * scale
                loss += depth_loss * cfg.depth_lambda

            # Supervise rendered depths with prior depth images
            if cfg.depth_image_loss and step >= cfg.depth_image_loss_from:
                depth_valid_mask = (depth_image > 0) & (depth_image < cfg.depth_image_max_distance) & (depths > 0)
                if depth_valid_mask.any():
                    scale = 1.0 if cfg.metric_scale else self.scene_scale
                    depth_loss = F.l1_loss(depths[depth_valid_mask], depth_image[depth_valid_mask]) * scale
                    loss += depth_loss * cfg.depth_lambda

            # Depth-normal consistency
            if cfg.depth_normal_lambda > 0.0 and "P" in self.render_mode and step >= cfg.depth_normal_loss_from:
                def _get_img_grad_weight(gt_image):
                    # gt_image: [..., H, W, 3]
                    *batch_dims, H, W, C = gt_image.shape
                    assert C == 3

                    # Move channels to NCHW
                    gt_img = gt_image.permute(*range(len(batch_dims)), -1, -3, -2)  # [..., 3, H, W]

                    # Gradients
                    bottom = gt_img[..., :, 2:H,   1:W-1]
                    top    = gt_img[..., :, 0:H-2, 1:W-1]
                    right  = gt_img[..., :, 1:H-1, 2:W]
                    left   = gt_img[..., :, 1:H-1, 0:W-2]

                    grad_x = torch.mean(torch.abs(right - left), dim=-3, keepdim=True)  # avg over channel
                    grad_y = torch.mean(torch.abs(top - bottom), dim=-3, keepdim=True)

                    grad = torch.cat((grad_x, grad_y), dim=-3)  # [..., 2, H-2, W-2]
                    grad, _ = torch.max(grad, dim=-3)           # [..., H-2, W-2]

                    grad_flat = grad.view(*batch_dims, -1)
                    gmin = grad_flat.min(dim=-1, keepdim=True).values
                    gmax = grad_flat.max(dim=-1, keepdim=True).values
                    grad = (grad - gmin[..., None]) / (gmax[..., None] - gmin[..., None] + 1e-6)

                    # Pad back to H, W and add channel dim
                    grad = torch.nn.functional.pad(grad, (1, 1, 1, 1))  # [..., H, W]
                    grad = grad.unsqueeze(-1)  # [..., H, W, 1]
                
                    return grad

                weights = (1.0 - _get_img_grad_weight(pixels)).clamp(0, 1).detach() ** 2  # [..., H, W, 1]
                weights = weights.unsqueeze(-4)            # [..., 1, H, W, 1]
                render_normals = info["render_normals_c"]  # [..., C, H, W, 3]
                depth_normals  = info["depth_normals"]     # [..., C, H, W, 3]

                # Valid mask (non-zero normals)
                render_valid = render_normals.norm(dim=-1, keepdim=True) > 1e-6
                depth_valid  = depth_normals.norm(dim=-1, keepdim=True) > 1e-6
                valid_mask = (render_valid & depth_valid).float()
                weights = weights * valid_mask

                diff = (render_normals - depth_normals).abs().sum(dim=-1, keepdim=True)  # [..., C, H, W, 1]
                loss += cfg.depth_normal_lambda * ((weights * diff).sum() / (weights.sum() + 1e-6))
            
            if cfg.post_processing == "bilateral_grid":
                post_processing_reg_loss = 10 * total_variation_loss(
                    self.post_processing_module.grids
                )
                loss += post_processing_reg_loss
            elif cfg.post_processing == "ppisp":
                post_processing_reg_loss = (
                    self.post_processing_module.get_regularization_loss()
                )
                loss += post_processing_reg_loss

            # regularizations
            if cfg.opacity_reg > 0.0:
                loss += cfg.opacity_reg * torch.sigmoid(self.splats["opacities"]).mean()
            if cfg.scale_reg > 0.0:
                loss += cfg.scale_reg * torch.exp(self.splats["scales"]).mean()
            if cfg.planar_reg > 0.0:
                radii = info["radii"]                    # [..., C, N, 2]
                valid_per_cam = (radii > 0).any(dim=-1)  # [..., C, N]
                visibility_filter = valid_per_cam.any(dim=-2)  # [..., N]
                if visibility_filter.sum() > 0:
                    scales = torch.exp(self.splats["scales"])  # [N, 3]
                    scales = scales[visibility_filter]

                    sorted_scale, _ = torch.sort(scales, dim=-1)
                    min_scale_loss = sorted_scale[..., 0]
                    loss += cfg.planar_reg * min_scale_loss.mean()

            loss.backward()

            # Update progress bar with postfix loss
            if world_rank == 0 and step % 100 == 0:
                postfix_dict = { "SH": f"{sh_degree_to_use}", "Loss": f"{loss.item():.5f}" }
                n_points = len(self.splats["means"])
                postfix_dict["Points"] = f"{n_points}"
                if cfg.depth_point_loss or (cfg.depth_image_loss and step >= cfg.depth_image_loss_from):
                    postfix_dict["Depth"] = f"{depth_loss.item():.5f}"
                if cfg.pose_opt and cfg.pose_noise:
                    pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                    postfix_dict["Pose"] = f"{pose_err.item():.5f}"
                pbar.set_postfix(postfix_dict)

            # Update tensorboard scalar values
            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_point_loss or (cfg.depth_image_loss and step >= cfg.depth_image_loss_from):
                    self.writer.add_scalar("train/depthloss", depth_loss.item(), step)
                if cfg.post_processing is not None:
                    self.writer.add_scalar(
                        "train/post_processing_reg_loss",
                        post_processing_reg_loss.item(),
                        step,
                    )
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellapsed_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                elapsed_minutes = stats["ellapsed_time"] / 60
                tqdm.write(f"[>] Iter {(step + 1):>5} | Memory: {mem:.2f}GB | Ellapsed time: {elapsed_minutes:.2f}mins")
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                if self.post_processing_module is not None:
                    data["post_processing"] = self.post_processing_module.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:

                if self.cfg.app_opt:
                    # eval at origin to bake the appeareance into the colors
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

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)

            # optimize
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
            for optimizer in self.post_processing_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Run post-backward steps after backward and optimizer
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step, stage="train" if cfg.test_every <= 0 else "val")
                self.render_traj(step)

            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        tqdm.write("--- Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellapsed_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            # Exposure metadata is available for any image with EXIF data (train or val)
            exposure = data["exposure"].to(device) if "exposure" in data else None

            torch.cuda.synchronize()
            tic = time.time()
            renders, alphas, meta = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
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

                # Save renders to tensorboard
                if not cfg.tb_save_image:
                    continue
                stem = self.parser.image_names[self.valset.indices[data["image_id"]]].rsplit(".", 1)[0]
                gt_image = torch.clamp(pixels, 0.0, 1.0).permute(0, 3, 1, 2) # [1, 3, H, W]
                color_render = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                alpha_render = alphas.permute(0, 3, 1, 2)  # [1, 1, H, W]

                self.writer.add_images(f"{stage}_{stem}/gt_rgb", gt_image, global_step=step + 1)
                self.writer.add_images(f"{stage}_{stem}/color", color_render, global_step=step + 1)
                self.writer.add_images(f"{stage}_{stem}/alpha", alpha_render, global_step=step + 1)

                def rescale_depth(depth_map):
                    d = depth_map.view(-1)
                    near = torch.quantile(d, 0.02)
                    far  = torch.quantile(d, 0.98)
                    scaled_depths = (depth_map - near) / (far - near + 1e-6)
                    scaled_depths = torch.clamp(scaled_depths, 0.0, 1.0)
                    return scaled_depths

                if "D" in self.render_mode:
                    depths = renders[..., 3:4]
                    depths = rescale_depth(depths)             # [1, H, W, 1]
                    depth_render = depths.permute(0, 3, 1, 2)  # [1, 1, H, W]
                    self.writer.add_images(f"{stage}_{stem}/depth", depth_render, global_step=step + 1)

                if cfg.depth_image_loss:
                    depth_image = rescale_depth(data["depth_image"])  # [1, H, W, 1]
                    depth_image = depth_image.permute(0, 3, 1, 2)     # [1, 1, H, W]
                    self.writer.add_images(f"{stage}_{stem}/gt_depth", depth_image, global_step=step + 1)

                if "P" in self.render_mode:
                    render_normals = meta["render_normals_c"]  # [1, H, W, 3]
                    depth_normals  = meta["depth_normals"]  # [1, H, W, 3]
                    
                    normal_render = fix_normal_coordinates(render_normals).permute(0, 3, 1, 2)  # [1, 3, H, W]
                    depth_normal  = fix_normal_coordinates(depth_normals).permute(0, 3, 1, 2)   # [1, 3, H, W]

                    self.writer.add_images(f"{stage}_{stem}/normal_render", normal_render, global_step=step + 1)
                    self.writer.add_images(f"{stage}_{stem}/normal_depth", depth_normal, global_step=step + 1)

        if world_rank == 0:
            ellapsed_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "time_per_image": ellapsed_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            if cfg.use_color_correction_metric:
                tqdm.write(
                    f"--- PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f}, "
                    f"CC_PSNR: {stats['cc_psnr']:.3f}, CC_SSIM: {stats['cc_ssim']:.4f}, CC_LPIPS: {stats['cc_lpips']:.3f}, "
                    f"Time: {stats['time_per_image']:.3f}s/image "
                    # f"Number of GS: {stats['num_GS']}"
                )
            else:
                tqdm.write(
                    f"--- PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f}, "
                    f"Time: {stats['time_per_image']:.3f}s/image "
                    # f"Number of GS: {stats['num_GS']}"
                )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        if self.cfg.disable_video:
            return
        tqdm.write("--- Rendering trajectory...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, n_interp=cfg.traj_num_interps
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"-!- Render trajectory type not supported: {cfg.render_traj_path}"
            )

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

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step + 1}.mp4", fps=30)
        for i in range(len(camtoworlds_all)): # desc="Rendering trajectory"
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, alphas, meta = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+PD",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            
            cutoff = self.scene_scale * 1.5
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - cfg.near_plane) / (cutoff - cfg.near_plane)
            depths = torch.clamp(depths, 0.0, 1.0)

            colored_alphas = apply_float_colormap(alphas.squeeze(0), "gray").unsqueeze(0)   # [1, H, W, 3]
            colored_depths = apply_float_colormap(depths.squeeze(0), "magma").unsqueeze(0)  # [1, H, W, 3]
            adated_normals = fix_normal_coordinates(meta["depth_normals"])                  # [1, H, W, 3]

            # Buid canvas
            canvas_1st_row = torch.cat([colors, colored_alphas], dim=2)
            canvas_2nd_row = torch.cat([colored_depths, adated_normals], dim=2)
            canvas = torch.cat([canvas_1st_row, canvas_2nd_row], dim=1).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)

            # pad the frame to standard resolutions
            FRAME_BLOCK_SIZE=16
            h, w, _ = canvas.shape
            pad_h = (FRAME_BLOCK_SIZE - h % FRAME_BLOCK_SIZE) % FRAME_BLOCK_SIZE
            pad_w = (FRAME_BLOCK_SIZE - w % FRAME_BLOCK_SIZE) % FRAME_BLOCK_SIZE
            if pad_h > 0 or pad_w > 0:
                padding_value = 255 if cfg.white_bkgd else 0
                canvas = np.pad(canvas, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=padding_value)

            writer.append_data(canvas)
        writer.close()
        tqdm.write(f"--- Video saved to {video_dir}/traj_{step + 1}.mp4")

    @torch.no_grad()
    def export_ppisp_reports(self) -> None:
        """Export PPISP visualization reports (PDF) and parameter JSON."""
        if self.cfg.post_processing != "ppisp":
            return
        print("Exporting PPISP reports for all cameras...")

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
        output_dir = Path(self.cfg.result_dir) / "ppisp_reports"
        pdf_paths = export_ppisp_report(
            self.post_processing_module,
            frames_per_camera,
            output_dir,
            camera_names=camera_names,
        )
        print(f"PPISP reports saved to {output_dir}")
        for path in pdf_paths:
            print(f" - {path.name}")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
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
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
        )  # [1, H, W, 3]
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
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
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    # Import post-processing modules based on configuration
    # These imports must be here (not in __main__) for distributed workers
    if cfg.post_processing == "bilateral_grid":
        global BilateralGrid, slice, total_variation_loss
        from fused_bilagrid import BilateralGrid, slice, total_variation_loss
    elif cfg.post_processing == "ppisp":
        global PPISP, PPISPConfig, export_ppisp_report
        from ppisp import PPISP, PPISPConfig
        from ppisp.report import export_ppisp_report

    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        if runner.post_processing_module is not None:
            pp_state = ckpts[0].get("post_processing")
            if pp_state is not None:
                runner.post_processing_module.load_state_dict(pp_state)
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()
        runner.export_ppisp_reports()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        # time.sleep(1000000)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down viewer...")
            runner.server.stop()
