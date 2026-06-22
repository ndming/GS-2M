import random
import copy

import numpy as np
import open3d as o3d
import torch
import pygltflib
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import colormaps
from patch_ncc import warp_patch_ncc

class CameraOptModule(torch.nn.Module):
    """Camera pose optimization module."""

    def __init__(self, n: int):
        super().__init__()
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(n, 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """Adjust camera pose based on deltas.

        Args:
            camtoworlds: (..., 4, 4)
            embed_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        assert camtoworlds.shape[:-2] == embed_ids.shape
        batch_dims = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_dims, -1)
        )  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_dims, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)


class AppearanceOptModule(torch.nn.Module):
    """Appearance optimization module."""

    def __init__(
        self,
        n: int,
        feature_dim: int,
        embed_dim: int = 16,
        sh_degree: int = 3,
        mlp_width: int = 64,
        mlp_depth: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sh_degree = sh_degree
        self.embeds = torch.nn.Embedding(n, embed_dim)
        layers = []
        layers.append(
            torch.nn.Linear(embed_dim + feature_dim + (sh_degree + 1) ** 2, mlp_width)
        )
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(mlp_width, 3))
        self.color_head = torch.nn.Sequential(*layers)

    def forward(
        self, features: Tensor, embed_ids: Tensor, dirs: Tensor, sh_degree: int
    ) -> Tensor:
        """Adjust appearance based on embeddings.

        Args:
            features: (N, feature_dim)
            embed_ids: (C,)
            dirs: (C, N, 3)

        Returns:
            colors: (C, N, 3)
        """
        from gsplat.cuda._torch_impl import _eval_sh_bases_fast

        C, N = dirs.shape[:2]
        # Camera embeddings
        if embed_ids is None:
            embeds = torch.zeros(C, self.embed_dim, device=features.device)
        else:
            embeds = self.embeds(embed_ids)  # [C, D2]
        embeds = embeds[:, None, :].expand(-1, N, -1)  # [C, N, D2]
        # GS features
        features = features[None, :, :].expand(C, -1, -1)  # [C, N, D1]
        # View directions
        dirs = F.normalize(dirs, dim=-1)  # [C, N, 3]
        num_bases_to_use = (sh_degree + 1) ** 2
        num_bases = (self.sh_degree + 1) ** 2
        sh_bases = torch.zeros(C, N, num_bases, device=features.device)  # [C, N, K]
        sh_bases[:, :, :num_bases_to_use] = _eval_sh_bases_fast(num_bases_to_use, dirs)
        # Get colors
        if self.embed_dim > 0:
            h = torch.cat([embeds, features, sh_bases], dim=-1)  # [C, N, D1 + D2 + K]
        else:
            h = torch.cat([features, sh_bases], dim=-1)
        colors = self.color_head(h)
        return colors


# Adapted from: Geometry-Grounded Gaussian Splatting
# https://github.com/HKUST-SAIL/Geometry-Grounded-Gaussian-Splatting
class PatchMatch:
    """Multi-view consistency module"""

    def __init__(
        self,
        pixel_noise_threshold,
        occlusion_threshold,
        max_num_samples,
        angle_factor,
        angle_noise_threshold,
        geo_weights_decay_rate,
        ncc_weights_decay_rate,
        optimize_geo=True,
        optimize_ncc=True,
        device="cuda",
    ):
        self.pixel_noise_th = pixel_noise_threshold
        self.occlusion_th = occlusion_threshold
        self.max_num_samples = max_num_samples
        self.angle_factor = angle_factor   # scale the per-point angular error
        self.angle_noise_th = angle_noise_threshold * np.pi / 180.0  # radians
        self.geo_weights_decay_rate = geo_weights_decay_rate
        self.ncc_weights_decay_rate = ncc_weights_decay_rate
        self.optimize_geo = optimize_geo
        self.optimize_ncc = optimize_ncc
        self.device = device
        self._grid_cache = {}

    def _grids(self, h, w):
        """Cache the per-resolution pixel grid and flat index range.

        These depend only on (H, W), which is constant across the dataset, so
        rebuilding them on every call wastes several kernel launches.
        """
        cached = self._grid_cache.get((h, w))
        if cached is None:
            ix_ref, iy_ref = torch.meshgrid(
                torch.arange(w, device=self.device),
                torch.arange(h, device=self.device),
                indexing="xy",
            )
            pixel_grid_flat = torch.stack([ix_ref, iy_ref], dim=-1).float().reshape(-1, 2)  # [H * W, 2]
            flat_indices    = torch.arange(h * w, device=self.device)                       # [H * W]
            cached = (pixel_grid_flat, flat_indices)
            self._grid_cache[(h, w)] = cached
        return cached

    def __call__(
        self,
        data_ref,
        data_nea,
        depth_ref,   # [1, H, W, 1]
        depth_nea,   # [1, H, W, 1]
        normal_ref,  # [1, H, W, 3]
        normal_nea,  # [1, H, W, 3]
    ):
        cam_ref_params = cam_params_from_data_batch(data_ref)
        cam_nea_params = cam_params_from_data_batch(data_nea)

        # c2w transforms of both views
        c2w_ref = data_ref["camtoworld"].to(self.device).squeeze()  # [4, 4]
        c2w_nea = data_nea["camtoworld"].to(self.device).squeeze()  # [4, 4]
        R_ref = c2w_ref[:3, :3]  # [3, 3]
        R_nea = c2w_nea[:3, :3]  # [3, 3]
        t_ref = c2w_ref[:3,  3]  # [3,]
        t_nea = c2w_nea[:3,  3]  # [3,]

        with torch.no_grad():
            w_ref, h_ref = cam_ref_params["W"], cam_ref_params["H"]
            fx_ref, fy_ref = cam_ref_params["fx"], cam_ref_params["fy"]
            cx_ref, cy_ref = cam_ref_params["cx"], cam_ref_params["cy"]
            ix = (torch.arange(w_ref, device=self.device, dtype=torch.float32) - cx_ref) / fx_ref
            iy = (torch.arange(h_ref, device=self.device, dtype=torch.float32) - cy_ref) / fy_ref
            R_nea_to_ref = R_nea.T @ R_ref
            t_nea_to_ref = (t_nea - t_ref) @ R_ref
            # Ref -> nea transform (constant): reused for both the geo
            # forward-projection below and the NCC patch warping
            R_ref_to_nea = R_ref.T @ R_nea
            t_ref_to_nea = (t_ref - t_nea) @ R_nea

            # Flat pixel grid + index range, cached per resolution
            pixel_grid_flat, flat_indices = self._grids(h_ref, w_ref)

        # Whether to also enforce multi-view normal consistency (part of Lgeo)
        compute_angle = self.optimize_geo and self.angle_factor > 0.0 and normal_nea is not None

        # Lgeo
        with torch.set_grad_enabled(self.optimize_geo):
            # Back-project pixels to camera points using depth map of reference view
            depth_reshape = depth_ref.squeeze().unsqueeze(-1) # [H, W, 1]
            pts_cam_ref = torch.cat(
                [
                    depth_reshape * ix[None, :, None],
                    depth_reshape * iy[:, None, None],
                    depth_reshape,
                ],
                dim=-1,
            ) # [H, W, 3]

            # Move cam points in reference view to nearest view (single affine)
            pts_cam_nea = pts_cam_ref @ R_ref_to_nea + t_ref_to_nea  # [H, W, 3]
            K_nea = data_nea["K"].to(self.device)                    # [3, 3]
            pts_proj_nea = pts_cam_nea @ K_nea.T      # [H, W, 3]
            w_nea, h_nea = cam_nea_params["W"], cam_nea_params["H"]
            pixels_nea = pts_proj_nea[..., :2] / pts_proj_nea[..., 2:3]  # [H, W, 2]
            valid_proj = (
                # Image coordinates must be non-negative, ...
                (pixels_nea[..., 0] >= 0) & (pixels_nea[..., 1] >= 0) &
                # ... within the image bounds, ...
                (pixels_nea[..., 0] < w_nea) & (pixels_nea[..., 1] < h_nea) &
                # ... land on pixels whose depth in front of the image plane, ...
                (pts_cam_nea[..., 2] > 0.2) &
                # ... and come from valid reference depths
                (pts_cam_ref[..., 2] > 0.2) & (depth_reshape[..., 0] > 1e-6)
            ) # [H, W]

            # Flatten valid_proj mask into 1D indices,
            # track which flat pixels survived projection
            valid_proj_indices = flat_indices[valid_proj.reshape(-1)]  # [M,], where M = valid_proj.sum()
            sample_pixels = pixels_nea[valid_proj]                     # [M, 2]
            sample_depths = sample_map(sample_pixels, depth_nea)       # [M, 1]
            if compute_angle:
                # Neighbor (cam-space) normals at the same projected pixels
                sample_normals = sample_map(sample_pixels, normal_nea)  # [M, 3]

            # Discard samples failing occlussion check and invalid depths
            queried_depths = pts_cam_nea[valid_proj][:, 2] # [M,]
            sampled_depths = sample_depths[:, 0]           # [M,]
            valid_occ_proj = (
                (sampled_depths > 1e-6) &
                (queried_depths - sampled_depths <= self.occlusion_th)
            ) # [M,]

            # Subindex into valid_proj_indices
            survived_indices = valid_proj_indices[valid_occ_proj]  # [N]
            pixels_nea_valid = sample_pixels[valid_occ_proj]       # [N, 2]
            depths_nea_valid = sample_depths[valid_occ_proj]       # [N, 1]

            # Back-project to camera points of nearest view using sampled values
            fx_nea, fy_nea = cam_nea_params["fx"], cam_nea_params["fy"]
            cx_nea, cy_nea = cam_nea_params["cx"], cam_nea_params["cy"]
            pts_cam_nea_recon = torch.stack(
                [
                    (pixels_nea_valid[:, 0] - cx_nea) / fx_nea * depths_nea_valid[:, 0],
                    (pixels_nea_valid[:, 1] - cy_nea) / fy_nea * depths_nea_valid[:, 0],
                    depths_nea_valid[:, 0],
                ],
                dim=-1,
            ) # [N, 3]

            # Find the corresponding camera points in reference view
            pts_cam_ref_recon = pts_cam_nea_recon @ R_nea_to_ref + t_nea_to_ref                                # [N, 3]
            pts_reprojections = pts_cam_ref_recon[..., :2] / torch.clamp_min(pts_cam_ref_recon[..., 2:], 1e-6) # [N, 2]
            pts_reprojections = torch.addcmul(
                pts_reprojections.new_tensor([cx_ref, cy_ref]),
                pts_reprojections.new_tensor([fx_ref, fy_ref]),
                pts_reprojections,
            ) # [N, 2]

            # Reference pixel coordinates for the N survivors
            pixel_f = pixel_grid_flat[survived_indices]                       # [N, 2]
            pixel_noise = torch.pairwise_distance(pts_reprojections, pixel_f) # [N,]
            valid_noise = pixel_noise < self.pixel_noise_th                   # [N,] booleans, P trues

            if not valid_noise.any():
                zero_tensor = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                return zero_tensor, zero_tensor

        with torch.no_grad():
            weights_geo = torch.exp(-pixel_noise * self.geo_weights_decay_rate)  # [N,]
            weights_geo[~valid_noise] = 0.0

        if self.optimize_geo:
            Lgeo = (weights_geo * pixel_noise).mean()
            # Multi-view normal consistency: angular error between the reference and the
            # neighbor surface normals at the matched 3D points. Both render_normals are
            # camera-space, so rotate each to world (n_world = n_cam @ R.T) before comparing.
            if compute_angle:
                n_ref = normal_ref.reshape(-1, 3)[survived_indices]  # [N, 3] cam-space (ref)
                n_ref = F.normalize(n_ref @ R_ref.T, dim=-1)         # -> world, unit
                n_nea = sample_normals[valid_occ_proj]               # [N, 3] cam-space (nea, detached)
                n_nea = F.normalize(n_nea @ R_nea.T, dim=-1)         # -> world, unit
                cos_sim = (n_ref * n_nea).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)  # [N,]
                angle_error = torch.acos(cos_sim)                    # [N,], radians
                angle_valid = angle_error < self.angle_noise_th      # [N,]
                if angle_valid.any():
                    angle_noise = self.angle_factor * angle_error
                    Lgeo += (weights_geo * angle_noise)[angle_valid].mean()
        else:
            Lgeo = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        if not self.optimize_ncc:
            zero_tensor = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            return zero_tensor, Lgeo

        # Lncc
        with torch.no_grad():
            # Only keep the P pixels that also passed the noise threshold
            final_indices = survived_indices[valid_noise]  # [P,]
            weights_ncc   = torch.exp(-pixel_noise * self.ncc_weights_decay_rate)[valid_noise]  # [P,]

            # Cap samples for NCC to save computation — applied here only,
            # geo loss above already used all N survivors unrestricted
            if self.max_num_samples > 0 and final_indices.shape[0] > self.max_num_samples:
                chosen = torch.randperm(
                    final_indices.shape[0], device=self.device
                )[: self.max_num_samples]
                final_indices = final_indices[chosen]
                weights_ncc   = weights_ncc[chosen]

            # Recover 2D integer pixel coords for warp_patch_ncc
            pixels_ref_ncc = pixel_grid_flat[final_indices].int()  # [P, 2]

        depth_ref_select = torch.index_select(depth_ref.reshape(-1), dim=0, index=final_indices)         # [P,]
        normal_ref_ = normal_ref.squeeze(0).permute(2, 0, 1)  # [3, H, W] 
        normal_ref_select = torch.index_select(normal_ref_.reshape(3, -1).T, dim=0, index=final_indices) # [P, 3]
        normal_ref_select = F.normalize(normal_ref_select, dim=-1)

        gray_ref = data_ref["gray"].to(self.device).squeeze() / 255.0  # [H, W]
        gray_nea = data_nea["gray"].to(self.device).squeeze() / 255.0  # [H, W] 

        cc, valid_mask = warp_patch_ncc(
            depth_ref_select,
            normal_ref_select,
            pixels_ref_ncc,
            R_ref_to_nea,
            t_ref_to_nea,
            gray_ref,
            gray_nea,
            fx_ref, fy_ref, cx_ref, cy_ref,
            fx_nea, fy_nea, cx_nea, cy_nea,
        )

        ncc = torch.clamp(1 - cc, 0.0, 2.0).squeeze()  # [P,]
        ncc_mask = (ncc < 0.9) & valid_mask.squeeze()  # [P,]
        ncc = ncc * weights_ncc
        ncc = ncc[ncc_mask]

        if ncc_mask.any():
            Lncc = ncc.mean()
        else:
            Lncc = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        return Lncc, Lgeo


def cam_params_from_data_batch(data_batch):
    gt_image = data_batch["image"].squeeze() # [H, W, 3]
    K = data_batch["K"].squeeze()            # [3, 3]
    H, W = gt_image.shape[:2]
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    return { "H": H, "W": W, "fx": fx, "fy": fy, "cx": cx, "cy": cy }


def sample_map(pixels, target):
    """
    Perform grid sampling into target at pixels
    Args:
        pixels: image coordinates to sample the target, shape [M, 2]
        target: the sample target, shape [1, H, W, C]

    Returns:
        sample values of shape [M, C]
    """

    w, h = target.shape[1:3]
    normalized_pixels = torch.stack(
        [
            pixels[:, 0] / (w - 1) * 2 - 1,
            pixels[:, 1] / (h - 1) * 2 - 1,
        ],
        dim=-1,
    ) # [M, 2], normalize to [-1, 1]

    # Prepare for grid sampling
    input = target.permute(0, 3, 1, 2)          # [1, C, H, W]
    grid  = normalized_pixels[None, :, None, :] # [1, M, 1, 2]

    # Bilinear sampling
    samples = F.grid_sample(input, grid, mode="bilinear", align_corners=True)  # [1, C, M, 1]
    samples = samples.squeeze(0).squeeze(-1)  # [C, M]
    samples = samples.transpose(0, 1)         # [M, C]

    return samples


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ref: https://github.com/hbb1/2d-gaussian-splatting/blob/main/utils/general_utils.py#L163
def colormap(img, cmap="jet"):
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H / dpi, W / dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data).float().permute(2, 0, 1)
    plt.close()
    return img


def apply_float_colormap(img: torch.Tensor, colormap: str = "turbo") -> torch.Tensor:
    """Convert single channel to a color img.

    Args:
        img (torch.Tensor): (..., 1) float32 single channel image.
        colormap (str): Colormap for img.

    Returns:
        (..., 3) colored img with colors in [0, 1].
    """
    img = torch.nan_to_num(img, 0)
    if colormap == "gray":
        return img.repeat(1, 1, 3)
    img_long = (img * 255).long()
    img_long_min = torch.min(img_long)
    img_long_max = torch.max(img_long)
    assert img_long_min >= 0, f"the min value is {img_long_min}"
    assert img_long_max <= 255, f"the max value is {img_long_max}"
    return torch.tensor(
        colormaps[colormap].colors,  # type: ignore
        device=img.device,
    )[img_long[..., 0]]


def apply_depth_colormap(
    depth: torch.Tensor,
    acc: torch.Tensor = None,
    near_plane: float = None,
    far_plane: float = None,
) -> torch.Tensor:
    """Converts a depth image to color for easier analysis.

    Args:
        depth (torch.Tensor): (..., 1) float32 depth.
        acc (torch.Tensor | None): (..., 1) optional accumulation mask.
        near_plane: Closest depth to consider. If None, use min image value.
        far_plane: Furthest depth to consider. If None, use max image value.

    Returns:
        (..., 3) colored depth image with colors in [0, 1].
    """
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))
    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0.0, 1.0)
    img = apply_float_colormap(depth, colormap="turbo")
    if acc is not None:
        img = img * acc + (1.0 - acc)
    return img


def image_grad_weight(image):
    # image: [..., H, W, 3]
    *batch_dims, H, W, C = image.shape
    assert C == 3

    # Move channels to NCHW
    img = image.permute(*range(len(batch_dims)), -1, -3, -2)  # [..., 3, H, W]

    # Gradients
    bottom = img[..., :, 2:H,   1:W-1]
    top    = img[..., :, 0:H-2, 1:W-1]
    right  = img[..., :, 1:H-1, 2:W]
    left   = img[..., :, 1:H-1, 0:W-2]

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


def fix_normal_coordinates(normal_map):
    # Flatten and normalize normals
    normals = normal_map.view(-1, 3).clone()  # [H * W, 3]
    normals = F.normalize(normals, dim=1, p=2)

    # Apply Y-up/Z-back coordinate fix
    T = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=normals.dtype, device=normals.device)
    normals = normals @ T.T

    # Adjust range
    normals = normals * 0.5 + 0.5 # [-1, 1] -> [0, 1]
    H, W = normal_map.shape[1:3]
    return normals.view(1, H, W, 3)


def post_process_mesh(mesh, cluster_to_keep=128, clusters_to_skip=[], min_triangles=0, decimate_target=0):
    """
    Filter out disconnected parts and performs mesh decimation.
    clusters_to_skip: list of 1-based ranks (after sorting by triangle count desc)
                      to exclude. e.g. clusters_to_skip=[2] removes the 2nd largest.
                      cluster_to_keep then applies to whatever ranks remain.
    """
    post = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        triangle_clusters, cluster_n_triangles, _ = post.cluster_connected_triangles()
    triangle_clusters   = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    print(f"[>] Found {len(cluster_n_triangles):,} clusters, largest has {cluster_n_triangles.max():,} triangles")

    # Sort all cluster IDs by triangle count descending → gives us a stable rank order
    sorted_ids = np.argsort(cluster_n_triangles)[::-1]   # sorted_ids[0] = rank-1 cluster ID

    # Build skip set from 1-based ranks in sorted order
    skip_set = set()
    if len(clusters_to_skip) > 0:
        for rank in clusters_to_skip:
            if 1 <= rank <= len(sorted_ids):
                skip_set.add(int(sorted_ids[rank - 1]))
        print(f"[>] Skipping rank(s) {clusters_to_skip} → cluster id(s) {skip_set}")

    # Remaining candidates in rank order (skipped ones removed)
    eligible_ids = [cid for cid in sorted_ids if cid not in skip_set]

    # Build rank_mask over all clusters (False by default)
    rank_mask = np.zeros(len(cluster_n_triangles), dtype=bool)
    if cluster_to_keep == 0:
        print(f"[>] Keeping all eligible clusters ({len(eligible_ids)} after skips)")
        keep_ids = eligible_ids
    else:
        print(f"[>] Keeping {cluster_to_keep} clusters in decreasing number of triangles")
        keep_ids = eligible_ids[:cluster_to_keep]   # top-N from the post-skip ranked list

    for cid in keep_ids:
        rank_mask[cid] = True

    # Per-triangle masks
    counts_per_triangle = cluster_n_triangles[triangle_clusters]
    size_mask = (counts_per_triangle >= min_triangles) if min_triangles > 0 \
        else np.ones(len(triangle_clusters), dtype=bool)

    cluster_passes_rank = rank_mask[triangle_clusters]
    triangles_to_remove = ~(cluster_passes_rank & size_mask)

    post.remove_triangles_by_mask(triangles_to_remove)
    post.compute_vertex_normals()
    post.remove_duplicated_vertices()
    post.remove_unreferenced_vertices()
    post.remove_degenerate_triangles()

    n_tri = len(np.asarray(post.triangles))
    n_tri_decimate = 0
    if decimate_target > 0 and n_tri > decimate_target:
        n_tri_decimate = n_tri - decimate_target
        print(f"[>] Decimating {n_tri:,} to {decimate_target:,} triangles...")
        post = post.simplify_quadric_decimation(decimate_target)
        post.compute_vertex_normals()
        post.remove_degenerate_triangles()
        post.remove_unreferenced_vertices()
    else:
        print(f"[>] Skipping decimation")

    remaining = len(np.asarray(post.triangles))
    print(f"[>] Removed {triangles_to_remove.sum() + n_tri_decimate:,} triangles, {remaining:,} remaining")
    print(f"[>] Num vertices post-process: {len(post.vertices):,}")
    return post


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0/2.4) - 0.055)


def write_mesh_glb(path, mesh):
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.uint32)
    colors = np.asarray(mesh.vertex_colors, dtype=np.float32)  # [N,3] float [0,1]

    # Pack into binary blobs
    verts_blob  = vertices.tobytes()
    faces_blob  = faces.tobytes()
    colors_blob = colors.tobytes()

    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[pygltflib.Node(mesh=0)],
        meshes=[pygltflib.Mesh(primitives=[
            pygltflib.Primitive(
                attributes=pygltflib.Attributes(
                    POSITION=0,
                    COLOR_0=1,
                ),
                indices=2,
                material=0,
            )
        ])],
        materials=[pygltflib.Material(
            name="vertex_color_flat",
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorFactor=[1.0, 1.0, 1.0, 1.0],  # white, so COLOR_0 shows through
                metallicFactor=0.0,
                roughnessFactor=1.0,
            ),
            extensions={"KHR_materials_unlit": {}},  # unlit = flat shading
        )],
        accessors=[
            # 0: positions
            pygltflib.Accessor(bufferView=0, componentType=pygltflib.FLOAT,
                               count=len(vertices), type=pygltflib.VEC3,
                               max=vertices.max(axis=0).tolist(),
                               min=vertices.min(axis=0).tolist()),
            # 1: colors
            pygltflib.Accessor(bufferView=1, componentType=pygltflib.FLOAT,
                               count=len(colors), type=pygltflib.VEC3),
            # 2: indices
            pygltflib.Accessor(bufferView=2, componentType=pygltflib.UNSIGNED_INT,
                               count=faces.size, type=pygltflib.SCALAR),
        ],
        bufferViews=[
            pygltflib.BufferView(buffer=0, byteOffset=0,               byteLength=len(verts_blob)),
            pygltflib.BufferView(buffer=0, byteOffset=len(verts_blob), byteLength=len(colors_blob)),
            pygltflib.BufferView(buffer=0, byteOffset=len(verts_blob)+len(colors_blob), byteLength=len(faces_blob)),
        ],
        buffers=[pygltflib.Buffer(byteLength=len(verts_blob)+len(colors_blob)+len(faces_blob))],
        extensionsUsed=["KHR_materials_unlit"],
    )

    gltf.set_binary_blob(verts_blob + colors_blob + faces_blob)
    gltf.save(str(path))


def write_mesh(path, mesh):
    if path.suffix.lower() == ".glb":
        print(f"[>] Exporting mesh as GLB format")
        write_mesh_glb(path, mesh)
    else:
        print(f"[>] Exporting mesh as PLY format")
        o3d.io.write_triangle_mesh(
            str(path), mesh,
            write_triangle_uvs=True,
            write_vertex_colors=True,
            write_vertex_normals=True
        )
    print(f"[>] Mesh written to: {path}")
