import torch
import torch.nn.functional as F

from .light import CubemapLight
from .shade import get_brdf_lut, pbr_shading, saturate_dot, linear_to_srgb, srgb_to_linear

__all__ = ["CubemapLight", "get_brdf_lut", "pbr_shading", "saturate_dot", "linear_to_srgb", "srgb_to_linear"]

def pbr_render(scene, viewpoint_cam, canonical_rays, render_pkg, metallic, gamma=False):
    # Build mips for environment light
    scene.cubemap.build_mips()

    # View directions in camera space
    H, W = viewpoint_cam.image_height, viewpoint_cam.image_width
    c2w = viewpoint_cam.world_view_transform[:3, :3]
    view_dirs = (-canonical_rays @ c2w.T) # (H * W, 3)
    view_dirs = F.normalize(view_dirs, p=2, dim=-1)
    view_dirs = view_dirs.reshape(H, W, 3) # (H, W, 3)

    # Normals to world space
    # normals = render_pkg["normal_map"].permute(1, 2, 0).reshape(-1, 3) # (H * W, 3)
    # normals = normals @ c2w.T # (H * W, 3)
    # normal_map = normals.reshape(H, W, 3).permute(2, 0, 1) # (3, H, W)
    normal_map = render_pkg["normal_map"].detach() # (3, H, W)
    normal_map = torch.where(torch.norm(normal_map, dim=0, keepdim=True) > 0, F.normalize(normal_map, dim=0, p=2), normal_map)

    albedo_map = render_pkg["albedo_map"].clamp(0, 1) # (3, H, W)
    metallic_map = render_pkg["metallic_map"] # (1, H, W)
    roughness_map = render_pkg["roughness_map"] # (1, H, W)

    # If not training metallic, estimate it from roughness
    if not metallic:
        alpha_map = render_pkg["alpha_map"].detach() # (1, H, W)
        metallic_map = (1.0 - roughness_map).clamp(0, 1) # (1, H, W)
        metallic_map = alpha_map * metallic_map # (1, H, W)
        metallic_map = metallic_map.detach()

    rmin, rmax = 0.04, 1.0
    roughness_map = roughness_map * (rmax - rmin) + rmin
    roughness_map = roughness_map.detach()

    pbr_pkg = pbr_shading(
        light=scene.cubemap,
        normals=normal_map.permute(1, 2, 0), # (H, W, 3)
        view_dirs=view_dirs,
        albedo=albedo_map.permute(1, 2, 0), # (H, W, 3)
        roughness=roughness_map.permute(1, 2, 0), # (H, W, 1)
        metallic=metallic_map.permute(1, 2, 0), # (H, W, 1)
        occlusion=torch.ones_like(roughness_map).permute(1, 2, 0),
        irradiance=torch.zeros_like(roughness_map).permute(1, 2, 0),
        brdf_lut=scene.brdf_lut,
        gamma=gamma)

    pbr_pkg.update({ "roughness_map": roughness_map, "metallic_map": metallic_map })
    return pbr_pkg

