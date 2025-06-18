import os
import torch
import uuid

import torch.nn.functional as F
import nvdiffrast.torch as dr

from argparse import Namespace
from pathlib import Path
from tqdm import tqdm

from utils.image_utils import psnr, convert_normal_for_save, convert_depth_for_save
from utils.loss_utils import l1_loss

def prepare_outdir(args):
    # Generate a random name for output directory if not provided
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print(f"[>] Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok = True)

    # Save the configuration arguments to a file
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

def prepare_logger(args):
    # Create Tensorboard writer if available
    tb_writer = None

    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(args.model_path)

        # Check and remove existing tensorboard log file
        log_dir = Path(args.model_path)
        for file in log_dir.glob("events.out.tfevents.*"):
            file.unlink()
    except ImportError:
        print("[>] Tensorboard not available: not logging progress")
    
    return tb_writer

def report_training(
        tb_writer, iteration, Lrgb, Lgeo, Lmat, loss, elapsed, testing_iterations,
        scene, metallic, gamma, render_func, render_args, pbr_func, pbr_stats, canonical_rays):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/Lrgb', Lrgb.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/Lgeo', Lgeo.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/Lmat', Lmat.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/Loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test',  'cameras': scene.getTestCameras()}, 
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        
        _, background = render_args
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test_rgb = 0.0
                l1_test_pbr = 0.0
                psnr_test_rgb = 0.0
                psnr_test_pbr = 0.0

                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = render_func(viewpoint, scene.gaussians, *render_args)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.gt_image.to("cuda"), 0.0, 1.0)
                    stem = viewpoint.image_name.rsplit('.', 1)[0]

                    H, W = viewpoint.image_height, viewpoint.image_width
                    c2w = viewpoint.world_view_transform[:3, :3]
                    view_dirs = -(canonical_rays @ c2w.T).reshape(H, W, 3)

                    normals = render_pkg["normal_map"].permute(1, 2, 0).reshape(-1, 3) # (H * W, 3)
                    normals = normals @ c2w.T # (H * W, 3)
                    normal_map = normals.reshape(H, W, 3) # (H, W, 3)
                    normal_map = torch.where(
                        torch.norm(normal_map, dim=0, keepdim=True) > 0,
                        F.normalize(normal_map, dim=0, p=2), normal_map)

                    normal_mask = render_pkg["normal_mask"] # (1, H, W)
                    scene.cubemap.build_mips() # build mip for environment light

                    albedo_map = render_pkg["albedo_map"] # (3, H, W)
                    metallic_map = render_pkg["metallic_map"] # (1, H, W)
                    roughness_map = render_pkg["roughness_map"] # (1, H, W)

                    pbr_pkg = pbr_func(
                        light=scene.cubemap,
                        normals=normal_map,
                        view_dirs=view_dirs,
                        mask=normal_mask.permute(1, 2, 0), # (H, W, 1)
                        albedo=albedo_map.permute(1, 2, 0), # (H, W, 3)
                        roughness=roughness_map.permute(1, 2, 0), # (H, W, 1)
                        metallic=metallic_map.permute(1, 2, 0) if metallic else None, # (H, W, 1)
                        occlusion=torch.ones_like(roughness_map).permute(1, 2, 0),
                        irradiance=torch.zeros_like(roughness_map).permute(1, 2, 0),
                        brdf_lut=scene.brdf_lut,
                        gamma=gamma)
                    
                    diffuse_map = pbr_pkg["diffuse_rgb"].clamp(0.0, 1.0).permute(2, 0, 1) # (3, H, W)
                    specular_map = pbr_pkg["specular_rgb"].clamp(0.0, 1.0).permute(2, 0, 1) # (3, H, W)
                    pbr_render = pbr_pkg["render_rgb"].clamp(0.0, 1.0).permute(2, 0, 1) # (3, H, W)

                    pbr_render = torch.where(normal_mask, pbr_render, background[:, None, None])
                    diffuse_map = torch.where(normal_mask, diffuse_map, background[:, None, None])
                    specular_map = torch.where(normal_mask, specular_map, background[:, None, None])

                    envmap = dr.texture(
                        scene.cubemap.base[None, ...], scene.envmap_dirs[None, ...].contiguous(),
                        filter_mode="linear", boundary_mode="cube")[0].permute(2, 0, 1) # (3, H, W)

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + f"_{stem}/rgb_render", image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + f"_{stem}/ground_truth", gt_image[None], global_step=iteration)

                        for key in render_pkg.keys():
                            if 'map' in key and 'dn' not in key and 'distance' not in key:
                                if 'normal' in key:
                                    render_pkg[key] = convert_normal_for_save(render_pkg[key], viewpoint)
                                if 'depth' in key:
                                    render_pkg[key] = convert_depth_for_save(render_pkg[key])
                                tb_writer.add_images(config['name'] + f"_{stem}/{key}", render_pkg[key][None], global_step=iteration)
    
                        tb_writer.add_images(config['name'] + f"_{stem}/z_pbr_render", pbr_render[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + f"_{stem}/z_shade_diffuse", diffuse_map[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + f"_{stem}/z_shade_specular", specular_map[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + f"_{stem}/z_uv_envmap", envmap[None], global_step=iteration)

                    l1_test_rgb += l1_loss(image, gt_image).mean().double()
                    l1_test_pbr += l1_loss(pbr_render, gt_image).mean().double()
                    psnr_test_rgb += psnr(image, gt_image).mean().double()
                    psnr_test_pbr += psnr(pbr_render, gt_image).mean().double()

                psnr_test_rgb /= len(config['cameras'])
                psnr_test_pbr /= len(config['cameras'])
                l1_test_rgb /= len(config['cameras'])
                l1_test_pbr /= len(config['cameras'])

                if not pbr_stats:
                    tqdm.write(f"[ITER {iteration:>5}] Evaluating {config['name']} (RGB): L1 - {l1_test_rgb:.4f} | PSNR - {psnr_test_rgb:.2f}")
                else:
                    tqdm.write(f"[ITER {iteration:>5}] Evaluating {config['name']} (PBR): L1 - {l1_test_pbr:.4f} | PSNR - {psnr_test_pbr:.2f}")

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss_rgb', l1_test_rgb, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss_pbr', l1_test_pbr, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr_rgb', psnr_test_rgb, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr_pbr', psnr_test_pbr, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()