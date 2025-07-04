import os
import torch
import uuid

from pbr import pbr_render, linear_to_srgb

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
        scene, metallic, gamma, rgb_render, pipe, white, pbr_stats, canonical_rays):
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

        bg_color = [1, 1, 1] if white else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        envmap = scene.cubemap.export_envmap(return_img=True).clamp(min=0.0).permute(2, 0, 1) # (3, H, W)
        if tb_writer:
            tb_writer.add_images("scene/envmap", envmap[None], global_step=iteration)

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test_rgb = 0.0
                l1_test_pbr = 0.0
                psnr_test_rgb = 0.0
                psnr_test_pbr = 0.0

                for idx, viewpoint in enumerate(config['cameras']):
                    stem = viewpoint.image_name.rsplit('.', 1)[0]
                    gt_image = torch.clamp(viewpoint.gt_image.to("cuda"), 0.0, 1.0)

                    render_pkg = rgb_render(viewpoint, scene.gaussians, pipe, background, True, True, True)
                    pbr_pkg = pbr_render(scene, viewpoint, canonical_rays, render_pkg, metallic, gamma)

                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    alpha_map = render_pkg["alpha_map"] # (1, H, W)

                    normal_map = convert_normal_for_save(render_pkg["normal_map"], viewpoint)
                    sobel_map = convert_normal_for_save(render_pkg["sobel_map"], viewpoint)
                    depth_map = convert_depth_for_save(render_pkg["depth_map"])

                    albedo_map = render_pkg["albedo_map"].clamp(0.0, 1.0)
                    roughness_map = pbr_pkg["roughness_map"]
                    metallic_map = pbr_pkg["metallic_map"]

                    diffuse_map = linear_to_srgb(pbr_pkg["diffuse_rgb"]) if gamma else pbr_pkg["diffuse_rgb"]
                    diffuse_map = diffuse_map.clamp(0.0, 1.0).permute(2, 0, 1) # (3, H, W)
                    specular_map = linear_to_srgb(pbr_pkg["specular_rgb"]) if gamma else pbr_pkg["specular_rgb"]
                    specular_map = specular_map.clamp(0.0, 1.0).permute(2, 0, 1) # (3, H, W)
                    pbr_image = pbr_pkg["render_rgb"].clamp(0.0, 1.0).permute(2, 0, 1) # (3, H, W)

                    if white:
                        alpha_mask = viewpoint.alpha_mask.cuda() > 0.5
                        gt_image = torch.where(alpha_mask, gt_image, background[:, None, None])

                        pbr_image = torch.where(alpha_mask, pbr_image, 1.0)
                        diffuse_map = torch.where(alpha_mask, diffuse_map, 1.0)
                        specular_map = torch.where(alpha_mask, specular_map, 1.0)
                        albedo_map = torch.where(alpha_mask, albedo_map, 1.0)
                        roughness_map = torch.where(alpha_mask, roughness_map, 1.0)
                        metallic_map = torch.where(alpha_mask, metallic_map, 1.0)

                        normal_map = torch.where(alpha_mask, normal_map, 1.0)
                        sobel_map = torch.where(alpha_mask, sobel_map, 1.0)
                    else:
                        normal_mask = render_pkg["normal_mask"] # (1, H, W)

                        pbr_image = torch.where(normal_mask, pbr_image, background[:, None, None])
                        diffuse_map = torch.where(normal_mask, diffuse_map, background[:, None, None])
                        specular_map = torch.where(normal_mask, specular_map, background[:, None, None])

                    if tb_writer and (idx < 5):
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + f"_{stem}/ground_truth", gt_image[None], global_step=iteration)

                        tb_writer.add_images(config['name'] + f"_{stem}/rgb_render", image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + f"_{stem}/alpha_map", alpha_map[None], global_step=iteration)

                        tb_writer.add_images(config['name'] + f"_{stem}/normal_map", normal_map[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + f"_{stem}/sobel_map", sobel_map[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + f"_{stem}/depth_map", depth_map[None], global_step=iteration)

                        tb_writer.add_images(config['name'] + f"_{stem}/albedo_map", albedo_map[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + f"_{stem}/roughness_map", roughness_map[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + f"_{stem}/metallic_map", metallic_map[None], global_step=iteration)
    
                        tb_writer.add_images(config['name'] + f"_{stem}/z_pbr_render", pbr_image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + f"_{stem}/z_shade_diffuse", diffuse_map[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + f"_{stem}/z_shade_specular", specular_map[None], global_step=iteration)

                    l1_test_rgb += l1_loss(image, gt_image).mean().double()
                    l1_test_pbr += l1_loss(pbr_image, gt_image).mean().double()
                    psnr_test_rgb += psnr(image, gt_image).mean().double()
                    psnr_test_pbr += psnr(pbr_image, gt_image).mean().double()

                psnr_test_rgb /= len(config['cameras'])
                psnr_test_pbr /= len(config['cameras'])
                l1_test_rgb /= len(config['cameras'])
                l1_test_pbr /= len(config['cameras'])

                if not pbr_stats:
                    tqdm.write(f"[ITER {iteration:>5}] Evaluating {config['name']:>5} (rgb): L1 - {l1_test_rgb:.4f} | PSNR - {psnr_test_rgb:.2f}")
                else:
                    tqdm.write(f"[ITER {iteration:>5}] Evaluating {config['name']:>5} (PBR): L1 - {l1_test_pbr:.4f} | PSNR - {psnr_test_pbr:.2f}")

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss_rgb', l1_test_rgb, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss_pbr', l1_test_pbr, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr_rgb', psnr_test_rgb, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr_pbr', psnr_test_pbr, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()