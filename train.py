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

import os
import sys
import torch

from random import randint
from tqdm import tqdm

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams

from gaussian_renderer import render
from scene import Scene, GaussianModel
from fused_ssim import fused_ssim

from pbr import pbr_shading
import nvdiffrast.torch as dr
import torch.nn.functional as F

from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, planar_loss, sparse_loss, tv_loss, masked_tv_loss, multi_view_loss, depth_normal_loss, luminance_loss
from utils.training_utils import prepare_outdir, prepare_logger, report_training

def train(model, opt, pipe, test_iterations, save_iterations, checkpoint_iterations, checkpoint):
    prepare_outdir(model)
    tb_writer = prepare_logger(model)

    gaussians = GaussianModel(model.sh_degree)
    scene = Scene(model, gaussians)
    gaussians.training_setup(opt)
    scene.training_setup(opt, model)
    canonical_rays = F.normalize(scene.get_canonical_rays(), p=2, dim=-1)

    first_iter = 0
    if checkpoint:
        ckp = torch.load(checkpoint)
        model_params = ckp["gaussians"]
        first_iter = ckp["iteration"]
        cubemap_state = ckp["cubemap"]
        light_optimizer_state = ckp["light_optimizer"]

        gaussians.restore(model_params, opt)
        scene.cubemap.load_state_dict(cubemap_state)
        scene.light_optimizer.load_state_dict(light_optimizer_state)

    bg_color = [1, 1, 1] if model.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end   = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_geo_loss_for_log = 0.0
    ema_mat_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="[>] Training", ncols=128)
    first_iter += 1 # off-by-one progress bar
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 iterations we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        geometry_stage = iteration > opt.geometry_from_iter
        material_stage = iteration > opt.material_from_iter
        render_pkg = render(
            viewpoint_cam, gaussians, pipe, background, geometry_stage, material_stage,
            sobel_normal=geometry_stage, inference=False, pad_normal=False)
        image, visibility_filter, radii = render_pkg["render"], render_pkg["visibility_filter"], render_pkg["radii"]
        gt_image = viewpoint_cam.gt_image.cuda()

        # RGB losses
        Lc = l1_loss(image, gt_image)
        lambda_ssim = opt.lambda_ssim
        Lssim = 1.0 - fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        Lrgb = lambda_ssim * Lssim
        if not material_stage or Lssim >= 0.5:
            Lrgb += (1.0 - lambda_ssim) * Lc

        # Planar and sparse losses
        Lplanar = planar_loss(visibility_filter, gaussians)
        Lsparse = sparse_loss(render_pkg["alpha_map"]) if opt.use_sparse_loss else 0.0

        # Total loss
        loss = Lrgb + opt.lambda_planar * Lplanar + opt.lambda_sparse * Lsparse

        # Geometry losses
        Lgeo = torch.tensor([0.0])
        if geometry_stage:
            # n_map = render_pkg["normal_map"] # (3, H, W)
            # n_map = torch.where(torch.norm(n_map, dim=0, keepdim=True) > 0, F.normalize(n_map, dim=0, p=2), n_map)
            # Ltv = tv_loss(gt_image, render_pkg["normal_map"])

            lambda_mv = opt.lambda_multi_view # expensive, only call if needed
            mv_args = (scene, viewpoint_cam, opt, render_pkg, pipe, background, material_stage)
            Lmv = 0.0 if lambda_mv == 0.0 else multi_view_loss(*mv_args)

            weight_map = None
            diffuse_ref = gt_image
            if material_stage:
                roughness_map = render_pkg["roughness_map"] # (1, H, W)
                weight_map = 1.0 + 4.0 * (1.0 - roughness_map).clamp(0, 1).detach()
                diffuse_ref = image.detach()
                # Ldn = (weight_map * (render_pkg["sobel_normal_map"] - render_pkg["normal_map"]).abs().sum(dim=0)).mean()
                # Ltv = laplacian_loss(render_pkg["normal_map"], smooth_map)
                # Ltv_d = laplacian_loss(render_pkg["depth_map"], smooth_map)
                # Ltv = Ltv_d + Ltv_n
            lambda_dn = opt.lambda_depth_normal
            # Ldn = (render_pkg["sobel_normal_map"] - render_pkg["normal_map"]).abs().sum(dim=0).mean()
            Ldn = depth_normal_loss(render_pkg["normal_map"], render_pkg["sobel_normal_map"], diffuse_ref, weight_map)

            # lambda_tv = opt.lambda_tv_normal
            # Ltv = weighted_tv_loss(gt_image, render_pkg["normal_map"], weight_map)

            Lgeo = lambda_dn * Ldn + lambda_mv * Lmv # + lambda_tv * Ltv
            loss += Lgeo

        # Material losses
        Lmat = torch.tensor([0.0])
        if material_stage:
            H, W = viewpoint_cam.image_height, viewpoint_cam.image_width
            c2w = viewpoint_cam.world_view_transform[:3, :3]
            view_dirs = -(canonical_rays @ c2w.T).reshape(H, W, 3) # (H, W, 3)

            # Normals to world space
            normals = render_pkg["normal_map"].permute(1, 2, 0).reshape(-1, 3) # (H * W, 3)
            normals = normals @ c2w.T # (H * W, 3)
            normal_map = normals.reshape(H, W, 3).permute(2, 0, 1) # (3, H, W)
            normal_map = torch.where(
                torch.norm(normal_map, dim=0, keepdim=True) > 0,
                F.normalize(normal_map, dim=0, p=2), normal_map)

            normal_mask = render_pkg["normal_mask"] # (1, H, W)
            scene.cubemap.build_mips() # build mip for environment light

            albedo_map = render_pkg["albedo_map"] # (3, H, W)
            metallic_map = render_pkg["metallic_map"] # (1, H, W)
            roughness_map = render_pkg["roughness_map"] # (1, H, W)
            # rmax, rmin = 1.0, 0.001
            # roughness_map = roughness_map * (rmax - rmin) + rmin

            if not model.metallic:
                metallic_map = (1.0 - render_pkg["roughness_map"]).clamp(0, 1).detach() # (1, H, W)
                metallic_map = torch.where(normal_mask, metallic_map, background[:, None, None])

            pbr_pkg = pbr_shading(
                light=scene.cubemap,
                normals=normal_map.permute(1, 2, 0).detach(), # (H, W, 3)
                view_dirs=view_dirs,
                mask=normal_mask.permute(1, 2, 0), # (H, W, 1)
                albedo=albedo_map.permute(1, 2, 0), # (H, W, 3)
                roughness=roughness_map.permute(1, 2, 0), # (H, W, 1)
                metallic=metallic_map.permute(1, 2, 0), # (H, W, 1)
                occlusion=torch.ones_like(roughness_map).permute(1, 2, 0),
                irradiance=torch.zeros_like(roughness_map).permute(1, 2, 0),
                brdf_lut=scene.brdf_lut,
                gamma=model.gamma,
                white_background=model.white_background)

            render_pbr = pbr_pkg["render_rgb"].permute(2, 0, 1) # (3, H, W)
            render_pbr = torch.where(normal_mask, render_pbr, background[:, None, None])
            Lpbr = l1_loss(render_pbr, gt_image)

            # Smoothness loss
            sm_maps = [albedo_map, roughness_map, metallic_map] if model.metallic else [albedo_map, roughness_map]
            arm = torch.cat(sm_maps, dim=0)
            lambda_tv_smooth = opt.lambda_tv_smooth
            Lsm = masked_tv_loss(normal_mask, diffuse_ref, arm) if (normal_mask == 0).sum() > 0 else tv_loss(diffuse_ref, arm)

            # Environment light loss
            # envmap = dr.texture(
            #     scene.cubemap.base[None, ...], scene.envmap_dirs[None, ...].contiguous(),
            #     filter_mode="linear", boundary_mode="cube")[0] # (H, W, 3)
            # tv_h1 = torch.pow(envmap[1:, :, :] - envmap[:-1, :, :], 2).mean()
            # tv_w1 = torch.pow(envmap[:, 1:, :] - envmap[:, :-1, :], 2).mean()
            # lambda_tv_envmap = opt.lambda_tv_envmap
            # Lenv = tv_h1 + tv_w1

            # Luminance loss
            diffuse_map = pbr_pkg["diffuse_rgb"].permute(2, 0, 1) # (3, H, W)
            diffuse_map = torch.where(normal_mask, diffuse_map, background[:, None, None])
            lambda_luminance = opt.lambda_luminance
            # Llm = luminance_loss(diffuse_map, diffuse_ref, normal_mask)
            Llm = (diffuse_map - diffuse_ref).abs().mean()

            Lmat = (1.0 - lambda_ssim) * Lpbr + lambda_tv_smooth * Lsm # + lambda_luminance * Llm
            loss += Lmat

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_geo_loss_for_log = 0.4 * Lgeo.item() + 0.6 * ema_geo_loss_for_log
            ema_mat_loss_for_log = 0.4 * Lmat.item() + 0.6 * ema_mat_loss_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Lgeo": f"{ema_geo_loss_for_log:.{5}f}",
                    "Lmat": f"{ema_mat_loss_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}",
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)

            # Log and save
            pbr_stats = iteration > opt.material_from_iter
            report_training(
                tb_writer, iteration, Lrgb, Lgeo, Lmat, loss, iter_start.elapsed_time(iter_end), test_iterations,
                scene, model.metallic, model.gamma, render, (pipe, background), pbr_shading, pbr_stats, canonical_rays)

            if (iteration in save_iterations):
                tqdm.write(f"[ITER {iteration:>5}] Saving Gaussians and lighting")
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                mask = (render_pkg["observe"] > 0) & visibility_filter
                gaussians.max_radii2D = torch.where(mask, torch.max(gaussians.max_radii2D, radii), gaussians.max_radii2D)
                gaussians.add_densification_stats(render_pkg["viewspace_points"], visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    radii2D_threshold = opt.radii2D_threshold if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, opt.densify_grad_abs_threshold, opt.opacity_prune_threshold,
                        scene.cameras_extent, radii2D_threshold)

            # Multi-view observe trim
            if opt.use_multi_view_trim and iteration % 1000 == 0 and iteration < opt.densify_until_iter:
                observe_the = 2
                observe_cnt = torch.zeros_like(gaussians.get_opacity)
                for view in scene.getTrainCameras():
                    render_pkg_tmp = render(view, gaussians, pipe, background, sobel_normal=False)
                    out_observe = render_pkg_tmp["observe"]
                    observe_cnt[out_observe > 0] += 1
                prune_mask = (observe_cnt < observe_the).squeeze()
                if prune_mask.sum() > 0:
                    gaussians.prune_points(prune_mask)

            if iteration < opt.densify_until_iter:
                if iteration % opt.opacity_reduce_interval == 0 and opt.use_opacity_reduce:
                    # Periodically reduce opacity to remove floaters
                    gaussians.reduce_opacity()

                if iteration % opt.opacity_reset_interval == 0 or (model.white_background and iteration == opt.densify_from_iter):
                    # Moderate the increase in the number of Gaussians by setting opacity close to zero 
                    gaussians.reset_opacity()

             # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                if iteration > opt.material_from_iter:
                    scene.light_optimizer.step()
                    scene.light_optimizer.zero_grad(set_to_none=True)
                    scene.cubemap.clamp_(min=0.0)

            if (iteration in checkpoint_iterations):
                tqdm.write(f"[ITER {iteration}] Saving checkpoint")
                ckp_dir = scene.model_path + "/checkpoints"
                os.makedirs(ckp_dir, exist_ok=True)
                ckp_dict = {
                    "gaussians": gaussians.capture(),
                    "cubemap": scene.cubemap.state_dict(),
                    "light_optimizer": scene.light_optimizer.state_dict(),
                    "iteration": iteration,
                }
                torch.save(ckp_dict, ckp_dir + "/ckp" + str(iteration) + ".pth")

            if iteration == opt.iterations:
                progress_bar.close()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    model_params = ModelParams(parser)
    optimization_params = OptimizationParams(parser)
    pipeline_params = PipelineParams(parser)

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 7_000, 10_000, 15_000, 20_000, 25_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("[>] Optimizing: " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Configure and start training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    train(
        model_params.extract(args), optimization_params.extract(args), pipeline_params.extract(args),
        args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    print("[>] Training complete!")