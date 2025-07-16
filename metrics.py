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

import json
import os
import torch
import torchvision.transforms.functional as tf

from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips

def read_images(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_path, split, method):
    split_dir = Path(model_path) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory {split_dir} does not exist, did you forget to specify --split?.")

    method_dir = split_dir / method
    gt_dir = method_dir/ "gt"
    render_dir = method_dir / "render"
    renders, gts, _ = read_images(render_dir, gt_dir)

    ssims = []
    psnrs = []
    lpipss = []

    print(f"[>] Evaluate metrics for: {method_dir}")

    for idx in tqdm(range(len(renders)), desc="[>] Evaluating", ncols=80):
        ssims.append(ssim(renders[idx], gts[idx]))
        psnrs.append(psnr(renders[idx], gts[idx]))
        lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

    print("[-] SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
    print("[-] PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
    print("[-] LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))

    metrics = {}
    metric_file = Path(model_path) / "metrics.json"
    if metric_file.exists():
        with open(metric_file, 'r') as f:
            metrics = json.load(f)

    metrics[method] = {
        "ssim": torch.tensor(ssims).mean().item(),
        "psnr": torch.tensor(psnrs).mean().item(),
        "lpips": torch.tensor(lpipss).mean().item(),
    }

    with open(metric_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"[>] Metrics saved to: {metric_file}")

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_path', '-m', required=True, type=str)
    parser.add_argument('--split', type=str, default="test", help="Split to evaluate on (train/test)")
    parser.add_argument('--method', type=str, default="ours_30000")

    args = parser.parse_args()
    evaluate(args.model_path, args.split, args.method)