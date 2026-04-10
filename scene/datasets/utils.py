import os
import numpy as np

from pathlib import Path
from PIL import Image
from tqdm import tqdm


def process_input_images(image_dir, target_dir, image_names, factor, reuse=False, mask_image=False, mask_dir=None):
    output_dir = Path(target_dir)
    os.makedirs(output_dir, exist_ok=True)

    original_dir = Path(image_dir)
    if (factor > 1):
        print(f"[>] Training with images downscaled by {factor}x in {output_dir}")

    # Remove existing files in target dir if not reusing processed images
    for image_file in original_dir.iterdir():
        output_file = output_dir / f"{image_file.stem}.png"
        if not reuse and output_file.exists():
            output_file.unlink()

    image_paths = []
    for image_name in tqdm(image_names, desc=f"[>] Processing images", ncols=128):
        image_file = original_dir / image_name

        # Skip if we hit some unexpected directory
        if not image_file.is_file():
            continue

        # Also skip if we're being asked to reuse exiting ones
        output_file = output_dir / f"{image_file.stem}.png"
        if reuse and output_file.exists():
            image_paths.append(str(output_file.resolve()))
            continue

        alpha_file = None if not mask_dir else Path(mask_dir) / f"{image_file.stem}.png"
        image = Image.open(str(image_file))
        alpha = None if not alpha_file else Image.open(str(alpha_file))

        if image.mode == 'RGBA':
            r, g, b, a = image.split()
            image = Image.merge("RGB", (r, g, b))
            alpha = a if alpha is None else alpha # prioritize provided mask over alpha channel

        # Mask GT images before resizing
        if mask_image and alpha is not None:
            image_np = np.array(image)[..., :3].astype(np.float32)
            alpha_np = np.expand_dims(np.array(alpha), axis=-1).astype(np.float32)
            rgb_masked = (image_np / 255.) * (alpha_np / np.max(alpha_np))
            rgb_masked = np.clip(rgb_masked, 0., 1.)
            image = Image.fromarray((rgb_masked * 255.).astype(np.uint8))

        width, height = image.size
        resolution = (width // factor, height // factor)
        if factor > 1:
            image = image.resize(resolution)

        if alpha is not None:
            if factor > 1:
                alpha = alpha.resize(resolution, Image.Resampling.NEAREST)
        else:
            alpha = Image.new("L", resolution, 255)

        image.putalpha(alpha)
        image.save(str(output_file))
        image_paths.append(str(output_file.resolve()))

    return image_paths


def process_input_depths(depth_dir, target_dir, image_names, factor, reuse=False):
    output_dir = Path(target_dir)
    os.makedirs(output_dir, exist_ok=True)

    original_dir = Path(depth_dir)

    # Clean up
    for depth_file in original_dir.iterdir():
        output_file = output_dir / f"{depth_file.stem}.png"
        if not reuse and output_file.exists():
            output_file.unlink()

    depth_paths = []
    for image_name in tqdm(image_names, desc=f"[>] Processing depths", ncols=128):
        depth_file = original_dir / f"{Path(image_name).stem}.png"

        if not depth_file.is_file():
            continue

        output_file = output_dir / f"{depth_file.stem}.png"
        if reuse and output_file.exists():
            depth_paths.append(str(output_file.resolve()))
            continue

        depth = Image.open(str(depth_file))
        depth = depth.convert("I")

        width, height = depth.size
        resolution = (width // factor, height // factor)
        if factor > 1:
            depth = depth.resize(resolution, Image.Resampling.NEAREST)

        depth.save(str(output_file))
        depth_paths.append(str(output_file.resolve()))

    return depth_paths
