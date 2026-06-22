import os
import numpy as np
import tempfile

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def _atomic_save(image: Image.Image, output_file: Path) -> None:
    """Write to a sibling temp file then rename — avoids torn reads by DataLoader workers."""
    tmp_fd, tmp_path = tempfile.mkstemp(dir=output_file.parent, suffix=".tmp.png")
    try:
        os.close(tmp_fd)
        image.save(tmp_path)
        os.replace(tmp_path, output_file)   # atomic on POSIX; best-effort on Windows
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ── helpers (must be module-level for pickle-ability in ProcessPoolExecutor) ──

def _process_single_image(
    image_name, original_dir, output_dir,
    factor, reuse, mask_image, mask_dir, exposure_bias,
):
    original_dir = Path(original_dir)
    output_dir = Path(output_dir)
    image_file = original_dir / image_name

    if image_file.is_dir():
        return None  # silently skip directories, same as original behaviour
    if not image_file.exists():
        raise FileNotFoundError(f"[!] Image file not found: {image_file}")

    output_file = output_dir / f"{image_file.stem}.png"
    if reuse and output_file.exists():
        return str(output_file.resolve())

    alpha_file = None if not mask_dir else Path(mask_dir) / f"{image_file.stem}.png"
    image = Image.open(str(image_file))
    alpha = None if not alpha_file else Image.open(str(alpha_file))

    if image.mode == "RGBA":
        r, g, b, a = image.split()
        image = Image.merge("RGB", (r, g, b))
        alpha = a if alpha is None else alpha

    if exposure_bias != 0.0:
        gain = 2.0 ** exposure_bias
        image_np = np.array(image)[..., :3].astype(np.float32) / 255.0
        image_np = image_np ** 2.2
        image_np *= gain
        image_np = np.clip(image_np, 0.0, 1.0)
        image_np = image_np ** (1.0 / 2.2)
        image = Image.fromarray((image_np * 255.0).astype(np.uint8))

    if mask_image and alpha is not None:
        image_np = np.array(image)[..., :3].astype(np.float32)
        alpha_np = np.expand_dims(np.array(alpha), axis=-1).astype(np.float32)
        rgb_masked = (image_np / 255.0) * (alpha_np / np.max(alpha_np))
        rgb_masked = np.clip(rgb_masked, 0.0, 1.0)
        image = Image.fromarray((rgb_masked * 255.0).astype(np.uint8))

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
    _atomic_save(image, output_file)
    return str(output_file.resolve())


def _process_single_depth(image_name, original_dir, output_dir, factor, reuse):
    original_dir = Path(original_dir)
    output_dir = Path(output_dir)
    depth_file = original_dir / f"{Path(image_name).stem}.png"

    if depth_file.is_dir():
        return None
    if not depth_file.exists():
        raise FileNotFoundError(f"[!] Depth file not found: {depth_file}")

    output_file = output_dir / f"{depth_file.stem}.png"
    if reuse and output_file.exists():
        return str(output_file.resolve())

    depth = Image.open(str(depth_file))
    depth = depth.convert("I")

    width, height = depth.size
    resolution = (width // factor, height // factor)
    if factor > 1:
        depth = depth.resize(resolution, Image.Resampling.NEAREST)

    _atomic_save(depth, output_file)
    return str(output_file.resolve())


# ── public API ────────────────────────────────────────────────────────────────

def process_input_images(
    image_dir, target_dir, image_names, factor,
    reuse=False, mask_image=False, mask_dir=None,
    exposure_bias=0.0,
    num_workers: int | None = None,   # None → os.cpu_count()
):
    output_dir = Path(target_dir)
    os.makedirs(output_dir, exist_ok=True)
    original_dir = Path(image_dir)

    if factor > 1:
        print(f"[>] Using reference images downscaled by {factor}x in {output_dir}")

    # Remove stale outputs up-front (single-threaded, fast)
    if not reuse:
        for image_file in original_dir.iterdir():
            output_file = output_dir / f"{image_file.stem}.png"
            if output_file.exists():
                output_file.unlink()

    worker = partial(
        _process_single_image,
        original_dir=str(original_dir),
        output_dir=str(output_dir),
        factor=factor,
        reuse=reuse,
        mask_image=mask_image,
        mask_dir=str(mask_dir) if mask_dir else None,
        exposure_bias=exposure_bias,
    )

    # ProcessPoolExecutor bypasses the GIL for numpy / Pillow CPU work
    image_paths = [None] * len(image_names)
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(worker, name): idx for idx, name in enumerate(image_names)}
        with tqdm(total=len(image_names), desc="[>] Processing images", ncols=128) as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()  # re-raises any worker exception
                image_paths[idx] = result
                pbar.update(1)

    # Filter out Nones from skipped files while preserving order
    return [p for p in image_paths if p is not None]


def process_input_depths(
    depth_dir, target_dir, image_names, factor,
    reuse=False,
    num_workers: int | None = None,
):
    output_dir = Path(target_dir)
    os.makedirs(output_dir, exist_ok=True)
    original_dir = Path(depth_dir)

    if not reuse:
        for depth_file in original_dir.iterdir():
            output_file = output_dir / f"{depth_file.stem}.png"
            if output_file.exists():
                output_file.unlink()

    worker = partial(
        _process_single_depth,
        original_dir=str(original_dir),
        output_dir=str(output_dir),
        factor=factor,
        reuse=reuse,
    )

    depth_paths = [None] * len(image_names)
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(worker, name): idx for idx, name in enumerate(image_names)}
        with tqdm(total=len(image_names), desc="[>] Processing depths", ncols=128) as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                depth_paths[idx] = result
                pbar.update(1)

    return [p for p in depth_paths if p is not None]
