from __future__ import annotations

import os
import math
import numpy as np
import piexif  # type: ignore
import tempfile

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Optional


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
    alpha = None if not alpha_file else Image.open(str(alpha_file)).convert("L")

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
        alpha_max = np.max(alpha_np)
        alpha_norm = alpha_np / alpha_max if alpha_max > 0 else np.zeros_like(alpha_np)
        rgb_masked = (image_np / 255.0) * alpha_norm
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


def _process_single_normal(image_name, original_dir, output_dir, factor, reuse):
    original_dir = Path(original_dir)
    output_dir = Path(output_dir)
    normal_file = original_dir / f"{Path(image_name).stem}.png"

    if normal_file.is_dir():
        return None
    if not normal_file.exists():
        raise FileNotFoundError(f"[!] Normal file not found: {normal_file}")

    output_file = output_dir / f"{normal_file.stem}.png"
    if reuse and output_file.exists():
        return str(output_file.resolve())

    normal = Image.open(str(normal_file))
    if normal.mode == "RGBA":
        r, g, b, _ = normal.split()
        normal = Image.merge("RGB", (r, g, b))
    else:
        normal = normal.convert("RGB")

    width, height = normal.size
    resolution = (width // factor, height // factor)
    if factor > 1:
        # Bilinear on the RGB-encoded direction; renormalized to unit at load time
        normal = normal.resize(resolution, Image.Resampling.BILINEAR)

    _atomic_save(normal, output_file)
    return str(output_file.resolve())


def _process_single_depth(
    image_name, original_dir, output_dir, factor, reuse, mask_image, masked_image_dir,
):
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

    # Mirror the masking applied to the GT image: the processed image carries
    # the baked-in alpha channel, so zero out depths wherever the GT was masked.
    if mask_image and masked_image_dir is not None:
        image_file = Path(masked_image_dir) / f"{Path(image_name).stem}.png"
        processed = Image.open(str(image_file))
        if processed.mode == "RGBA":
            alpha = processed.split()[-1]
            if alpha.size != depth.size:
                alpha = alpha.resize(depth.size, Image.Resampling.NEAREST)
            depth_np = np.array(depth)
            depth_np[np.array(alpha) == 0] = 0
            depth = Image.fromarray(depth_np, mode="I")

    _atomic_save(depth, output_file)
    return str(output_file.resolve())


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
        with tqdm(total=len(image_names), desc="[>] Processing images", ncols=80) as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()  # re-raises any worker exception
                image_paths[idx] = result
                pbar.update(1)

    # Filter out Nones from skipped files while preserving order
    return [p for p in image_paths if p is not None]


def process_input_depths(
    depth_dir, target_dir, image_names, factor,
    reuse=False, mask_image=False, masked_image_dir=None,
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
        mask_image=mask_image,
        masked_image_dir=str(masked_image_dir) if masked_image_dir else None,
    )

    depth_paths = [None] * len(image_names)
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(worker, name): idx for idx, name in enumerate(image_names)}
        with tqdm(total=len(image_names), desc="[>] Processing depths", ncols=80) as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                depth_paths[idx] = result
                pbar.update(1)

    return [p for p in depth_paths if p is not None]


def process_input_normals(
    normal_dir, target_dir, image_names, factor,
    reuse=False,
    num_workers: int | None = None,
):
    output_dir = Path(target_dir)
    os.makedirs(output_dir, exist_ok=True)
    original_dir = Path(normal_dir)

    if not reuse:
        for normal_file in original_dir.iterdir():
            output_file = output_dir / f"{normal_file.stem}.png"
            if output_file.exists():
                output_file.unlink()

    worker = partial(
        _process_single_normal,
        original_dir=str(original_dir),
        output_dir=str(output_dir),
        factor=factor,
        reuse=reuse,
    )

    normal_paths = [None] * len(image_names)
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(worker, name): idx for idx, name in enumerate(image_names)}
        with tqdm(total=len(image_names), desc="[>] Processing normals", ncols=80) as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                normal_paths[idx] = result
                pbar.update(1)

    return [p for p in normal_paths if p is not None]


def similarity_from_cameras(c2w, strict_scaling=False, center_method="focus"):
    """
    reference: nerf-factory
    Get a similarity transform to normalize dataset
    from c2w (OpenCV convention) cameras
    :param c2w: (N, 4)
    :return T (4,4) , scale (float)
    """
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # (1) Rotate the world so that z+ is the up axis
    # we estimate the up axis by averaging the camera up axes
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        # In the unlikely case the original data has y+ up axis,
        # rotate 180-deg about x axis
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    #  R_align = np.eye(3) # DEBUG
    R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]

    # (2) Recenter the scene.
    if center_method == "focus":
        # find the closest point to the origin for each camera's center ray
        nearest = t + (fwds * -t).sum(-1)[:, None] * fwds
        translate = -np.median(nearest, axis=0)
    elif center_method == "poses":
        # use center of the camera positions
        translate = -np.median(t, axis=0)
    else:
        raise ValueError(f"Unknown center_method {center_method}")

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    # (3) Rescale the scene using camera distances
    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))
    transform[:3, :] *= scale

    return transform


def align_principal_axes(point_cloud):
    # Compute centroid
    centroid = np.median(point_cloud, axis=0)

    # Translate point cloud to centroid
    translated_point_cloud = point_cloud - centroid

    # Compute covariance matrix
    covariance_matrix = np.cov(translated_point_cloud, rowvar=False)

    # Compute eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvectors by eigenvalues (descending order) so that the z-axis
    # is the principal axis with the smallest eigenvalue.
    sort_indices = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, sort_indices]

    # Check orientation of eigenvectors. If the determinant of the eigenvectors is
    # negative, then we need to flip the sign of one of the eigenvectors.
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1

    # Create rotation matrix
    rotation_matrix = eigenvectors.T

    # Create SE(3) matrix (4x4 transformation matrix)
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = -rotation_matrix @ centroid

    return transform


def transform_points(matrix, points):
    """Transform points using an SE(3) matrix.

    Args:
        matrix: 4x4 SE(3) matrix
        points: Nx3 array of points

    Returns:
        Nx3 array of transformed points
    """
    assert matrix.shape == (4, 4)
    assert len(points.shape) == 2 and points.shape[1] == 3
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def transform_cameras(matrix, camtoworlds):
    """Transform cameras using an SE(3) matrix.

    Args:
        matrix: 4x4 SE(3) matrix
        camtoworlds: Nx4x4 array of camera-to-world matrices

    Returns:
        Nx4x4 array of transformed camera-to-world matrices
    """
    assert matrix.shape == (4, 4)
    assert len(camtoworlds.shape) == 3 and camtoworlds.shape[1:] == (4, 4)
    camtoworlds = np.einsum("nij, ki -> nkj", camtoworlds, matrix)
    scaling = np.linalg.norm(camtoworlds[:, 0, :3], axis=1)
    camtoworlds[:, :3, :3] = camtoworlds[:, :3, :3] / scaling[:, None, None]
    return camtoworlds


def normalize(camtoworlds, points=None):
    T1 = similarity_from_cameras(camtoworlds)
    camtoworlds = transform_cameras(T1, camtoworlds)
    if points is not None:
        points = transform_points(T1, points)
        T2 = align_principal_axes(points)
        camtoworlds = transform_cameras(T2, camtoworlds)
        points = transform_points(T2, points)
        return camtoworlds, points, T2 @ T1
    else:
        return camtoworlds, T1


def _extract_shutter_time(exif: Dict) -> Optional[float]:
    # EXIF tag IDs (decimal)
    TAG_EXPOSURE_TIME = 33434  # ExposureTime (seconds)
    TAG_SHUTTER_SPEED_VALUE = 37377  # ShutterSpeedValue (APEX Tv)
    exif_ifd = exif.get("Exif") if isinstance(exif.get("Exif"), dict) else {}

    if TAG_EXPOSURE_TIME in exif_ifd:
        num, den = exif_ifd[TAG_EXPOSURE_TIME]
        seconds = num / den
        if seconds > 0.0 and math.isfinite(seconds):
            return seconds

    if TAG_SHUTTER_SPEED_VALUE in exif_ifd:
        num, den = exif_ifd[TAG_SHUTTER_SPEED_VALUE]
        tv = num / den
        seconds = math.pow(2.0, -tv)
        if seconds > 0.0 and math.isfinite(seconds):
            return seconds

    return None


def _extract_aperture_fnumber(exif: Dict) -> Optional[float]:
    # EXIF tag IDs (decimal)
    TAG_FNUMBER = 33437  # FNumber (f-number)
    TAG_APERTURE_VALUE = 37378  # ApertureValue (APEX Av)
    exif_ifd = exif.get("Exif") if isinstance(exif.get("Exif"), dict) else {}

    if TAG_FNUMBER in exif_ifd:
        num, den = exif_ifd[TAG_FNUMBER]
        fnum = num / den
        if fnum > 0.0 and math.isfinite(fnum):
            return fnum

    if TAG_APERTURE_VALUE in exif_ifd:
        num, den = exif_ifd[TAG_APERTURE_VALUE]
        av = num / den
        fnum = math.pow(2.0, av / 2.0)
        if fnum > 0.0 and math.isfinite(fnum):
            return fnum

    return None


def _extract_iso(exif: Dict) -> Optional[float]:
    # EXIF tag IDs (decimal)
    # PhotographicSensitivity / ISOSpeedRatings
    TAG_PHOTOGRAPHIC_SENSITIVITY = 34855
    TAG_STANDARD_OUTPUT_SENSITIVITY = 34857  # StandardOutputSensitivity (SOS)
    TAG_RECOMMENDED_EXPOSURE_INDEX = 34858  # RecommendedExposureIndex (REI)
    TAG_ISO_SPEED = 34859  # ISOSpeed
    exif_ifd = exif.get("Exif") if isinstance(exif.get("Exif"), dict) else {}

    candidates: List[int] = [
        TAG_PHOTOGRAPHIC_SENSITIVITY,
        TAG_RECOMMENDED_EXPOSURE_INDEX,
        TAG_STANDARD_OUTPUT_SENSITIVITY,
        TAG_ISO_SPEED,
    ]

    for tag in candidates:
        if tag in exif_ifd:
            value = float(exif_ifd[tag])
            if value > 0.0 and math.isfinite(value):
                return value

    return None


def compute_exposure_from_exif(path: Path) -> Optional[float]:
    """Return exposure in EV stops (log2 of relative exposure) or None if unavailable.

    Relative exposure is computed as (seconds / f^2 * ISO) then converted via log2.
    Returns None if the file format doesn't support EXIF (e.g., PNG).
    """
    try:
        exif = piexif.load(str(path))
    except piexif.InvalidImageDataError:
        # File format doesn't support EXIF (e.g., PNG)
        return None
    shutter_s = _extract_shutter_time(exif)
    aperture_f = _extract_aperture_fnumber(exif)
    iso_value = _extract_iso(exif)

    # If none of the components are available, we cannot compute exposure
    if shutter_s is None and aperture_f is None and iso_value is None:
        return None

    # Use available components; treat missing ones as 1 for exposure calculation
    seconds = shutter_s if shutter_s is not None else 1.0
    f_number = aperture_f if aperture_f is not None else 1.0
    iso = iso_value if iso_value is not None else 1.0

    rel_exposure = (seconds / (f_number * f_number)) * iso
    if rel_exposure <= 0.0 or not math.isfinite(rel_exposure):
        return None
    return math.log2(rel_exposure)
