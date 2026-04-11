# First install colmap: https://colmap.github.io/install.html

import cv2
import os
import shutil

from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

# The conversion is based on the convert.py script from the 3D Gaussian Splatting repo:
# https://github.com/graphdeco-inria/gaussian-splatting/blob/main/convert.py

def extract_features_and_mapping(colmap_exec, src, camera, colmap_feature_extraction, colmap_feature_matching):
    os.makedirs(f"{src}/distorted/sparse", exist_ok=True)

    db_path = f"{src}/distorted/database.db"
    img_path = f"{src}/input"
    out_path = f"{src}/distorted/sparse"

    # Extract feature
    feat_extract = (
        f"{colmap_exec} feature_extractor " 
        f"--database_path {db_path} "
        f"--image_path {img_path} "
        f"--ImageReader.single_camera 1 "
        f"--ImageReader.camera_model {camera} "
        f"--FeatureExtraction.use_gpu 1 "
        f"--FeatureExtraction.type {colmap_feature_extraction} "
        f"--AlikedExtraction.max_num_features 4096"
    )
    exit_code = os.system(feat_extract)
    if exit_code != 0:
        print(f"Feature extraction failed with code {exit_code}. Exiting...")
        exit(exit_code)

    # Match feature
    feat_match = (
        f"{colmap_exec} exhaustive_matcher "
        f"--database_path {db_path} "
        f"--FeatureMatching.use_gpu 1 "
        f"--FeatureMatching.type {colmap_feature_matching} "
    )
    exit_code = os.system(feat_match)
    if exit_code != 0:
        print(f"Feature matching failed with code {exit_code}. Exiting...")
        exit(exit_code)

    # Bundle adjustment
    mapper_cmd = (
        f"{colmap_exec} global_mapper "
        f"--database_path {db_path} "
        f"--image_path {img_path} "
        f"--output_path {out_path} "
        f"--GlobalMapper.gp_use_gpu 1 "
        # f"--GlobalMapper.ba_global_function_tolerance 1e-6"
    )
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        print(f"Global mapper failed with code {exit_code}. Exiting...")
        exit(exit_code)


def undistort_image(colmap_exec, src):
    """Undistort input images to ideal pinhole intrinsics."""

    undistort_cmd = (
        f"{colmap_exec} image_undistorter --image_path {src}/input --input_path {src}/distorted/sparse/0 "
        f"--output_path {src} --output_type COLMAP")
    exit_code = os.system(undistort_cmd)
    if exit_code != 0:
        print(f"Image undistortion failed with code {exit_code}. Exiting...")
        exit(exit_code)

    # Move each file from src to dest
    files = os.listdir(f"{src}/sparse")
    os.makedirs(f"{src}/sparse/0", exist_ok=True)
    for file in files:
        if file == '0':
            continue
        src_file = os.path.join(src, "sparse", file)
        dst_file = os.path.join(src, "sparse", "0", file)
        shutil.move(src_file, dst_file)


def resize_image(magick_exec, src):
    os.makedirs(src + "/images_2", exist_ok=True)
    os.makedirs(src + "/images_4", exist_ok=True)
    os.makedirs(src + "/images_8", exist_ok=True)

    # Get the list of files in the source directory
    files = os.listdir(f"{src}/images")

    # Copy each file from the source directory to the destination directory
    with tqdm(total=len(files), desc="Copying and resizing") as pbar:
        for file in files:
            src_file = os.path.join(src, "images", file)

            dst_file = os.path.join(src, "images_2", file)
            shutil.copy2(src_file, dst_file)
            exit_code = os.system(f"{magick_exec} mogrify -resize 50% {dst_file}")
            if exit_code != 0:
                print(f"50% resize failed with code {exit_code}. Exiting...")
                exit(exit_code)

            dst_file = os.path.join(args.source_path, "images_4", file)
            shutil.copy2(src_file, dst_file)
            exit_code = os.system(f"{magick_exec} mogrify -resize 25% {dst_file}")
            if exit_code != 0:
                print(f"25% resize failed with code {exit_code}. Exiting...")
                exit(exit_code)

            dst_file = os.path.join(args.source_path, "images_8", file)
            shutil.copy2(src_file, dst_file)
            exit_code = os.system(f"{magick_exec} mogrify -resize 12.5% {dst_file}")
            if exit_code != 0:
                print(f"12.5% resize failed with code {exit_code}. Exiting...")
                exit(exit_code)

            pbar.update(1)


def sample_from_image_dir(sample_dir: Path, interval, output_dir):
    images = sorted([p for p in sample_dir.iterdir() if p.is_file()])
    assert len(images) > 0, "Empty sample dir"

    print(f"Found {len(images)} images, sampling every {interval} frames")
    sampled = images[::interval]
    for sample in sampled:
        shutil.copy2(sample, output_dir / sample.name)


def sample_from_video_file(video_file, interval, output_dir):
    cap = cv2.VideoCapture(video_file)
    frame_count  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = frame_count // args.interval

    print(f"Found {len(frame_count)}, sampling every {interval} frames")
    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        frame_count   = 0
        extract_count = 0
        
        while True:
            ret, frame = cap.read()

            if not ret:
                break  # reached the end of the video

            if frame_count % args.interval == 0:
                frame_filename = output_dir / f"{frame_count:05d}.png"
                cv2.imwrite(frame_filename, frame)

                extract_count += 1
                pbar.update(1)

            frame_count += 1
    cap.release()


if __name__ == "__main__":
    parser = ArgumentParser("COLMAP 4.0 converter")
    parser.add_argument("--sample_from", default="", type=str)
    parser.add_argument("--sample_interval", default=1, type=int)
    parser.add_argument("--sample_overwrite", action='store_true')
    parser.add_argument("--skip_matching", action='store_true')
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--camera", default="OPENCV", type=str)
    parser.add_argument("--colmap_executable", default="", type=str)
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--magick_executable", default="", type=str)
    parser.add_argument(
        "--colmap_feature_extraction", type=str, default="SIFT",
        choices=["SIFT", "ALIKED_N16ROT", "ALIKED_N32"],
    )
    parser.add_argument(
        "--colmap_feature_matching", type=str, default="SIFT_BRUTEFORCE",
        choices=["SIFT_BRUTEFORCE", "SIFT_LIGHTGLUE", "ALIKED_BRUTEFORCE", "ALIKED_LIGHTGLUE"],
    )
    args = parser.parse_args()

    if args.sample_from != "":
        sample_target = Path(args.sample_from)
        print(f"Sampling frames from: {sample_target}")
        assert args.sample_interval > 0, f"Negative sameple interval: {args.sample_interval}"
        
        source_dir = Path(args.source_path)
        os.makedirs(source_dir, exist_ok=True)

        input_dir = source_dir / "input"
        if input_dir.exists() and any(input_dir.iterdir()) and not args.sample_overwrite:
            print(f"Warning: found assets under {input_dir}, please remove them or run with --sample_overwrite, exiting...")
            exit(1)
        os.makedirs(input_dir, exist_ok=True)
        for file in input_dir.iterdir():
            file.unlink()

        if sample_target.is_dir():
            sample_from_image_dir(sample_target, args.sample_interval, input_dir)
        elif sample_target.is_file():
            sample_from_video_file(sample_target, args.sample_interval, input_dir)

    # Get colmap and magick executable paths
    colmap_exec = f'"{args.colmap_executable}"' if len(args.colmap_executable) > 0 else "colmap"
    magick_exec = f'"{args.magick_executable}"' if len(args.magick_executable) > 0 else "magick"
    
    if not args.skip_matching:
        extract_features_and_mapping(
            colmap_exec, args.source_path, args.camera,
            args.colmap_feature_extraction, args.colmap_feature_matching)

    undistort_image(colmap_exec, args.source_path)

    if(args.resize):
        resize_image(magick_exec, args.source_path)

    print(f"Done extracting images to {args.source_path}")
