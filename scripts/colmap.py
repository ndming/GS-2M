# First install colmap: https://colmap.github.io/install.html

import os
import logging
import shutil

from argparse import ArgumentParser
from tqdm import tqdm

# The conversion is based on the convert.py script from the 3D Gaussian Splatting repo:
# https://github.com/graphdeco-inria/gaussian-splatting/blob/main/convert.py

def match_feature(colmap_exec, src, camera):
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    # Extract feature
    feat_extract = (
        f"{colmap_exec} feature_extractor --database_path {src}/distorted/database.db --image_path {src}/input "
        f"--ImageReader.single_camera 1 --ImageReader.camera_model {camera} --SiftExtraction.use_gpu 1")
    exit_code = os.system(feat_extract)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting...")
        exit(exit_code)

    # Match feature
    feat_match = f"{colmap_exec} exhaustive_matcher --database_path {src}/distorted/database.db --SiftMatching.use_gpu 1"
    exit_code = os.system(feat_match)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting...")
        exit(exit_code)

    # Bundle adjustment
    mapper_cmd = (
        f"{colmap_exec} mapper --database_path {src}/distorted/database.db --image_path {src}/input "
        f"--output_path {src}/distorted/sparse --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting...")
        exit(exit_code)


def undistort_image(colmap_exec, src):
    """Undistort input images to ideal pinhole intrinsics."""

    undistort_cmd = (
        f"{colmap_exec} image_undistorter --image_path {src}/input --input_path {src}/distorted/sparse/0 "
        f"--output_path {src} --output_type COLMAP")
    exit_code = os.system(undistort_cmd)
    if exit_code != 0:
        logging.error(f"Image undistortion failed with code {exit_code}. Exiting...")
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
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)

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
                logging.error(f"50% resize failed with code {exit_code}. Exiting...")
                exit(exit_code)

            dst_file = os.path.join(args.source_path, "images_4", file)
            shutil.copy2(src_file, dst_file)
            exit_code = os.system(f"{magick_exec} mogrify -resize 25% {dst_file}")
            if exit_code != 0:
                logging.error(f"25% resize failed with code {exit_code}. Exiting...")
                exit(exit_code)

            dst_file = os.path.join(args.source_path, "images_8", file)
            shutil.copy2(src_file, dst_file)
            exit_code = os.system(f"{magick_exec} mogrify -resize 12.5% {dst_file}")
            if exit_code != 0:
                logging.error(f"12.5% resize failed with code {exit_code}. Exiting...")
                exit(exit_code)

            pbar.update(1)


if __name__ == "__main__":
    parser = ArgumentParser("COLMAP converter")
    parser.add_argument("--skip_matching", action='store_true')
    parser.add_argument("--source-path", "-s", required=True, type=str)
    parser.add_argument("--camera", default="OPENCV", type=str)
    parser.add_argument("--colmap-executable", default="", type=str)
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--magick-executable", default="", type=str)
    args = parser.parse_args()

    # Get colmap and magick executable paths
    colmap_exec = f'"{args.colmap_executable}"' if len(args.colmap_executable) > 0 else "colmap"
    magick_exec = f'"{args.magick_executable}"' if len(args.magick_executable) > 0 else "magick"
    
    if not args.skip_matching:
        match_feature(colmap_exec, args.source_path, args.camera)

    undistort_image(colmap_exec, args.source_path)

    if(args.resize):
        resize_image(magick_exec, args.source_path)

    print(f"Done extracting images to {args.source_path}")
