import cv2
import os
import shutil
import tyro

from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from typing_extensions import Literal


@dataclass
class Args:
    # Path to the scene directory where the input directory is located
    source_path: str
    # Optional path to sample frames from to populate the input directory, can be a video file or an image directory
    sample_from: str = ""
    # Interval for sampling frames from the sample_from source, e.g. 5 means sample every 5 frames
    sample_interval: int = 1
    # Whether to overwrite existing files in the input directory when sampling frames from the sample_from source
    sample_overwrite: bool = False
    # Whether to skip feature extraction and matching steps, useful if the sparse directory already exists
    # and you just want to undistort images and convert the sparse model to COLMAP format
    skip_matching: bool = False
    # Camera model for COLMAP, e.g. OPENCV, PINHOLE, etc.
    camera: str = "OPENCV"
    # Camera parameters for COLMAP, comma separated, e.g. fx,fy,cx,cy,k1,k2,p1,p2
    params: str = ""
    # Path to the COLMAP executable, if not in PATH
    colmap_executable: str = ""
    # Whether to resize images to create multi-scale inputs
    resize: bool = False
    # Path to the ImageMagick executable, if not in PATH, used for resizing images
    magick_executable: str = ""
    # Feature extraction method for COLMAP: SIFT or ALIKED variants
    feature_extraction: Literal["SIFT", "ALIKED_N16ROT", "ALIKED_N32"] = "SIFT"
    # Max number of features for ALIKED extractor, only used if feature_extraction is set to an ALIKED variant
    max_aliked_features: int = 4096
    # Feature matching method for COLMAP, traditional brute-force or the newer LightGlue
    feature_matching: Literal["BRUTEFORCE", "LIGHTGLUE"] = "BRUTEFORCE"
    # Matcher type for COLMAP
    matcher: Literal["exhaustive", "sequential", "vocab_tree"] = "exhaustive"
    # Overlap for sequential matcher, only used if matcher is set to sequential
    sequential_overlap: int = 20
    # Whether to use quadratic overlap for sequential matcher, only used if matcher is set to sequential
    sequential_quadratic_overlap: bool = True
    # Whether to use loop detection for sequential matcher, only used if matcher is set to sequential
    sequential_loop_detection: bool = True
    # Whether to use the global mapper instead of the default COLMAP mapper, which can be more robust for large scenes
    use_global_mapper: bool = False


def extract_features_and_mapping(colmap_exec, args: Args):
    src = args.source_path
    camera = args.camera # camera model
    params = args.params # camera parameters
    feature_extraction = args.feature_extraction
    fm_prefix = "ALIKED" if "ALIKED" in feature_extraction else "SIFT"
    feature_matching = f"{fm_prefix}_{args.feature_matching}"

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
        f"--FeatureExtraction.type {feature_extraction} "
        f"--AlikedExtraction.max_num_features {args.max_aliked_features} "
    )
    if params != "":
        feat_extract += f"--ImageReader.camera_params {params}"
    exit_code = os.system(feat_extract)
    if exit_code != 0:
        print(f"[!] Feature extraction failed with code {exit_code}. Exiting...")
        exit(exit_code)

    # Match feature
    if args.matcher == "exhaustive":
        feat_match = (
            f"{colmap_exec} exhaustive_matcher "
            f"--database_path {db_path} "
            f"--FeatureMatching.use_gpu 1 "
            f"--FeatureMatching.type {feature_matching} "
        )
    elif args.matcher == "sequential":
        feat_match = (
            f"{colmap_exec} sequential_matcher "
            f"--database_path {db_path} "
            f"--FeatureMatching.use_gpu 1 "
            f"--FeatureMatching.type {feature_matching} "
            f"--SequentialMatching.overlap {args.sequential_overlap} "
            f"--SequentialMatching.quadratic_overlap {1 if args.sequential_quadratic_overlap else 0} "
            f"--SequentialMatching.loop_detection {1 if args.sequential_loop_detection else 0}"
        )
    elif args.matcher == "vocab_tree":
        feat_match = (
            f"{colmap_exec} vocab_tree_matcher "
            f"--database_path {db_path} "
            f"--FeatureMatching.use_gpu 1 "
            f"--FeatureMatching.type {feature_matching} "
        )
    else:
        raise ValueError(f"Unknown matcher type: {args.matcher}")
    exit_code = os.system(feat_match)
    if exit_code != 0:
        print(f"[!] Feature matching failed with code {exit_code}. Exiting...")
        exit(exit_code)

    # Bundle adjustment
    if args.use_global_mapper:
        mapper_cmd = (
            f"{colmap_exec} global_mapper "
            f"--database_path {db_path} "
            f"--image_path {img_path} "
            f"--output_path {out_path} "
            f"--GlobalMapper.gp_use_gpu 1"
        )
    else:
        mapper_cmd = (
            f"{colmap_exec} mapper "
            f"--database_path {db_path} "
            f"--image_path {img_path} "
            f"--output_path {out_path} "
            f"--Mapper.ba_use_gpu 1 "
            f"--Mapper.ba_global_function_tolerance 1e-6"
        )
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        print(f"[!] Global mapper failed with code {exit_code}. Exiting...")
        exit(exit_code)


def undistort_image(colmap_exec, src):
    """Undistort input images to ideal pinhole intrinsics."""

    undistort_cmd = (
        f"{colmap_exec} image_undistorter --image_path {src}/input --input_path {src}/distorted/sparse/0 "
        f"--output_path {src} --output_type COLMAP")
    exit_code = os.system(undistort_cmd)
    if exit_code != 0:
        print(f"[!] Image undistortion failed with code {exit_code}. Exiting...")
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
                print(f"[!] 50% resize failed with code {exit_code}. Exiting...")
                exit(exit_code)

            dst_file = os.path.join(args.source_path, "images_4", file)
            shutil.copy2(src_file, dst_file)
            exit_code = os.system(f"{magick_exec} mogrify -resize 25% {dst_file}")
            if exit_code != 0:
                print(f"[!] 25% resize failed with code {exit_code}. Exiting...")
                exit(exit_code)

            dst_file = os.path.join(args.source_path, "images_8", file)
            shutil.copy2(src_file, dst_file)
            exit_code = os.system(f"{magick_exec} mogrify -resize 12.5% {dst_file}")
            if exit_code != 0:
                print(f"[!] 12.5% resize failed with code {exit_code}. Exiting...")
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
    total_frames = frame_count // interval

    print(f"Found {len(frame_count)}, sampling every {interval} frames")
    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        frame_count   = 0
        extract_count = 0
        
        while True:
            ret, frame = cap.read()

            if not ret:
                break  # reached the end of the video

            if frame_count % interval == 0:
                frame_filename = output_dir / f"{frame_count:05d}.png"
                cv2.imwrite(frame_filename, frame)

                extract_count += 1
                pbar.update(1)

            frame_count += 1
    cap.release()


def main(args: Args):
    colmap_exec = f'"{args.colmap_executable}"' if args.colmap_executable != "" else "colmap"
    magick_exec = f'"{args.magick_executable}"' if args.magick_executable != "" else "magick"

    if args.sample_from != "":
        sample_target = Path(args.sample_from)
        print(f"[>] Sampling frames from: {sample_target}")
        assert args.sample_interval > 0, f"Negative sameple interval: {args.sample_interval}"
        
        source_dir = Path(args.source_path)
        os.makedirs(source_dir, exist_ok=True)

        input_dir = source_dir / "input"
        if input_dir.exists() and any(input_dir.iterdir()) and not args.sample_overwrite:
            print(f"[!] Warning: found assets under {input_dir}, please remove them or run with --sample_overwrite, exiting...")
            exit(1)
        os.makedirs(input_dir, exist_ok=True)
        for file in input_dir.iterdir():
            file.unlink()

        if sample_target.is_dir():
            sample_from_image_dir(sample_target, args.sample_interval, input_dir)
        elif sample_target.is_file():
            sample_from_video_file(sample_target, args.sample_interval, input_dir)

    if not args.skip_matching:
        extract_features_and_mapping(colmap_exec, args)

    undistort_image(colmap_exec, args.source_path)

    if(args.resize):
        resize_image(magick_exec, args.source_path)

    print(f"[>] Done extracting images to {args.source_path}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
