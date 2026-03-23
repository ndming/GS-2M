import os

from argparse import ArgumentParser
from pathlib import Path
from PIL import Image

def convert_webp(frame_dir, webp_file):
    frames = [Image.open(frame).convert("RGBA") for frame in sorted(frame_dir.iterdir(), key=lambda p: int(p.stem.split('_')[1])) if frame.is_file()]
    frames[0].save(
        webp_file, save_all=True, append_images=frames[1:], format='WEBP',
        duration=int(1000 / 24), loop=0, transparency=0, disposal=2)
    
def convert_webp_gt(frame_dir, webp_file):
    frames = [Image.open(frame).convert("RGBA") for frame in sorted(frame_dir.iterdir(), key=lambda p: int(p.stem.split('_')[1])) if frame.is_file() and frame.name.endswith("png") and not "normal" in frame.stem]
    frames[0].save(
        webp_file, save_all=True, append_images=frames[1:], format='WEBP',
        duration=int(1000 / 24), loop=0, transparency=0, disposal=2) 

if __name__ == "__main__":
    parser = ArgumentParser(description="Visualize Blender dataset")
    parser.add_argument("--model", "-m", required=True, type=str)
    parser.add_argument("--method", required=True, type=str)
    parser.add_argument("--dataset_dir", "-d", type=str, default="")
    parser.add_argument("--force", "-f", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model).resolve()
    method_dir = model_dir / "test" / args.method
    visual_dir = method_dir / "visual"
    os.makedirs(visual_dir, exist_ok=True)

    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir).resolve()
        gt_file = visual_dir / "gt.webp"
        if not gt_file.exists() or args.force:
            print(f"Converting: {gt_file}")
            gt_dir = dataset_dir / "test"
            convert_webp_gt(gt_dir, gt_file)

    albedo_file = visual_dir / "albedo.webp"
    if not albedo_file.exists() or args.force:
        print(f"Converting: {albedo_file}")
        albedo_dir = method_dir / "albedo"
        convert_webp(albedo_dir, albedo_file)
    else:
        print(f"Skipping: {albedo_file}")

    normal_file = visual_dir / "normal.webp"
    if not normal_file.exists() or args.force:
        print(f"Converting: {normal_file}")
        normal_dir = method_dir / "normal"
        convert_webp(normal_dir, normal_file)
    else:
        print(f"Skipping: {normal_file}")

    roughness_file = visual_dir / "roughness.webp"
    if not roughness_file.exists() or args.force:
        print(f"Converting: {roughness_file}")
        roughness_dir = method_dir / "roughness"
        convert_webp(roughness_dir, roughness_file)
    else:
        print(f"Skipping: {roughness_file}")

    diffuse_file = visual_dir / "diffuse.webp"
    if not diffuse_file.exists() or args.force:
        print(f"Converting: {diffuse_file}")
        diffuse_dir = method_dir / "diffuse"
        convert_webp(diffuse_dir, diffuse_file)
    else:
        print(f"Skipping: {diffuse_file}")

    specular_file = visual_dir / "specular.webp"
    if not specular_file.exists() or args.force:
        print(f"Converting: {specular_file}")
        specular_dir = method_dir / "specular"
        convert_webp(specular_dir, specular_file)
    else:
        print(f"Skipping: {specular_file}")