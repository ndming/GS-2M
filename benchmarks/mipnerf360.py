import json
import os

from argparse import ArgumentParser
from pathlib import Path


SCENES    = ["garden", "bicycle", "stump", "bonsai", "counter", "kitchen", "room"]  # treehill flowers
FACTORS   = [       4,         4,       4,        2,         2,         2,      2]
MAX_STEPS = 30000


def run(base_dir, out_dir, strategy, postfix):
    psnr = 0.0
    ssim = 0.0
    pips = 0.0
    time = 0.0  # seconds
    vram = 0.0  # GB

    scene_count = 0
    for scene, factor in zip(SCENES, FACTORS):
        scene_dir = base_dir / scene
        out_dir_name = scene if postfix == "" else f"{scene}_{postfix}"
        result_dir = out_dir / out_dir_name

        opt = f"--data-factor {factor} --save-steps {MAX_STEPS} --save-ply --eval-steps {MAX_STEPS} --test-every 8"
        etc = f"--disable-viewer --disable-video --normalize-world-space"
        if strategy == "mcmc":
            opt = f"{opt} --scale-reg 0.01 --opacity-reg 0.01 --init-opa 0.5 --init-scale 0.1"

        cmd = f"python train.py {strategy} --data-dir {scene_dir} --result-dir {result_dir} {opt} {etc}"
        print("=" * len(scene))
        print(f"{scene}")
        print("=" * len(scene))
        ret = os.system(cmd)

        if ret != 0:
            print(f"\n>>> Error occurred for scene {scene} <<<\n")
            continue

        with open(result_dir / "stats" / f"val_{MAX_STEPS}_metrics.json", 'r') as f:
            metrics = json.load(f)
        with open(result_dir / "stats" / f"train_{MAX_STEPS}_rank0.json", 'r') as f:
            runtime = json.load(f)

        psnr += metrics["psnr"]
        ssim += metrics["ssim"]
        pips += metrics["lpips"]
        time += runtime["ellapsed_time"]
        vram += runtime["mem"]
        scene_count += 1

    if scene_count == 0:
        print(f"Failed to run benchmark for every scene")
        exit(1)

    avg_psnr = psnr / scene_count
    avg_ssim = ssim / scene_count
    avg_pips = pips / scene_count
    avg_time = time / scene_count
    avg_vram = vram / scene_count

    stats_file = Path(out_dir) / "stats.json"
    stats_data = {}
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats_data = json.load(f)

    key = strategy if postfix == "" else postfix
    stats_data[key] = {
        "psnr": avg_psnr,
        "ssim": avg_ssim,
        "lpips": avg_pips,
        "time_mins": avg_time / 60,
        "vram_gb": avg_vram,
    }

    with open(stats_file, 'w') as f:
        json.dump(stats_data, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_base_dir", type=str, required=True, help="Directory containig all scenes")
    parser.add_argument("--densification", type=str, default="default", help="Strategy to use for densification")
    parser.add_argument("-o", "--out_dir", type=str, default="output/mipnerf360", help="Where store all outputs")
    parser.add_argument("-p", "--postfix", type=str, default="", help="Postfix label for the run (e.g. mcmc)")
    args = parser.parse_args()

    base_dir = Path(args.data_base_dir).resolve()
    out_dir  = Path(args.out_dir).resolve()

    if not base_dir.exists():
        print(f"[!] Could NOT find dataset directory: {base_dir}")
        exit(1)

    for scene_dir in base_dir.iterdir():
        if not scene_dir.is_dir():
            continue
        if not scene_dir.name in SCENES:
            print(f"[!] Unrecognized scene dir: {scene_dir.name} (expected one of {SCENES})")
            exit(1)

    supported_strategies = ["default", "mcmc"]
    strategy = args.densification
    if not strategy in supported_strategies:
        print(f"[!] Unsupported strategy: {strategy} (expected one of {supported_strategies})")
        exit(1)

    run(base_dir, out_dir, strategy, args.postfix)
