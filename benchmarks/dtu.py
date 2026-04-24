import json
import os

from argparse import ArgumentParser
from pathlib import Path


SCENES = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
MAX_STEPS = 30000


def run(base_dir, out_dir, strategy, postfix):
    psnr = 0.0
    ssim = 0.0
    pips = 0.0
    time = 0.0  # seconds
    vram = 0.0  # GB

    cd = 0.0
    scene_count = 0

    for scene in SCENES:
        scene_dir = base_dir / f"scan{scene}"
        out_dir_name = f"scan{scene}" if postfix == "" else f"scan{scene}_{postfix}"
        result_dir = out_dir / out_dir_name

        # Train
        opt = f"--data-factor 2 --save-steps {MAX_STEPS} --save-ply --eval-steps {MAX_STEPS} --depth-render-mode plane"
        etc = f"--disable-viewer --disable-video --no-normalize-world-space --traj-num-interps 8"
        reg = f"--planar-reg 100 --depth-normal-lambda 0.015"
        if strategy == "mcmc":
            opt = f"{opt} --scale-reg 0.01 --opacity-reg 0.01 --init-opa 0.5 --init-scale 0.1"

        cmd = f"python train.py {strategy} --data-dir {scene_dir} --result-dir {result_dir} {opt} {reg} {etc}"
        print("=" * (len(f"scan{scene}")))
        print(f"scan{scene}")
        print("=" * (len(f"scan{scene}")))
        ret = os.system(cmd)
        if ret != 0:
            print(f"\n>>> Error occurred for scan{scene} (training) <<<\n")
            continue

        # Extract mesh
        opt = f"--extraction.max-depth 5.0 --extraction.voxel-size 0.002 --num-clusters 1 --depth-cutoff-factor 2.5"
        cmd = f"python mesh.py tsdf_single --cfg-file {result_dir / 'cfg.yml'} {opt}"
        ret = os.system(cmd)
        if ret != 0:
            print(f"\n>>> Error occurred for scan{scene} (mesh extraction) <<<\n")
            continue

        # Evaluate mesh
        mesh_file = result_dir / "mesh" / f"tsdf_single_step{MAX_STEPS - 1}.ply"
        dtu = base_dir / "Official_DTU_Dataset"
        cmd = f"python scripts/eval_dtu/evaluate_single_scene.py --input_ply {mesh_file} --ref_dir {scene_dir} --dtu_dir {dtu}"
        ret = os.system(cmd)
        if ret != 0:
            print(f"\n>>> Error occurred for scan{scene} (mesh evaluation) <<<\n")
            continue

        with open(result_dir / "stats" / f"train_{MAX_STEPS}_metrics.json", 'r') as f:
            metrics = json.load(f)
        with open(result_dir / "stats" / f"train_{MAX_STEPS}_rank0.json", 'r') as f:
            runtime = json.load(f)
        with open(result_dir / "mesh" / "results.json", 'r') as f:
            chamfer = json.load(f)

        psnr += metrics["psnr"]
        ssim += metrics["ssim"]
        pips += metrics["lpips"]
        time += runtime["ellapsed_time"]
        vram += runtime["mem"]
        cd   += chamfer["overall"]
        scene_count += 1

    if scene_count == 0:
        print(f"Failed to run benchmark for every scene")
        exit(1)

    avg_psnr = psnr / scene_count
    avg_ssim = ssim / scene_count
    avg_pips = pips / scene_count
    avg_time = time / scene_count
    avg_vram = vram / scene_count
    avg_cd   = cd   / scene_count

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
        "chamfer": avg_cd,
    }

    with open(stats_file, 'w') as f:
        json.dump(stats_data, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_base_dir", type=str, required=True, help="Directory containig all scenes")
    parser.add_argument("--densification", type=str, default="default", help="Strategy to use for densification")
    parser.add_argument("-o", "--out_dir", type=str, default="output/dtu", help="Where store all outputs")
    parser.add_argument("-p", "--postfix", type=str, default="", help="Postfix label for the run (e.g. mcmc)")
    args = parser.parse_args()

    base_dir = Path(args.data_base_dir).resolve()
    out_dir  = Path(args.out_dir).resolve()

    if not base_dir.exists():
        print(f"[!] Could NOT find dataset directory: {base_dir}")
        exit(1)

    for scene_dir in base_dir.iterdir():
        if not scene_dir.is_dir() or scene_dir.name == "Official_DTU_Dataset":
            continue
        if not scene_dir.name in [f"scan{scene}" for scene in SCENES]:
            print(f"[!] Unrecognized scene dir: {scene_dir.name} (expected one of {SCENES})")
            exit(1)

    supported_strategies = ["default", "mcmc"]
    strategy = args.densification
    if not strategy in supported_strategies:
        print(f"[!] Unsupported strategy: {strategy} (expected one of {supported_strategies})")
        exit(1)

    run(base_dir, out_dir, strategy, args.postfix)
