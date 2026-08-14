import json
import os

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path


SCANS = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]

MAX_STEPS = 30000


def run(base_dir, out_dir):
    psnr = 0.0
    ssim = 0.0
    pips = 0.0
    time = 0.0  # seconds
    vram = 0.0  # GB

    cd = 0.0
    scene_count = 0

    for scan in SCANS:
        scene_dir = base_dir / f"scan{scan}"
        scene = scene_dir.name
        result_dir = out_dir / scene

        # Training configs
        opt = f"--data-factor 2 --depth-render-mode PD --save-steps {MAX_STEPS} --eval-steps {MAX_STEPS}"
        etc = f"--disable-viewer --no-disable-video --traj-interp-factor 8"
        den = f"--strategy.absgrad --strategy.grow-grad2d 0.0008 --strategy.grow_scale3d 0.001"
        reg = f"--planar-reg 100.0 --multi-view-max-num-samples 102400 --multi-view-trim"
        Ldn = f"--depth-normal-lambda 0.015 --depth-normal-loss-edge-aware"  # from step 7k
        Lmv = f"--multi-view-ncc-lambda 0.15 --multi-view-geo-lambda 0.03"   # from step 7k

        # Train
        cmd = f"python train.py adc --data-dir {scene_dir} --result-dir {result_dir} {opt} {etc} {den} {reg} {Ldn} {Lmv}"
        print("=" * len(scene))
        print(f"{scene}")
        print("=" * len(scene))
        if os.system(cmd) != 0:
            print(f"\n>>> Error occurred for scene {scene} <<<\n")
            continue

        # Extract mesh
        opt = f"--extraction.max-depth 8.0 --extraction.voxel-size 0.002 --extraction.trunc-voxels 4.0 --num-clusters 1"
        etc = f"--render-traj --render-traj-depth-cutoff-factor 2.5"  # render qualitative visualization
        cmd = f"python mesh.py tsdf --cfg-file {result_dir / 'cfg.yml'} {opt} {etc} --extraction.backend vbg"
        if os.system(cmd) != 0:
            print(f"\n>>> Error occurred for scene {scene} <<<\n")
            continue

        # Evaluate mesh
        mesh_file = result_dir / "mesh" / f"tsdf_step{MAX_STEPS - 1}.ply"
        dtu = base_dir / "Official_DTU_Dataset"
        cmd = f"python scripts/eval_dtu/evaluate_single_scene.py --input_ply {mesh_file} --ref_dir {scene_dir} --dtu_dir {dtu}"
        ret = os.system(cmd)
        if ret != 0:
            print(f"\n>>> Error occurred for scene {scene} (mesh evaluation) <<<\n")
            continue

        # Quantitative results
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
        vram += runtime["mem_gb"]
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

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_data[run_id] = {
        "total_scenes": len(SCANS),
        "completed_scenes": scene_count,
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
    parser.add_argument("-o", "--out_dir", type=str, default="output/dtu", help="Directory to store all scene outputs")
    args = parser.parse_args()

    base_dir = Path(args.data_base_dir).resolve()
    out_dir  = Path(args.out_dir).resolve()

    if not base_dir.exists():
        print(f"[!] Could NOT find dataset directory: {base_dir}")
        exit(1)

    for scan in SCANS:
        if not (base_dir / f"scan{scan}").exists():
            print(f"[!] Could NOT find scene directory: {base_dir / f'scan{scan}'}")
            exit(1)

    run(base_dir, out_dir)
