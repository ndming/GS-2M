import os
from argparse import ArgumentParser
from pathlib import Path


SCENES  = ["garden", "bicycle", "stump", "bonsai", "counter", "kitchen", "room"] # treehill flowers
FACTORS = [       4,         4,       4,        2,         2,         2,      2] # data-factor


def run(base_dir, out_dir, strategy, postfix):
    for scene, factor in zip(SCENES, FACTORS):
        scene_dir = base_dir / scene
        out_dir_name = scene if postfix == "" else f"{scene}_{postfix}"
        result_dir = out_dir / out_dir_name

        # Train without eval
        opt = "--disable-viewer --eval_steps -1"
        cmd = f"python train.py {strategy} --data-dir {scene_dir} --data-factor {factor} --result-dir {result_dir} {opt}"
        print(f"[>] {cmd}")
        ret = os.system(cmd)

        if ret != 0:
            print("[!] Error occured, exiting...")
            exit(1)

        # Run eval by specifying ckpt
        for ckpt_file in (result_dir / "ckpts").iterdir():
            if not ckpt_file.is_file():
                continue
            opt = f"--ckpt {ckpt_file} --disable-viewer --test-every 8 --traj-num-interps 16 --render-traj-path ellipse"
            cmd = f"python train.py {strategy} --data-dir {scene_dir} --data-factor {factor} --result-dir {result_dir} {opt}"
            print(f"[>] {cmd}")
            os.system(cmd)


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
