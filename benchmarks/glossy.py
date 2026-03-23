import json
import os
import time

from pathlib import Path
from argparse import ArgumentParser

scenes = ["angel", "bell", "cat", "horse", "luyu", "teapot", "potion", "tbell"]
data_base_path = '/home/zodnguy1/datasets/glossy'
out_base_path='output/glossy'

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--prep", action="store_true")
    args = parser.parse_args()

    runtime_file = Path(out_base_path) / 'runtime.json'
    runtime_data = {}
    if runtime_file.exists():
        with open(runtime_file, 'r') as f:
            runtime_data = json.load(f)

    runtimes = []
    label = "ours"

    for scene in scenes:
        if args.prep or not Path(f"{data_base_path}/{scene}_blender").exists():
            print(f"[>] Preparing scene: {scene}")
            cmd = f"python scripts/preprocess/nero2blender.py --path {data_base_path} --scene {scene}"
            print(f"[>] {cmd}")
            os.system(cmd)

        scene_start = time.time()

        common_args = f"--mask_gt --material --eval --white_background --reflection_threshold 0.2 --lambda_smooth 0.5 --lambda_normal 0.5 --iterations 10000"
        cmd = f"python train.py -s {data_base_path}/{scene}_blender -m {out_base_path}/{scene} {common_args}"
        print(f"[>] {cmd}")
        os.system(cmd)

        common_args = f"--blender --iteration 10000 --label {label}"
        cmd = f"python render.py -m {out_base_path}/{scene} {common_args}"
        print(f"[>] {cmd}")
        os.system(cmd)

        scene_time = time.time() - scene_start
        runtimes.append(scene_time)

    average_minutes = sum(runtimes) / len(runtimes) / 60
    runtime_data[label] = round(average_minutes, 2)