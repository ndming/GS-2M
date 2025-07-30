import json
import os
import time
from pathlib import Path

scenes = ["helmet", "car", "teapot", "ball", "coffee", "toaster"]
ref_thresholds = [0.2, 0.5, 0.1, 0.4, 0.2, 0.1]
lambda_smooths = [0.5, 0.0, 0.5, 0.1, 0.5, 0.8]
lambda_normals = [2.0, 0.5, 0.1, 8.0, 0.1, 4.0]

data_base_path='/home/zodnguy1/datasets/shiny'
out_base_path='output/shiny'

runtime_file = Path(out_base_path) / 'runtime.json'
runtime_data = {}
if runtime_file.exists():
    with open(runtime_file, 'r') as f:
        runtime_data = json.load(f)

label = "ours"
runtimes = []
for scene, ref, sm, norm in zip(scenes, ref_thresholds, lambda_smooths, lambda_normals):
    scene_start = time.time()

    common_args = f"--material --eval --white_background --reflection_threshold {ref} --lambda_smooth {sm} --lambda_normal {norm}"
    if scene == "ball": common_args += " --mask_gt"

    cmd = f'python train.py -s {data_base_path}/{scene} -m {out_base_path}/{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = f"--blender --label {label}"
    cmd = f'python render.py -m {out_base_path}/{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = f"--method {label} --force"
    cmd = f"python scripts/vis_blender.py -m {out_base_path}/{scene} -d {data_base_path}/{scene} {common_args}"
    print("[>] " + cmd)
    os.system(cmd)

average_minutes = sum(runtimes) / len(runtimes) / 60
runtime_data[label] = round(average_minutes, 2)

with open(runtime_file, 'w') as f:
    json.dump(runtime_data, f, indent=2)