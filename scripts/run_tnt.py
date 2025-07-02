import json
import os
import time
from pathlib import Path

scenes = ["Barn", "Caterpillar", "Courthouse", "Ignatius", "Meetingroom", "Truck"]
data_base_path='/home/zodnguy1/datasets/tnt'
out_base_path='output/tnt'

runtime_file = Path(out_base_path) / 'runtime.json'
runtime_data = {}
if runtime_file.exists():
    with open(runtime_file, 'r') as f:
      runtime_data = json.load(f)

label = f'ours_wo-brdf'
runtimes = []
for scene in scenes:
    scene_start = time.time()

    common_args = f"-r 2"
    cmd = f'python train.py -s {data_base_path}/{scene} -m {out_base_path}/{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = f"--tnt --label {label}"
    cmd = f'python render.py -m {out_base_path}/{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    scene_time = time.time() - scene_start
    runtimes.append(scene_time)

    cmd = f"python scripts/eval_tnt/run.py " + \
          f"--dataset-dir {data_base_path}/{scene} " + \
          f"--traj-path {data_base_path}/{scene}/{scene}_COLMAP_SfM.log " + \
          f"--ply-path {out_base_path}/{scene}/train/{label}_30000/meshes/tsdf_post.ply"
    print("[>] " + cmd)
    os.system(cmd)
    print(f"==> Done with scene: {scene} <===\n")

average_minutes = sum(runtimes) / len(runtimes) / 60
runtime_data[label] = round(average_minutes, 2)

with open(runtime_file, 'w') as f:
    json.dump(runtime_data, f, indent=2)