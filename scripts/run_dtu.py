import json
import os
import time
from pathlib import Path

scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
data_base_path='/home/zodnguy1/datasets/dtu'
out_base_path='output/dtu'

runtime_file = Path(out_base_path) / 'runtime.json'
runtime_data = {}
if runtime_file.exists():
    with open(runtime_file, 'r') as f:
      runtime_data = json.load(f)

label = f'ours_wo-brdf_dn-0.015_plane-50'
runtimes = []
for scene in scenes:
    scene_start = time.time()

    common_args = f"-r 2 --lambda_depth_normal 0.015 --lambda_plane 50"
    cmd = f'python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = f"--dtu --label {label}"
    cmd = f'python render.py -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    scene_time = time.time() - scene_start
    runtimes.append(scene_time)

    common_args = f"--split train --method {label}_30000"
    cmd = f"python metrics.py -m {out_base_path}/scan{scene} {common_args}"
    print("[>] " + cmd)
    os.system(cmd)

    cmd = f"python scripts/eval_dtu/evaluate_single_scene.py " + \
          f"--input_ply {out_base_path}/scan{scene}/train/{label}_30000/mesh/tsdf_post.ply " + \
          f"--ref_dir {data_base_path}/scan{scene} " + \
          f"--dtu_dir {data_base_path}/Official_DTU_Dataset"
    print("[>] " + cmd)
    os.system(cmd)
    print(f"==> Done with scene: scan{scene} <===\n")

average_minutes = sum(runtimes) / len(runtimes) / 60
runtime_data[label] = round(average_minutes, 2)


label = f'ours_wo-brdf_dn-0.015_plane-100'
runtimes = []
for scene in scenes:
    scene_start = time.time()

    common_args = f"-r 2 --lambda_depth_normal 0.015 --lambda_plane 100"
    cmd = f'python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = f"--dtu --label {label}"
    cmd = f'python render.py -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    scene_time = time.time() - scene_start
    runtimes.append(scene_time)

    common_args = f"--split train --method {label}_30000"
    cmd = f"python metrics.py -m {out_base_path}/scan{scene} {common_args}"
    print("[>] " + cmd)
    os.system(cmd)

    cmd = f"python scripts/eval_dtu/evaluate_single_scene.py " + \
          f"--input_ply {out_base_path}/scan{scene}/train/{label}_30000/mesh/tsdf_post.ply " + \
          f"--ref_dir {data_base_path}/scan{scene} " + \
          f"--dtu_dir {data_base_path}/Official_DTU_Dataset"
    print("[>] " + cmd)
    os.system(cmd)
    print(f"==> Done with scene: scan{scene} <===\n")

average_minutes = sum(runtimes) / len(runtimes) / 60
runtime_data[label] = round(average_minutes, 2)


with open(runtime_file, 'w') as f:
    json.dump(runtime_data, f, indent=2)