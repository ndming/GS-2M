import json
import os
import time

import numpy as np
import open3d as o3d
from pathlib import Path

scenes = ["Barn", "Truck"]
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
    # Export the scene's bounding box if we haven't done that
    if not Path(f"{data_base_path}/{scene}/transforms.json").exists():
        print(f"[>] Could not find transforms.json, exporting one...")
        cmd = f"python scripts/preprocess/convert_json.py --data_dir {data_base_path}/{scene}"
        print("[>] " + cmd)
        os.system(cmd)

    scene_start = time.time()

    common_args = f"-r 2 --densify_grad_abs_threshold 0.00015 --opacity_prune_threshold 0.05"
    cmd = f'python train.py -s {data_base_path}/{scene} -m {out_base_path}/{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = f"--tnt --label {label}"
    cmd = f'python render.py -m {out_base_path}/{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    scene_time = time.time() - scene_start
    runtimes.append(scene_time)

    if scene == "Truck":
        # Rotate the mesh to align with GT point cloud
        mesh_dir = Path(f"{out_base_path}/{scene}/train/{label}_30000/mesh")
        mesh = o3d.io.read_triangle_mesh(str(mesh_dir / "tsdf_post.ply"))
        theta = np.pi / 8
        R = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]])
        mesh.rotate(R, center=(0, 0, 0))
        o3d.io.write_triangle_mesh(str(mesh_dir / "tsdf_post.ply"), mesh)

    cmd = f"python scripts/eval_tnt/run.py " + \
          f"--dataset-dir {data_base_path}/{scene} " + \
          f"--traj-path {data_base_path}/{scene}/{scene}_COLMAP_SfM.log " + \
          f"--ply-path {out_base_path}/{scene}/train/{label}_30000/mesh/tsdf_post.ply"
    print("[>] " + cmd)
    os.system(cmd)
    print(f"==> Done with scene: {scene} <===\n")

average_minutes = sum(runtimes) / len(runtimes) / 60
runtime_data[label] = round(average_minutes, 2)


with open(runtime_file, 'w') as f:
    json.dump(runtime_data, f, indent=2)