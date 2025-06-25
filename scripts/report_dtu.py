from argparse import ArgumentParser
from pathlib import Path
from plyfile import PlyData

import json
import numpy as np

scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]

def calculate_average_chamfer(out_base_dir: Path, method: str):
    avg_d2s = 0.0
    avg_s2d = 0.0
    avg_chamfer = 0.0

    count = 0

    for scene in scenes:
        scene = out_base_dir / f"scan{scene}"
        if not scene.exists():
            print(f"Error: {scene} does not exist.")
            continue

        result_file = scene/ "train" / method / "meshes" / "results.json"
        if not result_file.exists():
            print(f"Error: {result_file} not found for scene {scene}.")
            continue

        ply_file = scene / "point_cloud"/ "iteration_30000" / "point_cloud.ply"
        ply_data = PlyData.read(ply_file)
        xyz = np.stack((
            np.asarray(ply_data.elements[0]["x"]),
            np.asarray(ply_data.elements[0]["y"]),
            np.asarray(ply_data.elements[0]["z"])), axis=1)
        n_points = xyz.shape[0]

        with open(result_file, "r") as f:
            result = json.load(f)
            print(f"{scene}:\t {result['overall']:0.2f}\t {n_points}")
            avg_d2s += result["mean_d2s"]
            avg_s2d += result["mean_s2d"]
            avg_chamfer += result["overall"]
            count += 1

    if count == 0:
        print("No scene to evaluate")
        return

    avg_chamfer /= count
    avg_d2s /= count
    avg_s2d /= count
    print(f"Average chamfer: {avg_chamfer:0.2f}")

    chamfer_file = out_base_dir / "chamfer.json"
    if chamfer_file.exists():
        with open(chamfer_file, "r") as f:
            chamfer = json.load(f)
    else:
        chamfer = {}

    if method not in chamfer:
        chamfer[method] = {}
    chamfer[method] = {
        "mean_d2s": avg_d2s,
        "mean_s2d": avg_s2d,
        "overall": avg_chamfer
    }
    print(f"Writing chamfer to {chamfer_file}")
    with open(chamfer_file, "w") as f:
        json.dump(chamfer, f, indent=4)

if __name__ == "__main__":
    parser = ArgumentParser(description="Calculate average Chamfer metrics.")
    parser.add_argument("--output_base_dir", "-d", type=str, required=True, help="Path to the model parent directory")
    parser.add_argument("--method", "-m", type=str, default="ours_30000")
    args = parser.parse_args()

    out_base_dir = Path(args.output_base_dir)
    if not out_base_dir.is_dir():
        print(f"Error: {out_base_dir} is not a valid directory.")
        exit(1)

    child_dirs = [child for child in out_base_dir.iterdir() if child.is_dir()]
    print(f"Number of child directories: {len(child_dirs)}")

    calculate_average_chamfer(out_base_dir, method=args.method)