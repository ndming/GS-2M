from argparse import ArgumentParser
from pathlib import Path

import json
import numpy as np

scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]

def calculate_average_chamfer(out_base_dir: Path, method: str, latex: bool):
    d2s_list = []
    s2d_list = []
    chamfer_list = []
    psnr_list = []
    point_list = []

    for scene in scenes:
        scene = out_base_dir / f"scan{scene}"
        if not scene.exists():
            print(f"Error: {scene} does not exist.")
            continue

        result_file = scene/ "train" / method / "meshes" / "results.json"
        if not result_file.exists():
            print(f"Error: {result_file} not found for scene {scene}.")

        point_file = scene / "points.json"
        if not point_file.exists():
            print(f"Error: {point_file} not found for scene {scene}.")

        metric_file = scene / "metrics.json"
        if not metric_file.exists():
            print(f"Error: {metric_file} not found for scene {scene}.")

        with open(point_file, "r") as f:
            points = json.load(f)
            try:
                n_points = points[method]
                point_list.append(n_points)
            except KeyError:
                print(f"Error: Method {method} not found in {point_file} for scene {scene}.")
                n_points = 0

        with open(metric_file, "r") as f:
            metrics = json.load(f)
            try:
                psnr = metrics[method]["psnr"]
                psnr_list.append(psnr)
            except KeyError:
                print(f"Error: Method {method} not found in {metric_file} for scene {scene}.")
                psnr = 0.0

        with open(result_file, "r") as f:
            result = json.load(f)
            print(f"{scene}:\t {result['overall']:0.2f}\t{psnr:0.2f}\t{n_points}")
            d2s_list.append(result["mean_d2s"])
            s2d_list.append(result["mean_s2d"])
            chamfer_list.append(result["overall"])

    avg_chamfer = np.mean(chamfer_list)
    avg_d2s = np.mean(d2s_list)
    avg_s2d = np.mean(s2d_list)
    avg_psnr = np.mean(psnr_list)
    print(f"Average chamfer: {avg_chamfer:0.2f}")
    print(f"Average PSNR: {avg_psnr:0.2f}")

    if latex:
        print("CD: ", *[f"{x:.2f} &" for x in chamfer_list])
        print("PSNR: ", *[f"{x:.2f} &" for x in psnr_list])
        print("Points: ", *[f"{x} &" for x in format_points(point_list)])

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

def format_points(points):
    formatted_points = []
    for x in points:
        if x >= 1000:
            # Round and convert to 'k' format
            formatted_points.append(f"{round(x/1000)}k")
        else:
            formatted_points.append(str(x))
    return formatted_points

if __name__ == "__main__":
    parser = ArgumentParser(description="Calculate average Chamfer metrics.")
    parser.add_argument("--output_base_dir", "-d", type=str, defaul="output/dtu", help="Path to the model parent directory")
    parser.add_argument("--method", "-m", type=str, default="ours_30000")
    parser.add_argument("--latex", action="store_true", help="Output results in lines separated by & for LaTeX table")
    args = parser.parse_args()

    out_base_dir = Path(args.output_base_dir)
    if not out_base_dir.is_dir():
        print(f"Error: {out_base_dir} is not a valid directory.")
        exit(1)

    child_dirs = [child for child in out_base_dir.iterdir() if child.is_dir()]
    print(f"Number of child directories: {len(child_dirs)}")

    calculate_average_chamfer(out_base_dir, method=args.method, latex=args.latex)