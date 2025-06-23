import json
import argparse
from pathlib import Path

def calculate_average_metrics(parent_dir: Path, method: str):
    ssim  = 0.0
    psnr  = 0.0
    lpips = 0.0
    count = 0
    
    for scene in parent_dir.iterdir():
        if not scene.is_dir():
            continue

        results_file = scene / 'results.json'
        if not results_file.exists():
            print(f"[WARNING] {results_file} does not exist for scene {scene}.")
            continue

        with open(results_file, 'r') as f:
            data = json.load(f)
            if method in data:
                s = data[method]["SSIM"]
                p = data[method]["PSNR"]
                l = data[method]["LPIPS"]
                ssim  += s
                psnr  += p
                lpips += l
                count += 1
                print(f"Scene {scene}: PSNR={p:.2f} | SSIM={s:.2f} | LPIPS={l:.2f}")
            else:
                print(f"[WARNING] {method} not found in result file for scene {scene}.")

    if count > 0:
        ssim  /= count
        psnr  /= count
        lpips /= count

    output_file = parent_dir / "nvs.json"
    if output_file.exists():
        with open(output_file, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = {}

    metrics[method] = {
        "SSIM": ssim,
        "PSNR": psnr,
        "LPIPS": lpips
    }
    print(f"Average metrics for {method}:")
    print(f"- SSIM  ↑: {ssim:.2f}")
    print(f"- PSNR  ↑: {psnr:.2f}")
    print(f"- LPIPS ↓: {lpips:.2f}")

    print(f"Writing metrics to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate average metrics from results.json files.")
    parser.add_argument("--dataset_dir", "-d", type=str, required=True, help="Path to the dataset parent directory")
    parser.add_argument("--method", "-m", type=str, default="ours_30000", help="Method name to calculate average metrics for")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        print(f"Error: {dataset_dir} is not a valid directory.")
        exit(1)

    child_dirs = [child for child in dataset_dir.iterdir() if child.is_dir()]
    print(f"Number of child directories: {len(child_dirs)}")

    calculate_average_metrics(dataset_dir, method=args.method)