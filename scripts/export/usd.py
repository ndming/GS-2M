from argparse import ArgumentParser
from pathlib import Path

from nurec import GaussianPLYModel, NuRecExporter, default_nurec_config


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert 3DGS PLY to NuRec USDZ for Isaac Sim")
    parser.add_argument("-i", "--input_ply", required=True, help="Path to input .ply file")
    parser.add_argument("-o", "--output", default="", help="Path to output .usdz (default: same name as input)")
    parser.add_argument("--sh_degree", type=int, default=3, help="Max SH degree (default: 3)")
    args = parser.parse_args()

    ply_file = Path(args.input_ply)
    assert ply_file.exists(), str(ply_file)

    out_file = Path(args.output) if args.output != "" else ply_file.with_suffix(".usdz")

    print(f"[>] Loading model: {ply_file}")
    model = GaussianPLYModel(str(ply_file), max_sh_degree=args.sh_degree)

    exporter = NuRecExporter()
    exporter.export(model, out_file, dataset=None, conf=default_nurec_config())
    print(f"[>] Done!")
