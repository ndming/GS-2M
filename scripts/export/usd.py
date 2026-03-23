from argparse import ArgumentParser
from pathlib import Path

from nurec import GaussianPLYModel, NuRecExporter, default_nurec_config, add_mesh_to_usdz


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert 3DGS PLY to NuRec USDZ for Isaac Sim")
    parser.add_argument("-i", "--input_ply", required=True, help="Path to input .ply file")
    parser.add_argument("-o", "--output", default="", help="Path to output .usdz (default: same name as input)")
    parser.add_argument("-m", "--mesh_ply", default="", help="Collision mesh to inject to the converted USD if provided")
    parser.add_argument("--sh_degree", type=int, default=3, help="Max SH degree (default: 3)")
    parser.add_argument("--collision", action="store_true", help="Enable physics collision on mesh")
    parser.add_argument("--invisible", action="store_true", help="Make injected mesh invisible")
    args = parser.parse_args()

    ply_file = Path(args.input_ply)
    assert ply_file.exists(), str(ply_file)

    out_file = Path(args.output) if args.output != "" else ply_file.with_suffix(".usdz")

    print(f"[>] Loading model: {ply_file}")
    model = GaussianPLYModel(str(ply_file), max_sh_degree=args.sh_degree)

    exporter = NuRecExporter()
    exporter.export(model, out_file, dataset=None, conf=default_nurec_config())

    if args.mesh_ply != "":
        mesh_file = Path(args.mesh_ply)
        out_file_mesh = out_file.with_stem(f"{out_file.stem}_mesh")
        add_mesh_to_usdz(
            input_usdz=str(out_file),
            output_usdz=str(out_file_mesh),
            mesh_ply_path=str(args.mesh_ply),
            mesh_usd_path=None,
            set_collision=args.collision,
            set_invisible=args.invisible,
        )

    print(f"[>] Done!")
