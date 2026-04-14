import os
import json
import tyro
import yaml

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union
from typing_extensions import assert_never

from scene import Config, Runner, get_ckpt_files
from scene.utils import post_process_mesh, write_mesh


@dataclass
class TsdfExtraction:
    # Depth samples beyond this distance (meters) are ignored during TSDF fusion, default to 2x scene scale
    max_depth: float = -1
    # Side length of each voxel (meters) in the TSDF grid (spatial resolution), default to max_depth / 1024
    voxel_size: float = -1
    # Only distances within [-sdf_trunc, +sdf_trunc] from a surface are integrated, default to 4x voxel_size
    sdf_trunc: float = -1

    def safe_get(self, scene_scale):
        max_depth = self.max_depth if self.max_depth > 0 else 2.0 * scene_scale
        voxel_size = self.voxel_size if self.voxel_size > 0 else max_depth / 1024.0
        sdf_trunc = self.sdf_trunc if self.sdf_trunc > 0 else 4.0 * voxel_size
        return max_depth, voxel_size, sdf_trunc


@dataclass
class Args:
    # Path to the config cfg.yml file saved during training
    cfg_file: str = "results/garden/cfg.yml"
    # The specific training step to load checkpoint, default to the latest step
    ckpt_step: int = -1
    # Which method to use for mesh extraction
    extraction: Union[TsdfExtraction] = field(default_factory=TsdfExtraction)
    # How many clusters to keep during post-processing, 0 to keep all
    num_clusters: int = 0
    # Clusters with triangle count smaller than this number will be removed, 0 to keep all
    min_triangles: int = 0
    # Render trajectory with the extracted mesh
    render_traj: bool = True


def main(args: Args, cfg: Config):
    ckpt_dir = Path(cfg.result_dir) / "ckpts"
    ckpt_files, ckpt_step = get_ckpt_files(ckpt_dir, args.ckpt_step)
    print(f"[>] Rendering from {len(ckpt_files)} checkpoint(s) at step {ckpt_step}")

    mesh_dir = Path(cfg.result_dir) / "mesh"
    os.makedirs(mesh_dir, exist_ok=True)

    # Override runner params for rendering
    cfg.ckpt = [str(f) for f in ckpt_files]
    cfg.disable_viewer = True
    runner = Runner(local_rank=0, world_rank=0, world_size=1, cfg=cfg)

    if isinstance(args.extraction, TsdfExtraction):
        max_depth, voxel_size, sdf_trunc = args.extraction.safe_get(runner.scene_scale)
        config = {
            "max_depth": max_depth,
            "voxel_size": voxel_size,
            "sdf_trunc": sdf_trunc,
            "num_clusters": args.num_clusters,
            "min_triangles": args.min_triangles,
        }
        with open(mesh_dir / f"tsdf_step{ckpt_step}.json", "w") as f:
            json.dump(config, f, indent=4)

        volume = runner.run_tsdf_fusion(max_depth, voxel_size, sdf_trunc)
        print(f"[>] Extracting mesh from TSDF volume...")
        mesh = volume.extract_triangle_mesh()
        print(f"[>] Num vertices: {len(mesh.vertices)}")

        print(f"[>] Post-processing mesh...")
        post = post_process_mesh(mesh, args.num_clusters, args.min_triangles)
        print(f"[>] Num vertices post-process: {len(mesh.vertices)}")

        mesh_file = mesh_dir / f"tsdf_step{ckpt_step}.ply"
        write_mesh(mesh_file, post)

        if args.render_traj:
            runner.render_traj_with_mesh(mesh_file, mesh_dir / f"tsdf_traj_{ckpt_step + 1}.mp4")
    else:
        assert_never(args.extraction)


if __name__ == "__main__":
    extraction_types = {
        "tsdf": (
            "Extract mesh using TSDF fusion of rendered depths",
            Args(extraction=TsdfExtraction()),
        ),
    }
    args = tyro.extras.overridable_config_cli(extraction_types)

    cfg_file = Path(args.cfg_file)
    assert cfg_file.exists(), cfg_file

    with open(cfg_file, "r") as f:
        cfg_dict = yaml.unsafe_load(f)
    
    cfg = Config(**cfg_dict)
    main(args, cfg)
