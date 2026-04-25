import os
import json
import tyro
import yaml

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Union
from typing_extensions import assert_never

import numpy as np

from scene import Config, Runner, get_ckpt_files
from scene.utils import post_process_mesh, write_mesh


@dataclass
class TsdfSingleExtraction:
    # Depth samples beyond this distance (meters) are ignored during TSDF fusion, default to scene radius if < 0
    max_depth: float = -1
    # Side length of each voxel (meters) in the TSDF grid, default to max_depth / 512 if < 0
    voxel_size: float = -1
    # The SDF truncation distance is computed from this parameter as voxel_size * sdf_trunc_factor
    sdf_trunc_factor: float = 4.0

    def safe_get(self, scene_scale):
        max_depth = self.max_depth if self.max_depth > 0 else scene_scale
        voxel_size = self.voxel_size if self.voxel_size > 0 else max_depth / 512
        return max_depth, voxel_size, self.sdf_trunc_factor
    
    def get_name(self):
        return "tsdf_single"
    

@dataclass
class TsdfMultiExtraction:
    # Depth samples beyond this distance (meters) are ignored during TSDF fusion, default to scene radius if < 0
    max_depth: float = -1
    # Side length (meters) of each foreground voxel in the TSDF grid, default to max_depth / 512 if < 0
    base_voxel_size: float = -1
    # How many levels to fuse depths in multi-resolution TSDF grid, where each level double the voxel size from the previous, 
    # default to as many levels as possible until voxel size > max_depth / 16. The number of levels is capped to 4.
    num_levels: int = -1
    # The SDF truncation distance is computed from this parameter as voxel_size * sdf_trunc_factor
    sdf_trunc_factor: float = 4.0

    def safe_get(self, scene_scale):
        max_depth = self.max_depth if self.max_depth > 0 else scene_scale
        base_voxel_size = self.base_voxel_size if self.base_voxel_size > 0 else max_depth / 512
        num_levels = min(self.num_levels, 4) if self.num_levels > 0 else self.num_levels
        if num_levels <= 0:
            # Each level doubles voxel size; stop when coarsest voxel > scene_scale / 16
            # i.e. 2^(n-1) * base_voxel < scene_scale / 16  →  n < log2(scene_scale / (16 * base))
            num_levels = max(2, int(np.ceil(np.log2(scene_scale / (16.0 * base_voxel_size)))) + 1)
            num_levels = min(num_levels, 4)  # cap at 5 levels
        return max_depth, base_voxel_size, num_levels, self.sdf_trunc_factor

    def get_name(self):
        return "tsdf_multi"


@dataclass
class Args:
    # Path to the config cfg.yml file saved during training
    cfg_file: str = "results/garden/cfg.yml"
    # The specific training step to load checkpoint, default to the latest step if < 0
    ckpt_step: int = -1
    # Which method to use for mesh extraction
    extraction: Union[TsdfSingleExtraction, TsdfMultiExtraction] = field(default_factory=TsdfSingleExtraction)
    # How many clusters to keep during post-processing, 0 to keep all
    num_clusters: int = 0
    # Clusters with triangle count smaller than this number will be removed, 0 to keep all
    min_triangles: int = 0
    # Reduce the number of triangle count to be at most this decimation target, 0 to keep all 
    decimate_target: int = 0
    # Render trajectory with the extracted mesh
    render_traj: bool = True
    # Depth cutoff distance in trajectory rendering (render_traj) will be multiplied by this factor
    depth_cutoff_factor: float = 1.0


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

    # Extract mesh based on the chosen method
    if isinstance(args.extraction, TsdfSingleExtraction):
        max_depth, voxel_size, sdf_trunc_factor = args.extraction.safe_get(runner.scene_scale)
        print(
            f"[>] Single-level TSDF fusion: "
            f"max_depth={max_depth:.2f} | voxel_size={voxel_size:.2f} | trunc_factor={sdf_trunc_factor:.1f}"
        )
        mesh = runner.run_tsdf_mesh_extraction(max_depth, voxel_size, sdf_trunc_factor)
    elif isinstance(args.extraction, TsdfMultiExtraction):
        max_depth, base_voxel_size, num_levels, sdf_trunc_factor = args.extraction.safe_get(runner.scene_scale)
        print(
            f"[>] Multi-level TSDF fusion: "
            f"max_depth={max_depth:.2f} | base_voxel_size={base_voxel_size:.2f} | num_levels={num_levels} | trunc_factor={sdf_trunc_factor:.1f}"
        )
        mesh = runner.run_hierarchical_tsdf_mesh_extraction(max_depth, base_voxel_size, num_levels, sdf_trunc_factor)
    else:
        assert_never(args.extraction)

    # Filter disconnected parts and perform mesh decimation
    post = post_process_mesh(mesh, args.num_clusters, args.min_triangles, args.decimate_target)
    mesh_file = mesh_dir / f"{args.extraction.get_name()}_step{ckpt_step}.ply"
    write_mesh(mesh_file, post)  # save mesh

    # Save config params
    config = asdict(args.extraction)
    config.update({
        "num_clusters": args.num_clusters,
        "min_triangles": args.min_triangles,
        "decimate_target": args.decimate_target,
    })
    with open(mesh_dir / f"{args.extraction.get_name()}_step{ckpt_step}.json", "w") as f:
        json.dump(config, f, indent=4)

    # Render trajectory but with the rendered mesh instead of rendered alpha
    if args.render_traj:
        video_file = mesh_dir / f"{args.extraction.get_name()}_traj_{ckpt_step + 1}.mp4"
        runner.render_traj_with_mesh(
            mesh_file=mesh_file,
            video_file=video_file,
            depth_cutoff_factor=args.depth_cutoff_factor
        )
        print(f"[>] Video saved to: {video_file}")


if __name__ == "__main__":
    extraction_types = {
        TsdfSingleExtraction().get_name(): (
            "Extract mesh using single-level TSDF fusion of rendered depths",
            Args(extraction=TsdfSingleExtraction()),
        ),
        TsdfMultiExtraction().get_name(): (
            "Extract mesh using multi-level TSDF fusion of rendered depths (experimental)",
            Args(extraction=TsdfMultiExtraction()),
        ),
    }
    args = tyro.extras.overridable_config_cli(extraction_types)

    cfg_file = Path(args.cfg_file)
    assert cfg_file.exists(), cfg_file

    with open(cfg_file, "r") as f:
        cfg_dict = yaml.unsafe_load(f)
    
    cfg = Config(**cfg_dict)
    main(args, cfg)
