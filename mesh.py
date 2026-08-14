import os
import json
import tyro
import yaml

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Union, List, Literal
from typing_extensions import assert_never

import numpy as np

from scene import Config, Runner, get_ckpt_files
from scene.utils import post_process_mesh, write_mesh


@dataclass
class TsdfExtraction:
    # TSDF grid resolution, ignored when voxel_size > 0
    resolution: int = 512
    # Absolute voxel side length in world units, overrides `resolution` when > 0, mainly used to reproduce metrics
    voxel_size: float = 0.0
    # SDF truncation band width, in voxels (truncation distance = trunc_voxels * voxel_size)
    trunc_voxels: float = 4.0
    # Rendered depth beyond this distance (world units) is ignored during fusion, default to scene scale if <= 0
    max_depth: float = 0.0
    # Discard grazing-incidence depths (rendered normal near-perpendicular to the view ray) before fusion
    depth_filter: bool = False
    # Path to a JSON bounds file with an "aabb_range" key: [[xmin,xmax],[ymin,ymax],[zmin,zmax]], world frame
    bounds_file: str = ""
    # Marching-cubes backend: "vbg" is recommended for object-centric scens, "scalable" is suitable for large scenes
    backend: Literal["auto", "vbg", "scalable"] = "auto"
    # The scene half-extent (world units) below which "auto" backend may pick vbg
    vbg_max_scale: float = 3.0

    def safe_get(self, scene_scale):
        max_depth = self.max_depth if self.max_depth > 0 else scene_scale
        voxel_size = self.voxel_size if self.voxel_size > 0 else (2.0 * scene_scale) / self.resolution
        return max_depth, voxel_size, self.trunc_voxels, self.depth_filter

    def get_name(self):
        return "tsdf"


@dataclass
class Args:
    # Path to the config cfg.yml file saved during training
    cfg_file: str = "results/cfg.yml"
    # The specific training step to load checkpoint, default to the latest step if < 0
    ckpt_step: int = -1
    # Which method to use for mesh extraction
    extraction: Union[TsdfExtraction] = field(default_factory=TsdfExtraction)
    # How many clusters to keep during post-processing, 0 to keep all
    num_clusters: int = 0
    # List of 1-based ranks to exclude during post-processing
    skip_clusters: List[int] = field(default_factory=lambda: [])
    # Clusters with triangle count smaller than this number will be removed, 0 to keep all
    min_triangles: int = 0
    # Reduce the number of triangle count to be at most this decimation target, 0 to keep all 
    decimate_target: int = 0
    # Whether to save the extracted mesh in .glb format (instead of .ply)
    export_glb: bool = False
    # Render trajectory with the extracted mesh, for visualization purposes only
    render_traj: bool = False
    # Depth cutoff distance in trajectory rendering (render_traj) will be multiplied by this factor
    render_traj_depth_cutoff_factor: float = 1.0


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
    if isinstance(args.extraction, TsdfExtraction):
        max_depth, voxel_size, trunc_voxels, depth_filter = args.extraction.safe_get(runner.scene_scale)
        # TSDF fusion needs a rendered depth; the grazing-angle filter also needs the
        # rendered camera-space normal, which the depth-only "ZD" mode does not produce.
        assert cfg.depth_render_mode is not None, \
            "TSDF fusion requires a depth_render_mode to be set, did you miss it during trainig?"
        assert not depth_filter or cfg.depth_render_mode != "ZD", \
            "depth_filter requires a depth_render_mode that renders normals (anything but ZD)"

        # Optional bounding box: cull back-projected points outside the box, and size the grid to the box
        # at ~2048^3 unless an explicit voxel_size was given. The box is in the (un-normalized) world frame.
        bounds = None
        bounds_file = args.extraction.bounds_file
        if args.extraction.bounds_file:
            with open(bounds_file, "r") as f:
                meta = json.load(f)
            assert "aabb_range" in meta, f"{bounds_file} has no 'aabb_range' key"
            bounds = np.array(meta["aabb_range"], dtype=np.float64)  # [3, 2] per-axis [min, max]
            if cfg.normalize_world_space or cfg.center_world_space:
                print("[!] Warning: bounds are in the world frame but this scene was normalized/recentered.")
            if args.extraction.voxel_size <= 0:
                voxel_size = float(np.max(bounds[:, 1] - bounds[:, 0]) / 2048.0)

        # Resolve the marching-cubes backend
        backend = args.extraction.backend
        if backend == "auto":
            grid_dim = 2.0 * runner.scene_scale / max(voxel_size, 1e-8)
            use_vbg = runner.scene_scale < args.extraction.vbg_max_scale and grid_dim < 4000.0
            backend = "vbg" if use_vbg else "scalable"

        print(
            f"[>] TSDF fusion: max_depth={max_depth:.2f} | voxel_size={voxel_size:.4f} | "
            f"trunc_voxels={trunc_voxels:.1f} | depth_filter={depth_filter} | "
            f"bounds={'yes' if bounds is not None else 'no'} | backend={backend}"
        )
        mesh = runner.extract_tsdf_mesh(max_depth, voxel_size, trunc_voxels, depth_filter, bounds, backend)
    else:
        assert_never(args.extraction)

    # Filter disconnected parts and perform mesh decimation
    print("[>] Post-processing mesh...")
    post = post_process_mesh(mesh, args.num_clusters, args.skip_clusters, args.min_triangles, args.decimate_target)
    suffix = "glb" if args.export_glb else "ply"
    mesh_file = mesh_dir / f"{args.extraction.get_name()}_step{ckpt_step}.{suffix}"
    write_mesh(mesh_file, post)  # save mesh based on the chosen format

    # Save config params
    config = asdict(args.extraction)
    config.update({
        "num_clusters": args.num_clusters,
        "skip_clusters": args.skip_clusters,
        "min_triangles": args.min_triangles,
        "decimate_target": args.decimate_target,
    })
    with open(mesh_dir / f"{args.extraction.get_name()}_step{ckpt_step}.json", "w") as f:
        json.dump(config, f, indent=4)

    # Render trajectory but with the rendered mesh instead of rendered alpha
    if args.render_traj:
        video_file = mesh_dir / f"{args.extraction.get_name()}_traj_{ckpt_step + 1}.mp4"
        runner.render_traj_with_mesh(
            mesh_file=mesh_file, video_file=video_file,
            depth_cutoff_factor=args.render_traj_depth_cutoff_factor
        )
        print(f"[>] Video saved to: {video_file}")


if __name__ == "__main__":
    extraction_types = {
        TsdfExtraction().get_name(): (
            "Extract mesh using single-level TSDF fusion of rendered depths",
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
