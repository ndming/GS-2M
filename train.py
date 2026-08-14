import pathlib
import time
import tyro

from gsplat.distributed import cli
from gsplat.strategy import DefaultStrategy, MCMCStrategy

from scene import Config, Runner


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    # Distributed training is not compatible with some features
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("[!] Viewer will be disabled in distributed training.")
    if cfg.post_processing is not None:
        assert world_size == 1, "Distributed training is not supported for post-processing"
    if cfg.depth_render_mode is not None and cfg.depth_render_mode != "ZD":
        assert world_size == 1, "Distributed training is not supported for depth render modes other than ZD"

    # Init runner and start training
    runner = Runner(local_rank, world_rank, world_size, cfg)
    runner.train()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down viewer...")
            runner.server.stop()


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=9 python train.py [...]

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py [...] --steps-scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "adc": (
            "Gaussian splatting training using densification heuristics from the original paper (adaptive density control).",
            Config(strategy=DefaultStrategy(verbose=False)),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5, init_scale=0.1,
                opacity_reg=0.01, scale_reg=0.01,
                strategy=MCMCStrategy(verbose=False)
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    # Try importing extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    if cfg.with_ut:
        assert cfg.with_eval3d, "Training with UT requires setting `with_eval3d` flag."

    if cfg.post_processing == "ppisp":
        import torch
        import warnings
        from packaging import version
        # PPISP modules uses SequentialLR which emits an anoying warning when PyTorch < 2.9
        if version.parse(torch.__version__) < version.parse("2.9"):
            warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

    if cfg.normalize_world_space and cfg.center_world_space:
        print("[!] Disabling world space centering: normalize_world_space is enabled and takes precedence")
        cfg.center_world_space = False

    # Check if depth render mode should be specified
    should_set_depth_render_mode = (
           cfg.depth_point_lambda    > 0.0
        or cfg.depth_image_lambda    > 0.0
        or cfg.depth_normal_lambda   > 0.0
        or cfg.multi_view_geo_lambda > 0.0
        or cfg.multi_view_ncc_lambda > 0.0
    )
    if should_set_depth_render_mode:
        assert cfg.depth_render_mode is not None, ("depth_render_mode was not set for depth-related losses, "
        "please choose a depth_render_mode or disable losses that require rendered depths.")

    # Check if depth render mode should support normal rendering
    depth_render_mode_must_support_normal = (
           cfg.depth_normal_lambda   > 0.0
        or cfg.multi_view_geo_lambda > 0.0
        or cfg.multi_view_ncc_lambda > 0.0
    )
    if depth_render_mode_must_support_normal:
        assert cfg.depth_render_mode != "ZD", ("The chosen depth render mode (ZD) does not support normal rendering, "
        "please choose another depth_render_mode or disable losses that require rendered normals.")

    # A few functionalities have not yet supported batch size > 1
    if cfg.multi_view_geo_lambda > 0.0 or cfg.multi_view_ncc_lambda > 0.0:
        assert cfg.batch_size == 1, ("Multi-view losses require batch size 1, "
        "please set batch_size to 1 or set both multi_view_geo_lambda and multi_view_ncc_lambda to 0.0.")
    if cfg.post_processing == "ppisp":
        assert cfg.batch_size == 1, "PPISP requires batch size 1."

    assert pathlib.Path(cfg.data_dir).exists(), f"Could NOT find data directory {cfg.data_dir}"
    cli(main, cfg, verbose=True)
