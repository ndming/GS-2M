import time
import tyro

from gsplat.distributed import cli
from gsplat.strategy import DefaultStrategy, MCMCStrategy

from scene import Config, Runner


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    # Init runner and start training
    runner = Runner(local_rank, world_rank, world_size, cfg)
    runner.train()
    runner.export_ppisp_reports()

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
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(strategy=DefaultStrategy(verbose=False)),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(strategy=MCMCStrategy(verbose=False)),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    # try import extra dependencies
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

    # Check if depth regularization is properly configured
    should_set_depth_render_mode = (
        cfg.depth_point_lambda  > 0.0 or
        cfg.depth_image_lambda  > 0.0 or
        cfg.depth_normal_lambda > 0.0
    )
    if should_set_depth_render_mode:
        assert cfg.depth_render_mode is not None, "Depth regularization is enabled but depth_render_mode is not set"
    if cfg.depth_normal_lambda > 0.0:
        assert cfg.depth_render_mode == "plane", "Depth normal consistency loss is only supported with plane depth"

    cli(main, cfg, verbose=True)
