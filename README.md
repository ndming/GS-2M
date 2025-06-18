# GS-2M: Gaussian Splatting for Joint Mesh Reconstruction and Material Decomposition

## Installation
The code has been tested on Windows and Linux.

In general, you will need a working C++ compiler to build all CUDA submodules:
- Windows: please install VS BuildTools version [`17.9.7`](https://learn.microsoft.com/en-us/visualstudio/releases/2022/release-history)
(MSVC `19.39`) as newer versions require CUDA `>=12.4`
- Linux: a recent version of GCC is sufficient (we tested with `11.4`)

Please use `conda`/`mamba` to manage your local environment:
```
conda env create --file environment.yml
conda activate gs2m
```

## Acknowledgements
This repository and the entire project are based on previous Gaussian splatting works. We acknowledge and appreciate
all the great research and publicly available code that made this possible.
- Baseline and core structure: [3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- High-quality surface reconstruction: [PGSR](https://zju3dv.github.io/pgsr/)
- Material decomposition: [GS-IR](https://lzhnb.github.io/project-pages/gs-ir.html) and [GS-ROR2](https://arxiv.org/abs/2406.18544)
- Deferred reflection: [3DGS-DR](https://gapszju.github.io/3DGS-DR/)
- Preprocessed DTU dataset: [2DGS](https://surfsplatting.github.io/)
- Preprocessed TnT dataset: [GOF](https://niujinshuchong.github.io/gaussian-opacity-fields/)
