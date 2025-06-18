# GS-2M: Gaussian Splatting for Joint Mesh Reconstruction and Material Decomposition

## Installation
The code has been tested on Windows and Linux. In general, you will need a working C++ compiler to build all CUDA submodules:
- Windows: please install VS BuildTools version [`17.9.7`](https://learn.microsoft.com/en-us/visualstudio/releases/2022/release-history)
(MSVC `19.39`) as newer versions require CUDA `>=12.4`
- Linux: a recent version of GCC is sufficient (we tested with `11.4`)

Please use `conda`/`mamba` to manage your local environment:
```
conda env create --file environment.yml
conda activate gs2m
```
