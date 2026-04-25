# GS-2M: Material-aware Gaussian Splatting for High-fidelity Mesh Reconstruction

<p>
  <a href="https://arxiv.org/abs/2509.22276" target="_blank"><img src="https://img.shields.io/badge/arXiv-2509.22276-b31b1b.svg?style=for-the-badge" alt="arXiv"></a> <a href="https://ndming.github.io/publications/gs2m" target="_blank"><img src="https://img.shields.io/badge/Project-Page-blue.svg?style=for-the-badge" alt="Project Page"></a> <a href="https://onlinelibrary.wiley.com/doi/10.1111/cgf.70347" target="_blank"><img src="https://img.shields.io/badge/CGF-Paper-F5E642?style=for-the-badge" alt="CGF Paper"></a>
</p>

![cover](media/cover.png)

🚧 Migration to [gsplat](https://github.com/nerfstudio-project/gsplat) is not fully implemented,
expect breaking changes and incomplete features.

## Updates
- **2026.04.08**: Added COLMAP `4.0` conversion script with the ability to sample frames from videos
- **2026.03.23**: Added USDZ export script to pack trained Gaussians and the extracted mesh into a `.usdz` scene
- **2026.03.08**: Added foreground masking instructions for high-quality object-centric reconstructions
- **2026.02.06**: GS-2M accpeted to Eurographics 2026
- **2025.09.15**: Initial code release

## Installation
Please make sure you have a working C++ compiler compatible with your preferred CUDA Toolkit version. This usually means using
the default GCC on Linux or installing the correct version of [Visual Studio](https://learn.microsoft.com/en-us/visualstudio/releases/2022/release-history) on Windows.

<details>
<summary><span style="font-weight: bold;">Compatibility matrix between MSVC and CUDA for Windows installation</span></summary>

| **MSVC** | **Visual Studio** | **Minimum CUDA** |
|----------|-------------------|------------------|
| `19.44` | `17.14` | `12.4` |
| `19.43` | `17.13` | `12.4` |
| `19.42` | `17.12` | `12.4` |
| `19.41` | `17.11` | `12.4` |
| `19.40` | `17.10` | `11.6` |
| `19.39` | `17.9`  | `11.6` |
| `19.38` | `17.8`  | `11.6` |

</details>


First, clone the repo and submodules:
```bash
git clone https://github.com/ndming/GS-2M.git --recursive
cd GS-2M
```

From here, you can proceed with the installation either by:
- Installing to a Python virtual environment if CUDA Toolkit is pre-installed to your system, or
- Installing with `conda`/`mamba` if you want CUDA Toolkit to be self-contained in the virtual environment

### Installing to a Python virtual environment
This installation path allows you to use your pre-installed CUDA Toolkit with `nvcc` available system-wide.
Additionally, you will have the freedom to choose the PyTorch version suitable to your setup.

Create a local [venv](https://docs.python.org/3/library/venv.html) with `pip<=25.2`:
```bash
python -m venv .venvs/gs2m
source .venvs/gs2m/bin/activate     # Linux (bash)
.\.venvs\gs2m\Scripts\Activate.ps1  # Windows (PowerShell)
```

Upgrade to `setuptools==68` and pin to `numpy<2.0.0`:
```bash
pip install --upgrade setuptools==68 wheel numpy==1.26.4
```

Install a version of [PyTorch and TorchVision](https://pytorch.org/get-started/previous-versions/) compatible with
your CUDA version (`cu12.8` as an example):
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

Install the remaining packages and submodules:
```bash
pip install -r requirements.txt --no-build-isolation
```

### Installing with `conda`/`mamba`
This installation path does not require a pre-installed CUDA Toolkit, as CUDA will be installed automatically inside
the Conda environment. By default, the CUDA Toolkit and PyTorch version are pinned to `12.8` and `2.9.1`, respectively.
```bash
conda env create --file environment.yml
conda activate gs2m
pip install -r requirements.txt --no-build-isolation
``` 

## Usage
As with other Gaussian splatting pipelines, reconstructing a scene from multi-view images involves three key steps:
- Structure from Motion (SfM)
- Training Gaussians
- Mesh extraction: this is specific to surface reconstruction pipelines, including GS-2M

### Structure from Motion
SfM prepares training data from unposed images, for which [COLMAP](https://github.com/colmap/colmap) is the recommended choice.

While you can follow the instructions from the original [3DGS](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#processing-your-own-scenes) repo to run COLMAP for your scene, we prepared an automatic COLMAP run script
for convenience. It is based on the original `convert.py` script of 3DGS, modified to account for the new features of
COLMAP `4.0` such as global mapping ([GLOMAP](https://lpanaf.github.io/eccv24_glomap/)) or neural feature extraction and
matching from [ALIKED](https://github.com/Shiaoming/ALIKED) and [LightGlue](https://github.com/cvg/lightglue).

First, please follow COLMAP's [offical installation guide](https://colmap.github.io/install.html) to install the base 
`colmap` binary with version `4.0` as the minimum. Once done, organize your images with the following structure:
```
scene/
└── input/
    ├── 001.png
    ├── 002.png
    └── ...
```

Run the following command to begin the SfM process (make sure the installed `gs2m` environment is activated):
```bash
python scripts/colmap.py --source_path <path/to/scene>
```

<details>
<summary><span style="font-weight: bold;">Options to sample from videos or enable advanced COLMAP features</span></summary>

- `sample_from`: path to a video or directory used to generate input frames. When provided, frames are sampled (for videos)
or copied (for directories) and written to a fresh `input` directory created under `scene` (or overwrite the existing ones 
if `--sample_overwrite` is given). If not set (default), the script uses the existing contents of `input` without modification.
- `sample_interval`: sample frames every this interval, only applies if `sample_from` is not empty
- `colmap_feature_extraction`: one of `SIFT`, `ALIKED_N16ROT`, or `ALIKED_N32` (default to `SIFT`)
- `colmap_feature_matching`: one of `SIFT_BRUTEFORCE`, `SIFT_LIGHTGLUE`, `ALIKED_BRUTEFORCE`, or `ALIKED_LIGHTGLUE` 
(default to `SIFT_BRUTEFORCE`)

</details>

A COLMAP-prepared training directory for GS-2M may look like the following. Note that `masks` is completely optional.
If you have foreground masks of the target object, organize them as shown below. 
```
scene/
├── images/
│   ├── 001.png
│   ├── 002.png
│   └── ...
├── sparse/
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
├── masks/
│   ├── 001.png
│   ├── 002.png
│   └── ...
└── database.db
```

<details>
<summary><span style="font-weight: bold;">Masking for object-centric reconstruction</span></summary>

To reconstruct geometrically corrected objects from scenes with overwhelming background, consider following these steps
to extract foreground masks of the target object. We will be using the amazing model [BiRefNet](https://github.com/ZhengPeng7/BiRefNet?tab=readme-ov-file) from [this paper](https://arxiv.org/pdf/2401.03407).

Clone the BiRefNet repo to the `scripts` directory (the location is crucial):
```bash
git clone https://github.com/ZhengPeng7/BiRefNet.git scripts/birefnet
```

Optional: download a [pre-trained model](https://drive.google.com/drive/folders/1s2Xe0cjq-2ctnJBR24563yMSCOu4CcxM) from
the official repo to use locally if your network cannot access HuggingFace. We recommend using the general checkpoint:
`BiRefNet_HR-general-epoch_130.pth`

Once you have obtained undistorted RGB images from SfM, run the following script (with `gs2m` activated):
```bash
# On Linux (bash):
PYTHONPATH=scripts/birefnet python scripts/masking.py -i </path/to/scene>/images [-o /path/to/output -w /path/to/weight]

# On Windows (PowerShell)
$env:PYTHONPATH="scripts\birefnet"; python scripts/masking.py -i </path/to/scene>/images [-o /path/to/output -w /path/to/weight]
```
- if `-w` is omitted, the model will fetch weights from HuggingFace
- if `-o` is omitted, the output `masks` dir is created at the location of `images`, matching the above file structure

</details>

### Training
```bash
python train.py default --data-dir <path/to/scene> --result-dir output/scene
```

Please tune the extraction parameters of `train.py`. You can inspect them by using the `-h` flag:
```
python train.py -h
python train.py default -h
```


### Mesh extraction
```bash
python mesh.py tsdf_single --cfg-file output/scene/cfg.yml
```

The `.ply` file of the extracted triangle mesh can be found at:
```
output/scene/mesh/tsdf_single_step*.ply
```

Please tune the extraction parameters of `mesh.py` depending on the nature of your scene:
```
python mesh.py -h
python mesh.py tsdf_single -h
```

### Export trained Gaussians and the extracted mesh to a USDZ scene
With demand for robotic simulation on the rise, we provide an export script that packs trained Gaussians and
the extracted mesh into a single `.usdz` file that can be imported to, for example, IsaacSim `5.1`.

After the extraction step, the Gaussians and mesh are collocated in the same coordinate space, allowing a simulated
robot sensor to pick up high-quality novel views while the robot itself physically interacts within the scene.
```bash
python scripts/export/usd.py --input_ply output/scene/ply/point_cloud_30000.ply \
                             --mesh_ply output/scene/mesh/tsdf_single_step29999.ply \
                             --collision [-o <path/to>/scene.usdz --invisible]
```

We appreciate the amazing [3DGRUT repo](https://github.com/nv-tlabs/3dgrut), from which this export script was based on.

## Evaluation
Please follow these steps to reproduce the evaluation results.

### Mesh reconstruction on the DTU dataset
- Obtain the preprocessed dataset from [2DGS](https://surfsplatting.github.io/), the dataset should be organized as:
```
dtu/
├── scan24/
│   ├── images/
│   ├── sparse/
│   └── ...
├── scan37/
├── scan40/
└── ...
```
- Download the ground truth point clouds from [DTU](https://roboimagedata.compute.dtu.dk/?page_id=36): only the
`SampleSets` and `Points` are required.
- Create a directory named `Official_DTU_Dataset` under `dtu/`, copy `Calibration`, `Cleaned`, `ObsMask`, `Points`,
`Rectified`, `Surfaces` directories from `SampleSets/MVS Data` to `Official_DTU_Dataset/`
- Replace the copied `Official_DTU_Dataset/Points/stl` with `Points/stl`
- Make sure the structure of `Official_DTU_Dataset` is as follows:
```
dtu/Official_DTU_Dataset/
├── Calibration/
├── Cleaned/
├── ObsMask/
├── Points/
├── Rectified/
├── Surfaces/
└── ...
```
- Run the following script:
```bash
python benchmarks/dtu.py --data_base_dir <path/to/dtu>
```
- The reconstruction statistics can be found at:
```bash
output/dtu/stats.json
```

### Material decomposition on the Shiny Blender Synthetic dataset
- Obtain a copy of the [ShinyBlender](https://dorverbin.github.io/refnerf/) synthetic dataset, organized as:
```
shiny/
├── ball/
│   ├── test/
│   ├── train/
│   ├── transforms_test.json
│   └── transforms_train.json
├── car/
├── coffee/
└── ...
```
- Run the following script:
```bash
# You may need to adjust `data_base_path` in `run_shiny.py` to point to your `shiny/`
python scripts/run_shiny.py
```
- Check the decomposition results under:
```
output/shiny/<scene>/test/ours_30000/visual/
```

### Material decomposition on the Glossy Blender Synthetic dataset
- Obtain a copy of the [GlossyBlender](https://liuyuan-pal.github.io/NeRO/) synthetic dataset, organized as:
```
glossy/
├── angel/
│   ├── 0.png
│   ├── 0-camera.pkl
│   ├── 0-depth.png
│   └── ...
├── bell/
├── cat/
└── ...
```
- Run the following script:
```bash
# You may need to adjust `data_base_path` in `run_glossy.py` to point to your `glossy/`
python scripts/run_glossy.py
```
- Check the reconstruction results under:
```
output/glossy/<scene>/test/ours_10000/
```

### Mesh reconstruction on the TnT dataset
- Obtain a copy of the preprocessed dataset from [GOF](https://huggingface.co/datasets/ZehaoYu/gaussian-opacity-fields/tree/main)
- Visit the [download page](https://www.tanksandtemples.org/download/) of the Tanks and Temples Benchmark for GT
- Download Camera Poses (`*_COLMAP_SfM.log`), Alignment (`*_trans.txt`), Cropfiles (`*.json`), and GT (`*.ply`) for
the Barn and Truck scenes
- Please organize your files as follows:
```
tnt/
├── Barn/
│   ├── sparse/
│   ├── images/
│   ├── Barn_COLMAP_SfM.log
│   ├── Barn.ply
│   ├── Barn.json
│   └── Barn_trans.txt
├── Truck/
└── ...
```
- Run the following script:
```bash
# You may need to adjust `data_base_path` in `run_tnt.py` to point to your `tnt/`
python scripts/run_tnt.py
```
- Check the reconstruction results under:
```
output/tnt/<scene>/train/ours_wo-brdf_30000/mesh/evaluation/
```

### Novel-view synthesis on the MipNeRF 360 dataset
- Download all [MipNeRF 360 scenes](https://jonbarron.info/mipnerf360/) and organize the dataset as follows:
```
mipnerf360/
├── bicycle/
│   ├── sparse/
│   ├── images/
│   ├── poses_bounds.npy
├── bonsai/
└── ...
```
- Run the following script:
```bash
python benchmarks/mipnerf360.py --data_base_dir <path/to/mipnerf360>
```
- The reconstruction statistics can be found at:
```bash
output/mipnerf360/stats.json
```

## Acknowledgements
This repository and the entire project are based on previous Gaussian splatting works. We acknowledge and appreciate
all the great research and publicly available code that made this possible.
- Baseline and core structure: [3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- Modular training engine: [gsplat](https://github.com/nerfstudio-project/gsplat)
- High-quality surface reconstruction: [PGSR](https://zju3dv.github.io/pgsr/)
- Improved densification: [AbsGS](https://ty424.github.io/AbsGS.github.io/)
- Material decomposition: [GS-IR](https://lzhnb.github.io/project-pages/gs-ir.html) and [GS-ROR2](https://arxiv.org/abs/2406.18544)
- Depth-normal rendering: [GaussianShader](https://asparagus15.github.io/GaussianShader.github.io/)
- Deferred reflection: [3DGS-DR](https://gapszju.github.io/3DGS-DR/)
- Preprocessed DTU dataset: [2DGS](https://surfsplatting.github.io/)
- Preprocessed TnT dataset: [GOF](https://niujinshuchong.github.io/gaussian-opacity-fields/)
- ShinyBlender synthetic dataset: [Ref-NeRF](https://dorverbin.github.io/refnerf/)
- GlossyBlender synthetic dataset: [NeRO](https://liuyuan-pal.github.io/NeRO/)

## BibTeX
```
@article{nguyen2026gs2m,
   title={GS‐2M: Material‐aware Gaussian Splatting for High‐fidelity Mesh Reconstruction},
   ISSN={1467-8659},
   url={http://dx.doi.org/10.1111/cgf.70347},
   DOI={10.1111/cgf.70347},
   journal={Computer Graphics Forum},
   publisher={Wiley},
   author={Nguyen, D. M. and Avenhaus, M. and Lindemeier, T.},
   year={2026},
   month=apr
}
```
