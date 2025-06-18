# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import nvdiffrast

from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

with open("README.md", "r") as fh:
    long_description = fh.read()

cuda_sources = [
    'nvdiffrast/common/cudaraster/impl/Buffer.cpp',
    'nvdiffrast/common/cudaraster/impl/CudaRaster.cpp',
    'nvdiffrast/common/cudaraster/impl/RasterImplCUDA.cu',
    'nvdiffrast/common/cudaraster/impl/RasterImpl.cpp',
    'nvdiffrast/common/common.cpp',
    'nvdiffrast/common/rasterize.cu',
    'nvdiffrast/common/interpolate.cu',
    'nvdiffrast/common/textureCUDA.cu',
    'nvdiffrast/common/texture.cpp',
    'nvdiffrast/common/antialias.cu',
    'nvdiffrast/torch/torch_bindings.cpp',
    'nvdiffrast/torch/torch_rasterize.cpp',
    'nvdiffrast/torch/torch_interpolate.cpp',
    'nvdiffrast/torch/torch_texture.cpp',
    'nvdiffrast/torch/torch_antialias.cpp',
]

# Compiler and linker options
extra_compile_args = {
    'cxx': ['-DNVDR_TORCH'],
    'nvcc': ['-DNVDR_TORCH', '-lineinfo'],
}

setup(
    name="nvdiffrast",
    version=nvdiffrast.__version__,
    author="Samuli Laine",
    author_email="slaine@nvidia.com",
    description="nvdiffrast - modular primitives for high-performance differentiable rendering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NVlabs/nvdiffrast",
    packages=find_packages(),
    install_requires=['numpy'], # note: can't require torch here as it will install torch even for a TensorFlow container
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    ext_modules=[
        CUDAExtension(
            name='nvdiffrast_plugin',
            sources=cuda_sources,
            include_dirs=[
                os.path.join(os.path.dirname(__file__), 'nvdiffrast', 'common'),
                os.path.join(os.path.dirname(__file__), 'nvdiffrast', 'torch'),
            ],
            libraries=['cuda', 'nvrtc'],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)
