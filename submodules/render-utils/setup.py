from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

# Compiler/linker configuration
extra_compile_args = {
    'cxx': ['-DNVDR_TORCH'],
    'nvcc': ['-DNVDR_TORCH']
}

extra_link_args = []
if os.name == 'nt':
    extra_link_args += ['advapi32.lib']

# Source files (same structure as JIT version)
source_files = [
    'c_src/mesh.cu',
    'c_src/loss.cu',
    'c_src/bsdf.cu',
    'c_src/normal.cu',
    'c_src/cubemap.cu',
    'c_src/common.cpp',
    'c_src/torch_bindings.cpp'
]

setup(
    name='render_utils',
    packages=['render_utils'],
    ext_modules=[
        CUDAExtension(
            name='renderutils_plugin',
            sources=source_files,
            libraries=['cuda', 'nvrtc'],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)
