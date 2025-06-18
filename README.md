# DGSR: Deferred 3D Gaussian Splatting for Accurate Surface Reconstruction

## Build instructions

### Build the CUDA rasterizer with Clang
The CUDA rasterizer can be compiled with Clang for better development tooling, i.e. code completion and diagnostics via `ClangD`.

###### Windows
- CMake version `3.29`
- Ninja
- Visual Studio [BuildTools](https://learn.microsoft.com/en-us/visualstudio/releases/2022/release-history) version `17.9.7`
(only MSVC v143 and Windows SDK are required)
- [Clang-CL](https://github.com/llvm/llvm-project/releases/tag/llvmorg-19.1.7) version `19.1.7`
- Add `path\to\BuildTools\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64` to `PATH`
- Add `C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64` to `PATH`
- Download `libtorch` for [Windows](https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-2.0.1%2Bcu118.zip)
- Create a `config.yaml` file under `%LocalAppData%\clangd` with the following entries:
```yaml
CompileFlags:
  Add:
    - --cuda-path=path/to/conda/prefix
    - -Ipath/to/conda/prefix/include
    - -Ipath/to/libtorch/include
    - -Ipath/to/libtorch/include/torch/csrc/api/include
  Remove:
    - -forward-unknown-to-host-compiler
    - --generate-code*
    - -Xcompiler=*
```

Prior to CMake configuration, the `LIB` environment variable must be set to the locations containing system libraries.
These locations can be found by checking the `LIB` variable in a VS 2022 Developer Command Prompt/PowerShell.

Tell CMake where to find the CUDA compiler by setting `CMAKE_PREFIX_PATH` to the `CONDA_PREFIX` variable:
```
cd submodules/diff-deferred-rasterization
cmake -B build -G Ninja -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_PREFIX_PATH=path/to/conda/prefix
``` 

###### Linux
- CMake version `3.29`
- Ninja
- [Clang](https://github.com/llvm/llvm-project/releases/tag/llvmorg-18.1.8) version `18.1.8`
- Download `libtorch` for [Linux](https://download.pytorch.org/libtorch/cu118/libtorch-shared-with-deps-2.0.1%2Bcu118.zip)
- Create a `config.yaml` file under `~/.config/clangd` with the following entries:
```yaml
CompileFlags:
  Add:
    - -stdlib=libc++
    - -lc++abi
    - -Ipath/to/conda/prefix/include/python3.9
    - -Ipath/to/libtorch/include
    - -Ipath/to/libtorch/include/torch/csrc/api/include
  Remove:
    - -forward-unknown-to-host-compiler
    - --generate-code*
```

Tell CMake where to find the CUDA compiler by setting `CMAKE_PREFIX_PATH` to the `CONDA_PREFIX` variable:
```
cd submodules/diff-deferred-rasterization
cmake -B build -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_PREFIX_PATH=path/to/conda/prefix
```