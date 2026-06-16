#include <torch/extension.h>
#include "patch_ncc.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("warp_patch_ncc", &WarpPatchNCC);
}