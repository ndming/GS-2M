import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData

from .base import ExportableModel
from .exporter import NuRecExporter, _get_default_nurec_conf as default_nurec_config


class GaussianPLYModel(ExportableModel):

    def __init__(self, ply_path: str, max_sh_degree: int = 3):
        plydata = PlyData.read(ply_path)
        el = plydata.elements[0]
        N = len(el["x"])

        # Positions
        positions = np.stack(
            [el["x"], el["y"], el["z"]], axis=1
        ).astype(np.float32)

        # Density — stored as logit (pre-sigmoid) in standard 3DGS
        densities = np.asarray(
            el["opacity"], dtype=np.float32
        )[..., np.newaxis]

        # Albedo — DC spherical harmonic coefficients
        albedo = np.stack(
            [el["f_dc_0"], el["f_dc_1"], el["f_dc_2"]], axis=1
        ).astype(np.float32)

        # Specular — higher-order SH coefficients
        # PLY layout: channel-first [R₀..Rₖ, G₀..Gₖ, B₀..Bₖ]
        # NuRec expects: coeff-first interleaved [C₀R,C₀G,C₀B, C₁R,C₁G,C₁B,...]
        num_speculars = (max_sh_degree + 1) ** 2 - 1
        expected = 3 * num_speculars
        rest_names = sorted(
            [p.name for p in el.properties if p.name.startswith("f_rest_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        specular = np.zeros((N, expected), dtype=np.float32)
        if len(rest_names) == expected:
            raw = np.stack(
                [el[n] for n in rest_names], axis=1
            )                                               # (N, 3*K)
            raw = raw.reshape(N, 3, num_speculars)          # (N, 3, K)
            raw = raw.transpose(0, 2, 1)                    # (N, K, 3)
            specular = raw.reshape(N, expected)             # (N, K*3)

        # Scales — stored as log-scale in standard 3DGS
        scale_names = sorted(
            [p.name for p in el.properties if p.name.startswith("scale_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        scales = np.stack(
            [el[n] for n in scale_names], axis=1
        ).astype(np.float32)

        # Rotations — stored as raw quaternion (w, x, y, z)
        rot_names = sorted(
            [p.name for p in el.properties if p.name.startswith("rot")],
            key=lambda x: int(x.split("_")[-1]),
        )
        rotations = np.stack(
            [el[n] for n in rot_names], axis=1
        ).astype(np.float32)

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._positions  = torch.tensor(positions,  device=dev)
        self._densities  = torch.tensor(densities,  device=dev)
        self._albedo     = torch.tensor(albedo,     device=dev)
        self._specular   = torch.tensor(specular,   device=dev)
        self._scales     = torch.tensor(scales,     device=dev)
        self._rotations  = torch.tensor(rotations,  device=dev)
        self._max_sh     = max_sh_degree
        self._n_active   = max_sh_degree

    def get_positions(self):          return self._positions
    def get_max_n_features(self):     return self._max_sh
    def get_n_active_features(self):  return self._n_active
    def get_features_albedo(self):    return self._albedo
    def get_features_specular(self):  return self._specular

    def get_scale(self, preactivation=False):
        # Standard 3DGS stores log-scale; exp gives actual scale
        return self._scales if preactivation else torch.exp(self._scales)

    def get_rotation(self, preactivation=False):
        return self._rotations if preactivation else F.normalize(self._rotations, p=2, dim=-1)

    def get_density(self, preactivation=False):
        # Standard 3DGS stores logit; sigmoid gives actual opacity
        return self._densities if preactivation else torch.sigmoid(self._densities)
