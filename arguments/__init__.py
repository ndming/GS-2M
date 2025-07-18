#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        # GS-IR
        self.gamma = False
        self.metallic = False
        # Ours
        self.material = False
        self.mask_gt = False # whether to mask GT images during training
        self.masks = "" # foreground masks directory name, empty if not used
        self.depths = "" # depths directory name, empty if not used
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.z_depth = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.001
        self.lambda_ssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.opacity_prune_threshold = 0.005
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        # AbsGS
        self.densify_grad_abs_threshold = 0.0008
        self.use_opacity_reduce = False
        self.opacity_reduce_interval = 500
        self.prune_init_points = True
        self.radii2D_threshold = 20
        # PGSR
        self.multi_view_num = 8
        self.multi_view_ncc_weight = 0.15
        self.multi_view_geo_weight = 0.03
        self.multi_view_ncc_scale = -1.0
        self.multi_view_max_angle = 30
        self.multi_view_min_dist = 0.01
        self.multi_view_max_dist = 1.5
        self.use_multi_view_trim = True
        self.multi_view_sample_num = 102400
        self.multi_view_patch_size = 3
        # Ours
        self.geometry_from_iter = 5000
        self.material_from_iter = 30_000
        self.lambda_alpha = 0.2
        self.lambda_plane = 100.0
        self.lambda_depth_normal = 0.015
        self.lambda_multi_view = 1.0
        self.lambda_normal = 2.0
        self.lambda_smooth = 0.01
        self.lambda_rough = 1e-4
        self.mv_angle_threshold = 30
        self.mv_angle_factor = 2.0
        self.mv_occlusion_threshold = 5e-4
        self.mv_geo_weight_decay = 3.0
        self.reflection_threshold = 1.0
        self.nearby_cam_num = 16
        self.nearby_cam_max_angle = 60
        self.nearby_cam_min_angle = 10
        self.nearby_cam_min_dist = 0.05
        self.nearby_cam_max_dist = 2.5

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        with open(cfgfilepath) as cfg_file:
            print(f"[>] Found config file: {cfgfilepath}")
            cfgfile_string = cfg_file.read()
    except TypeError:
        print(f"[!] Config file not found at: {cfgfilepath}")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)