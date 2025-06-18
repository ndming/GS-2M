import cv2
import glob
import os
import torch
import trimesh

import numpy as np
import render_utils as rend_util
import torch.nn.functional as F

from argparse import ArgumentParser
from pathlib import Path
from skimage.morphology import binary_dilation, disk
from tqdm import tqdm

def cull_mesh(file, cull_out_file, ref_dir, mask_cull=False):
    n_images = len(list((ref_dir / "images").glob("*.png")))
    cam_file = ref_dir / "cameras.npz"
    cam_dict = np.load(cam_file)
    scale_mats = [cam_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    world_mats = [cam_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

    # Load mesh
    mesh = trimesh.load(file)

    # Don't cull the mesh against masks, transform vertices to world and return
    if not mask_cull:
        print(f"[>] Mask culling - skipped")
        scale_mat = scale_mats[0]
        mesh.vertices = mesh.vertices * scale_mat[0, 0] + scale_mat[:3, 3][None]
        mesh.export(cull_out_file)
        del mesh
        return

    intrinsics_all = []
    pose_all = []
    for scale_mat, world_mat in zip(scale_mats, world_mats):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
        intrinsics_all.append(torch.from_numpy(intrinsics).float())
        pose_all.append(torch.from_numpy(pose).float())

    # Load mask
    mask_dir = ref_dir / "mask"
    mask_paths = sorted(glob.glob(str(mask_dir / "*.png")))
    masks = []
    for path in mask_paths:
        mask = cv2.imread(path)
        masks.append(mask)

    # Hard-coded image shape
    W, H = 1600, 1200

    # Project and filter
    vertices = mesh.vertices # (N, 3)
    vertices = torch.from_numpy(vertices).cuda()
    vertices = torch.cat((vertices, torch.ones_like(vertices[:, :1])), dim=-1) # (N, 4)
    vertices = vertices.permute(1, 0).float() # (4, N)

    sampled_masks = []
    for i in tqdm(range(n_images), desc="[>] Mask culling", ncols=80):
        pose = pose_all[i]
        w2c = torch.inverse(pose).cuda()
        intrinsic = intrinsics_all[i].cuda()

        with torch.no_grad():
            # Transform and project
            cam_points = intrinsic @ w2c @ vertices
            pix_coords = cam_points[:2, :] / (cam_points[2, :].unsqueeze(0) + 1e-6)
            pix_coords = pix_coords.permute(1, 0)
            pix_coords[..., 0] /= W - 1
            pix_coords[..., 1] /= H - 1
            pix_coords = (pix_coords - 0.5) * 2
            valid = ((pix_coords > -1.) & (pix_coords < 1.)).all(dim=-1).float()
            
            # Dialate mask similar to unisurf
            maski = masks[i][:, :, 0].astype(np.float32) / 256.
            maski = torch.from_numpy(binary_dilation(maski, disk(24))).float()[None, None].cuda()

            sampled_mask = F.grid_sample(maski, pix_coords[None, None], mode='nearest', padding_mode='zeros', align_corners=True)[0, -1, 0]

            sampled_mask = sampled_mask + (1. - valid)
            sampled_masks.append(sampled_mask)

    # Filter
    sampled_masks = torch.stack(sampled_masks, -1)
    mask = (sampled_masks > 0.).all(dim=-1).cpu().numpy()
    face_mask = mask[mesh.faces].all(axis=1)

    mesh.update_vertices(mask)
    mesh.update_faces(face_mask)

    # Transform vertices to world 
    scale_mat = scale_mats[0]
    mesh.vertices = mesh.vertices * scale_mat[0, 0] + scale_mat[:3, 3][None]
    mesh.export(cull_out_file)
    del mesh

if __name__ == "__main__":
    parser = ArgumentParser(description='Arguments to evaluate the reconstructed triangle mesh.')

    parser.add_argument('--input_ply', type=str, required=True, help='path to the mesh to be evaluated')
    parser.add_argument('--mask_cull', action='store_true', help='whether to use mask culling')
    parser.add_argument('--ref_dir', type=str, required=True, help='path to the reference scan folder')
    parser.add_argument('--dtu_dir', type=str, default='Offical_DTU_Dataset', help='path to the DTU GT directory')
    parser.add_argument('--out_dir', type=str, default='', help='path to the output dir, default to input_ply\'s dir')

    args = parser.parse_args()

    ref_dir = Path(args.ref_dir)
    scan_id = int(ref_dir.name.replace("scan", ""))

    input_ply = Path(args.input_ply)
    out_dir = input_ply.parent if not args.out_dir else Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    cull_out_file = out_dir / "culled_mesh.ply"
    cull_mesh(Path(args.input_ply), cull_out_file, ref_dir, args.mask_cull)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = f"python {script_dir}/eval.py --data {cull_out_file} --scan {scan_id} --mode mesh " + \
          f"--dataset_dir {args.dtu_dir} --vis_out_dir {out_dir}"
    os.system(cmd)

