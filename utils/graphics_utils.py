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

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def is_orthonormal(R, atol=1e-6):
    # Check if the matrix is 3x3
    if R.shape != (3, 3):
        return False
    
    # Check orthogonality: dot product of columns should form the identity matrix
    orthogonal = np.allclose(np.dot(R.T, R), np.eye(3), atol=atol)
    
    # Check normalization: each column should have a norm of 1
    normal = np.allclose(np.linalg.norm(R, axis=0), np.ones(3), atol=atol)
    
    return orthogonal and normal

def correct_rotation(R):
    x = R[:, 0]
    y = R[:, 1]
    error = np.dot(x, y)
    x_ort = x - (error / 2) * y
    y_ort = y - (error / 2) * x
    z_ort = np.cross(x_ort, y_ort)
    x_new = x_ort / np.linalg.norm(x_ort)
    y_new = y_ort / np.linalg.norm(y_ort)
    z_new = z_ort / np.linalg.norm(z_ort)
    
    return np.column_stack((x_new, y_new, z_new))

def find_intersection(P, d):
    n = P.shape[0]
    A = np.zeros((3, 3))
    b = np.zeros(3)

    for i in range(n):
        di = d[i]
        Pi = P[i]
        A += np.eye(3) - np.outer(di, di)
        b += (np.eye(3) - np.outer(di, di)) @ Pi

    p = np.linalg.solve(A, b)
    return p

def get_envmap_dirs(res=[512, 1024]) -> torch.Tensor:
    gy, gx = torch.meshgrid(
        torch.linspace( 0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device="cuda"),
        torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device="cuda"),
        indexing="ij")

    sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
    sinphi, cosphi = torch.sin(gx * np.pi), torch.cos(gx * np.pi)

    reflvec = torch.stack((sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1) # [H, W, 3]
    return reflvec