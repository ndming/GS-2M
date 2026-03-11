import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
from time import perf_counter
from tqdm import tqdm

import os
import json

from fused_bilagrid import _C

from scipy.optimize import least_squares

from typing import Literal, List, Tuple


_CATEGORY_MODE: Literal["each", "version"] = "each"


def _profile_uniform_sample_backward_once(presets):

    X, Y = [], []

    N, m = 1, 1
    for (w, h), (L, H, W), backward_args in tqdm(presets):
        X.append([w, h, L, H, W, *backward_args])

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        bilagrid = torch.randn((N, 12, L, H, W)).cuda()
        rgb = torch.randn((N, m, h, w, 3)).cuda()
        v_output = torch.randn((N, m, h, w, 3)).cuda()

        torch.cuda.synchronize()
        time0 = perf_counter()
        _C.bilagrid_uniform_sample_backward(bilagrid, rgb, v_output, *backward_args)
        torch.cuda.synchronize()
        time1 = perf_counter()

        dt = 1e3 * (time1-time0)
        Y.append(dt)

    return X, Y


def generate_uniform_sample_backward_v1_embeddings(X):
    im_size_nm = np.log(X[:,0] * X[:,1] * (2**-20))
    im_size = 0.5*np.log(X[:,0] * X[:,1])
    bilagrid_depth = np.log(X[:,2])
    bilagrid_size = 0.5*np.log(X[:,3] * X[:,4])
    block_size = 0.5*np.log(X[:,6] * X[:,7])
    cell = np.sqrt(X[:,8])
    return np.stack([
        np.ones_like(im_size),
        im_size_nm, bilagrid_size, bilagrid_depth, block_size, block_size**2, cell, cell**2,
        im_size*bilagrid_size, im_size*bilagrid_depth, im_size*block_size, im_size*block_size**2, im_size*cell, im_size*cell**2,
        bilagrid_size*im_size_nm, bilagrid_size*block_size, bilagrid_size*block_size**2, bilagrid_size*cell, bilagrid_size*cell**2,
    ]).T


def generate_uniform_sample_backward_embeddings(X):
    im_size_nm = np.log(X[:,0] * X[:,1] * (2**-20))
    im_size = 0.5*np.log(X[:,0] * X[:,1])
    bilagrid_depth = np.log(X[:,2])
    bilagrid_size = 0.5*np.log(X[:,3] * X[:,4])
    return np.stack([
        np.ones_like(im_size),
        im_size_nm, bilagrid_size, bilagrid_depth
    ]).T


def map_y(Y):
    return np.log(Y)


def _solver(X, Y, reg=1e-2):
    # print(np.linalg.svd(X)[1])
    # return np.linalg.solve(X.T@X, X.T@Y)

    reg = reg*X.shape[0]/X.shape[1]
    # def resid(p): return np.exp(X.dot(p)) - np.exp(Y)
    # def resid(p): return X.dot(p) - Y
    def resid(p): return np.concatenate([X.dot(p) - Y, reg*p])
    # def resid(p): return (X.dot(p)+np.exp(X.dot(p))) - (Y+np.exp(Y))
    return least_squares(resid, x0=np.zeros(X.shape[1])).x


def _log_time_loss(x, y, y_ref):
    if len(x) == 0:
        return float('nan')
    # plt.figure()
    # plt.scatter(np.exp(x), y)
    # plt.xlim([0, 8])
    # plt.ylim([0, 8])
    # plt.show()
    return np.abs(np.exp(x) - np.exp(y)).mean() / np.exp(y_ref).mean()
    # return np.abs(x - y).mean()

def _train_val_split(X, Y, val=0.1):
    split = int((1-val) * len(X))
    train_x = X[:split]
    val_x = X[split:]
    train_y = Y[:split]
    val_y = Y[split:]
    return (train_x, train_y), (val_x, val_y)


def _train_linear_model(name: str, embeddings, y, val=0.1):
    # print(np.sqrt(np.mean(embeddings**2, axis=0)))

    (train_x, train_y), (val_x, val_y) = _train_val_split(embeddings, y, val)
    print(f"{name}: {len(train_y)} train, {len(val_y)} val")

    params = _solver(train_x, train_y)
    print("params:", params)
    train_loss = _log_time_loss(train_x @ params, train_y, y)
    val_loss = _log_time_loss(val_x @ params, val_y, y)
    print(f"relative error: train={train_loss:.2f}, val={val_loss:.2f}")
    print()

    return params


def fit_model_version(X, Y, val=0.1):
    # columns: w, h, L, H, W, version, block_x, block_y, cell

    version = X[:,5]

    v1_mask = np.where(version == 1)[0]
    v1_embeddings = generate_uniform_sample_backward_v1_embeddings(X[v1_mask])
    v1_y = map_y(Y[v1_mask])

    v2_mask = np.where(version == 2)[0]
    v2_embeddings = generate_uniform_sample_backward_embeddings(X[v2_mask])
    v2_y = map_y(Y[v2_mask])

    v1_params = _train_linear_model(
        'bilagrid_uniform_sample_backward_v1',
        v1_embeddings, v1_y, val
    )

    v2_params = _train_linear_model(
        'bilagrid_uniform_sample_backward_v2',
        v2_embeddings, v2_y, val
    )

    return [
        ('BILAGRID_UNIFORM_SAMPLE_BACKWARD_V1_PARAMS', v1_params),
        ('BILAGRID_UNIFORM_SAMPLE_BACKWARD_V2_PARAMS', v2_params),
    ]


def fit_model_each(X, Y, val=0.1):
    # columns: w, h, L, H, W, version, block_x, block_y, cell

    all_params = []

    presets = generate_uniform_sample_backward_v1_args_presets()
    for preset in presets:
        mask = np.where(np.all(X[:, 5:9] == preset, axis=1))[0]

        embeddings = generate_uniform_sample_backward_embeddings(X[mask])
        y = map_y(Y[mask])

        params = _train_linear_model(preset, embeddings, y, val)

        all_params.append(params)

    all_params = np.array(all_params)

    return [
        ('BILAGRID_UNIFORM_SAMPLE_BACKWARD_PARAMS', all_params),
    ]


def generate_uniform_sample_backward_v1_args_presets():
    backward_args_presets = [
        (2, 0, 0, 0),
    ]
    block_args = [
        (4, 4),
        (8, 4),
        (8, 8),
        (16, 8),
        (16, 16),
    ]
    cell_args = [
        1, 2, 3, 4, 5, 6, 7, 8
    ]
    for block, cell in itertools.product(block_args, cell_args):
        backward_args_presets.append((1, *block, cell))
    return backward_args_presets


def _get_gpu_info():
    def get_command_output(*args):
        import subprocess
        result = subprocess.run([*args], stdout=subprocess.PIPE)
        return result.stdout.decode('utf-8')
    try:
        return '\n'.join([
            get_command_output('nvidia-smi', '-L'),
            get_command_output('nvidia-smi')
        ])
    except:
        return None


def _save_params(params: List[Tuple[str, np.ndarray]]):

    info = _get_gpu_info()

    print("Calibration results saved to:")

    saved = set()
    for file in [__file__, _C.__file__]:
        save_path = os.path.join(os.path.dirname(os.path.abspath(file)), "_calibration_results.py")
        if save_path in saved:
            continue

        with open(save_path, 'w') as fp:
            fp.write(f"import numpy\n\n")
            if info is None:
                fp.write(f'_SYSTEM_INFO = None\n\n')
            else:
                fp.write(f'_SYSTEM_INFO = """\n{info}"""\n\n')

            for varname, weights in params:
                weights = np.round(weights, 8)
                array = json.dumps(weights.tolist(), separators=(',', ':'))
                fp.write(f"# shape: {weights.shape}\n")
                fp.write(f"{varname} = numpy.array({array})\n\n")

        print(save_path)
        saved.add(save_path)


def choose_uniform_sample_backward_args_version(h, w, L, H, W):
    from fused_bilagrid._calibration_results import (
        BILAGRID_UNIFORM_SAMPLE_BACKWARD_V1_PARAMS,
        BILAGRID_UNIFORM_SAMPLE_BACKWARD_V2_PARAMS
    )

    # check v2
    X_v2 = np.array([[w, h, L, H, W]])
    v2_time = generate_uniform_sample_backward_embeddings(X_v2) @ \
        BILAGRID_UNIFORM_SAMPLE_BACKWARD_V2_PARAMS
    v2_time = v2_time[0]

    # check v1
    args = generate_uniform_sample_backward_v1_args_presets()
    args0, args = args[0], args[1:]
    X_v1 = np.concatenate([X_v2.repeat(len(args), 0), args], 1)
    v1_time = generate_uniform_sample_backward_v1_embeddings(X_v1) @ \
        BILAGRID_UNIFORM_SAMPLE_BACKWARD_V1_PARAMS

    # get the best one
    v1_best = np.argmin(v1_time)
    if v2_time < v1_time[v1_best]:
        return args0
    return args[v1_best]


def choose_uniform_sample_backward_args_each(h, w, L, H, W):
    from fused_bilagrid._calibration_results import BILAGRID_UNIFORM_SAMPLE_BACKWARD_PARAMS

    presets = generate_uniform_sample_backward_v1_args_presets()
    
    X = np.array([[w, h, L, H, W]])
    embeds = generate_uniform_sample_backward_embeddings(X)

    log_times = BILAGRID_UNIFORM_SAMPLE_BACKWARD_PARAMS @ embeds.T

    # print(presets[np.argmin(log_times)])

    # plt.figure()
    # plt.plot(np.exp(log_times))
    # plt.show()

    return presets[np.argmin(log_times)]


def choose_uniform_sample_backward_args(h, w, L, H, W, _cache={}):

    key = (h, w, L, H, W)
    if key in _cache:
        return _cache[key]

    if _CATEGORY_MODE == "version":
        best_args = choose_uniform_sample_backward_args_version(h, w, L, H, W)
    elif _CATEGORY_MODE == "each":
        best_args = choose_uniform_sample_backward_args_each(h, w, L, H, W)

    # save cache
    _cache[key] = best_args
    return best_args



if __name__ == "__main__":

    # only w*h matters in embedding
    img_res_presets = [
        (100, 100),
        (160, 160),
        (240, 240),
        (320, 320),
        (480, 480),
        (640, 640),
        (768, 768),
        (960, 960),
        (1080, 1080),
        (1440, 1440),
        (1920, 1920),
        (2400, 2400),
        (3000, 3000),
        (4000, 4000),
    ]

    bilagrid_presets = [
        (12, 24, 24),
        (8, 16, 16),
        (6, 12, 12),
        (4, 8, 8),
        (3, 6, 6),
        (8, 12, 12),
        (4, 12, 12),
        (16, 16, 16),
        (8, 8, 8),
        (4, 4, 4),
    ]

    backward_args_presets = generate_uniform_sample_backward_v1_args_presets()

    random.seed(42)
    all_presets = []
    for img_res, bilagrid_res, backward_args in itertools.product(
            img_res_presets, bilagrid_presets, backward_args_presets[1:]):
        if random.random() < 1.0:  # for speed, depending on generalizability
            all_presets.append((img_res, bilagrid_res, backward_args))
    for img_res, bilagrid_res, backward_args in itertools.product(
            img_res_presets, bilagrid_presets, backward_args_presets[:1]):
        if random.random() < 1.0:
            all_presets.append((img_res, bilagrid_res, backward_args))

    X, Y = [], []

    for i in range(1):
        random.seed(i)

        print("Warming up...")
        presets = all_presets[:20]
        random.shuffle(presets)
        Xi, Yi = _profile_uniform_sample_backward_once(presets)

        print("Profiling...")
        presets = all_presets[:]
        random.shuffle(presets)
        Xi, Yi = _profile_uniform_sample_backward_once(presets)

        X.extend(Xi)
        Y.extend(Yi)
    print()

    X = np.array(X)
    Y = np.array(Y)

    params = sum([
        fit_model_each(X, Y),
        fit_model_version(X, Y),
    ], [])

    _save_params(params)
