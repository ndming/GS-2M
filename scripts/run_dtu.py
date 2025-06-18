import os

scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
data_base_path='/home/zodnguy1/datasets/dtu'
out_base_path='output/dtu'

lambda_tv = 0.5
angle_factor = 1.0
geo_from = 7000
label = f'tv-{lambda_tv}_ang-{angle_factor}_g7k'

for scene in scenes:
    common_args = f"-r 2 --lambda_tv_normal {lambda_tv} --mv_angle_factor {angle_factor} --geometry_from_iter {geo_from}"
    cmd = f'python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = f"--dtu --label {label}"
    cmd = f'python render.py -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    cmd = f"python scripts/eval_dtu/evaluate_single_scene.py " + \
          f"--input_ply {out_base_path}/scan{scene}/train/{label}_30000/meshes/tsdf_post.ply " + \
          f"--ref_dir {data_base_path}/scan{scene} " + \
          f"--dtu_dir {data_base_path}/Official_DTU_Dataset"
    print("[>] " + cmd)
    os.system(cmd)
    print(f"==> Done with scene: scan{scene} <===\n")

lambda_tv = 0.5
angle_factor = 0.5
geo_from = 7000
label = f'tv-{lambda_tv}_ang-{angle_factor}_g7k'

for scene in scenes:
    common_args = f"-r 2 --lambda_tv_normal {lambda_tv} --mv_angle_factor {angle_factor} --geometry_from_iter {geo_from}"
    cmd = f'python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = f"--dtu --label {label}"
    cmd = f'python render.py -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    cmd = f"python scripts/eval_dtu/evaluate_single_scene.py " + \
          f"--input_ply {out_base_path}/scan{scene}/train/{label}_30000/meshes/tsdf_post.ply " + \
          f"--ref_dir {data_base_path}/scan{scene} " + \
          f"--dtu_dir {data_base_path}/Official_DTU_Dataset"
    print("[>] " + cmd)
    os.system(cmd)
    print(f"==> Done with scene: scan{scene} <===\n")


lambda_tv = 0.1
angle_factor = 1.0
geo_from = 7000
label = f'tv-{lambda_tv}_ang-{angle_factor}_g7k'

for scene in scenes:
    common_args = f"-r 2 --lambda_tv_normal {lambda_tv} --mv_angle_factor {angle_factor} --geometry_from_iter {geo_from}"
    cmd = f'python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = f"--dtu --label {label}"
    cmd = f'python render.py -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    cmd = f"python scripts/eval_dtu/evaluate_single_scene.py " + \
          f"--input_ply {out_base_path}/scan{scene}/train/{label}_30000/meshes/tsdf_post.ply " + \
          f"--ref_dir {data_base_path}/scan{scene} " + \
          f"--dtu_dir {data_base_path}/Official_DTU_Dataset"
    print("[>] " + cmd)
    os.system(cmd)
    print(f"==> Done with scene: scan{scene} <===\n")


lambda_tv = 0.1
angle_factor = 0.5
geo_from = 7000
label = f'tv-{lambda_tv}_ang-{angle_factor}_g7k'

for scene in scenes:
    common_args = f"-r 2 --lambda_tv_normal {lambda_tv} --mv_angle_factor {angle_factor} --geometry_from_iter {geo_from}"
    cmd = f'python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = f"--dtu --label {label}"
    cmd = f'python render.py -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    cmd = f"python scripts/eval_dtu/evaluate_single_scene.py " + \
          f"--input_ply {out_base_path}/scan{scene}/train/{label}_30000/meshes/tsdf_post.ply " + \
          f"--ref_dir {data_base_path}/scan{scene} " + \
          f"--dtu_dir {data_base_path}/Official_DTU_Dataset"
    print("[>] " + cmd)
    os.system(cmd)
    print(f"==> Done with scene: scan{scene} <===\n")


lambda_tv = 0.5
angle_factor = 0.5
geo_from = 5000
label = f'tv-{lambda_tv}_ang-{angle_factor}_g5k'

for scene in scenes:
    common_args = f"-r 2 --lambda_tv_normal {lambda_tv} --mv_angle_factor {angle_factor} --geometry_from_iter {geo_from}"
    cmd = f'python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = f"--dtu --label {label}"
    cmd = f'python render.py -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    cmd = f"python scripts/eval_dtu/evaluate_single_scene.py " + \
          f"--input_ply {out_base_path}/scan{scene}/train/{label}_30000/meshes/tsdf_post.ply " + \
          f"--ref_dir {data_base_path}/scan{scene} " + \
          f"--dtu_dir {data_base_path}/Official_DTU_Dataset"
    print("[>] " + cmd)
    os.system(cmd)
    print(f"==> Done with scene: scan{scene} <===\n")


lambda_tv = 0.1
angle_factor = 0.5
geo_from = 5000
label = f'tv-{lambda_tv}_ang-{angle_factor}_g5k'

for scene in scenes:
    common_args = f"-r 2 --lambda_tv_normal {lambda_tv} --mv_angle_factor {angle_factor} --geometry_from_iter {geo_from}"
    cmd = f'python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = f"--dtu --label {label}"
    cmd = f'python render.py -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    cmd = f"python scripts/eval_dtu/evaluate_single_scene.py " + \
          f"--input_ply {out_base_path}/scan{scene}/train/{label}_30000/meshes/tsdf_post.ply " + \
          f"--ref_dir {data_base_path}/scan{scene} " + \
          f"--dtu_dir {data_base_path}/Official_DTU_Dataset"
    print("[>] " + cmd)
    os.system(cmd)
    print(f"==> Done with scene: scan{scene} <===\n")


lambda_tv = 0.1
angle_factor = 0.5
geo_from = 5000
mat_from = 15_000
label = f'tv-{lambda_tv}_ang-{angle_factor}_g5k_m15k'

for scene in scenes:
    common_args = f"-r 2 --lambda_tv_normal {lambda_tv} --mv_angle_factor {angle_factor} --geometry_from_iter {geo_from} --material_from_iter {mat_from}"
    cmd = f'python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = f"--dtu --label {label}"
    cmd = f'python render.py -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    cmd = f"python scripts/eval_dtu/evaluate_single_scene.py " + \
          f"--input_ply {out_base_path}/scan{scene}/train/{label}_30000/meshes/tsdf_post.ply " + \
          f"--ref_dir {data_base_path}/scan{scene} " + \
          f"--dtu_dir {data_base_path}/Official_DTU_Dataset"
    print("[>] " + cmd)
    os.system(cmd)
    print(f"==> Done with scene: scan{scene} <===\n")


lambda_tv = 0.1
angle_factor = 0.5
geo_from = 7000
mat_from = 15_000
label = f'tv-{lambda_tv}_ang-{angle_factor}_g7k_m15k'

for scene in scenes:
    common_args = f"-r 2 --lambda_tv_normal {lambda_tv} --mv_angle_factor {angle_factor} --geometry_from_iter {geo_from} --material_from_iter {mat_from}"
    cmd = f'python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = f"--dtu --label {label}"
    cmd = f'python render.py -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    cmd = f"python scripts/eval_dtu/evaluate_single_scene.py " + \
          f"--input_ply {out_base_path}/scan{scene}/train/{label}_30000/meshes/tsdf_post.ply " + \
          f"--ref_dir {data_base_path}/scan{scene} " + \
          f"--dtu_dir {data_base_path}/Official_DTU_Dataset"
    print("[>] " + cmd)
    os.system(cmd)
    print(f"==> Done with scene: scan{scene} <===\n")

lambda_tv = 0.1
angle_factor = 0.5
geo_from = 7000
mat_from = 20_000
label = f'tv-{lambda_tv}_ang-{angle_factor}_g7k_m20k'

for scene in scenes:
    common_args = f"-r 2 --lambda_tv_normal {lambda_tv} --mv_angle_factor {angle_factor} --geometry_from_iter {geo_from} --material_from_iter {mat_from}"
    cmd = f'python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = f"--dtu --label {label}"
    cmd = f'python render.py -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    cmd = f"python scripts/eval_dtu/evaluate_single_scene.py " + \
          f"--input_ply {out_base_path}/scan{scene}/train/{label}_30000/meshes/tsdf_post.ply " + \
          f"--ref_dir {data_base_path}/scan{scene} " + \
          f"--dtu_dir {data_base_path}/Official_DTU_Dataset"
    print("[>] " + cmd)
    os.system(cmd)
    print(f"==> Done with scene: scan{scene} <===\n")


lambda_tv = 0.1
angle_factor = 0.5
geo_from = 5000
mat_from = 20_000
label = f'tv-{lambda_tv}_ang-{angle_factor}_g5k_m20k'

for scene in scenes:
    common_args = f"-r 2 --lambda_tv_normal {lambda_tv} --mv_angle_factor {angle_factor} --geometry_from_iter {geo_from} --material_from_iter {mat_from}"
    cmd = f'python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = f"--dtu --label {label}"
    cmd = f'python render.py -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    cmd = f"python scripts/eval_dtu/evaluate_single_scene.py " + \
          f"--input_ply {out_base_path}/scan{scene}/train/{label}_30000/meshes/tsdf_post.ply " + \
          f"--ref_dir {data_base_path}/scan{scene} " + \
          f"--dtu_dir {data_base_path}/Official_DTU_Dataset"
    print("[>] " + cmd)
    os.system(cmd)
    print(f"==> Done with scene: scan{scene} <===\n")

lambda_tv = 0.5
angle_factor = 0.5
geo_from = 5000
mat_from = 15_000
label = f'tv-{lambda_tv}_ang-{angle_factor}_g5k_m15k'

for scene in scenes:
    common_args = f"-r 2 --lambda_tv_normal {lambda_tv} --mv_angle_factor {angle_factor} --geometry_from_iter {geo_from} --material_from_iter {mat_from}"
    cmd = f'python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = f"--dtu --label {label}"
    cmd = f'python render.py -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    cmd = f"python scripts/eval_dtu/evaluate_single_scene.py " + \
          f"--input_ply {out_base_path}/scan{scene}/train/{label}_30000/meshes/tsdf_post.ply " + \
          f"--ref_dir {data_base_path}/scan{scene} " + \
          f"--dtu_dir {data_base_path}/Official_DTU_Dataset"
    print("[>] " + cmd)
    os.system(cmd)
    print(f"==> Done with scene: scan{scene} <===\n")


lambda_tv = 0.5
angle_factor = 0.5
geo_from = 7000
mat_from = 15_000
label = f'tv-{lambda_tv}_ang-{angle_factor}_g7k_m15k'

for scene in scenes:
    common_args = f"-r 2 --lambda_tv_normal {lambda_tv} --mv_angle_factor {angle_factor} --geometry_from_iter {geo_from} --material_from_iter {mat_from}"
    cmd = f'python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = f"--dtu --label {label}"
    cmd = f'python render.py -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    cmd = f"python scripts/eval_dtu/evaluate_single_scene.py " + \
          f"--input_ply {out_base_path}/scan{scene}/train/{label}_30000/meshes/tsdf_post.ply " + \
          f"--ref_dir {data_base_path}/scan{scene} " + \
          f"--dtu_dir {data_base_path}/Official_DTU_Dataset"
    print("[>] " + cmd)
    os.system(cmd)
    print(f"==> Done with scene: scan{scene} <===\n")
