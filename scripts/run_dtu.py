import os

scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
data_base_path='/home/zodnguy1/datasets/dtu'
out_base_path='output/proto/dtu'

lambda_mv = 0.0
label = f'mv-{lambda_mv}_metallic_mat15k_iter30k'

for scene in scenes:
    common_args = f"-r 2 --lambda_multi_view {lambda_mv} --metallic"
    cmd = f'python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = f"--num_cluster 1 --voxel_size 0.002 --max_depth 5.0 --extract_mesh --skip_test --label {label}"
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

lambda_mv = 0.0
label = f'mv-{lambda_mv}_metallic_mat15k_iter40k'

for scene in scenes:
    common_args = f"-r 2 --lambda_multi_view {lambda_mv} --metallic --iterations 40000"
    cmd = f'python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = f"--num_cluster 1 --voxel_size 0.002 --max_depth 5.0 --extract_mesh --skip_test --label {label}"
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

lambda_mv = 0.0
label = f'mv-{lambda_mv}_metallic_mat30k_iter40k'

for scene in scenes:
    common_args = f"-r 2 --lambda_multi_view {lambda_mv} --metallic --iterations 40000 --material_from_iter 30000"
    cmd = f'python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = f"--num_cluster 1 --voxel_size 0.002 --max_depth 5.0 --extract_mesh --skip_test --label {label}"
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

lambda_mv = 0.0
label = f'mv-{lambda_mv}_metallic_mat30k_iter45k'

for scene in scenes:
    common_args = f"-r 2 --lambda_multi_view {lambda_mv} --metallic --iterations 45000 --material_from_iter 30000"
    cmd = f'python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = f"--num_cluster 1 --voxel_size 0.002 --max_depth 5.0 --extract_mesh --skip_test --label {label}"
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

lambda_mv = 0.0
label = f'mv-{lambda_mv}_non-metal_mat15k_iter30k'

for scene in scenes:
    common_args = f"-r 2 --lambda_multi_view {lambda_mv}"
    cmd = f'python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = f"--num_cluster 1 --voxel_size 0.002 --max_depth 5.0 --extract_mesh --skip_test --label {label}"
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

lambda_mv = 0.0
label = f'mv-{lambda_mv}_non-metal_mat15k_iter40k'

for scene in scenes:
    common_args = f"-r 2 --lambda_multi_view {lambda_mv} --iterations 40000"
    cmd = f'python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = f"--num_cluster 1 --voxel_size 0.002 --max_depth 5.0 --extract_mesh --skip_test --label {label}"
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

lambda_mv = 0.0
label = f'mv-{lambda_mv}_non-metal_mat30k_iter40k'

for scene in scenes:
    common_args = f"-r 2 --lambda_multi_view {lambda_mv} --iterations 40000 --material_from_iter 30000"
    cmd = f'python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = f"--num_cluster 1 --voxel_size 0.002 --max_depth 5.0 --extract_mesh --skip_test --label {label}"
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

lambda_mv = 0.0
label = f'mv-{lambda_mv}_non-metal_mat30k_iter45k'

for scene in scenes:
    common_args = f"-r 2 --lambda_multi_view {lambda_mv} --iterations 45000 --material_from_iter 30000"
    cmd = f'python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = f"--num_cluster 1 --voxel_size 0.002 --max_depth 5.0 --extract_mesh --skip_test --label {label}"
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

lambda_mv = 0.01
label = f'mv-{lambda_mv}_metallic_mat15k_iter30k'

for scene in scenes:
    common_args = f"-r 2 --lambda_multi_view {lambda_mv} --metallic"
    cmd = f'python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = f"--num_cluster 1 --voxel_size 0.002 --max_depth 5.0 --extract_mesh --skip_test --label {label}"
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

lambda_mv = 0.05
label = f'mv-{lambda_mv}_metallic_mat15k_iter30k'

for scene in scenes:
    common_args = f"-r 2 --lambda_multi_view {lambda_mv} --metallic"
    cmd = f'python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = f"--num_cluster 1 --voxel_size 0.002 --max_depth 5.0 --extract_mesh --skip_test --label {label}"
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

lambda_mv = 0.005
label = f'mv-{lambda_mv}_metallic_mat15k_iter30k'

for scene in scenes:
    common_args = f"-r 2 --lambda_multi_view {lambda_mv} --metallic"
    cmd = f'python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = f"--num_cluster 1 --voxel_size 0.002 --max_depth 5.0 --extract_mesh --skip_test --label {label}"
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

lambda_mv = 0.0
lambda_sm = 0.1
label = f'sm-{lambda_sm}_mv-{lambda_mv}_metallic_mat15k_iter30k'

for scene in scenes:
    common_args = f"-r 2 --lambda_multi_view {lambda_mv} --metallic"
    cmd = f'python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = f"--num_cluster 1 --voxel_size 0.002 --max_depth 5.0 --extract_mesh --skip_test --label {label}"
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

lambda_mv = 0.0
lambda_sm = 0.5
label = f'sm-{lambda_sm}_mv-{lambda_mv}_metallic_mat15k_iter30k'

for scene in scenes:
    common_args = f"-r 2 --lambda_multi_view {lambda_mv} --metallic"
    cmd = f'python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/scan{scene} {common_args}'
    print("[>] " + cmd)
    os.system(cmd)

    common_args = f"--num_cluster 1 --voxel_size 0.002 --max_depth 5.0 --extract_mesh --skip_test --label {label}"
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