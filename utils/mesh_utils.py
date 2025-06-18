import copy
import os
import trimesh

import numpy as np
import open3d as o3d

from tqdm import tqdm
from collections import deque

def post_process_mesh(mesh, cluster_to_keep=1):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    print(f"[>] Post processing mesh to keep {cluster_to_keep} clusters")
    post = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (post.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    post.remove_triangles_by_mask(triangles_to_remove)
    post.remove_unreferenced_vertices()
    post.remove_degenerate_triangles()
    return post

def write_mesh(file, mesh):
    o3d.io.write_triangle_mesh(file, mesh, write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)

def fuse_depths(tsdf_depths, views, render_dir, max_depth, voxel_size, sdf_trunc=-1):
    if sdf_trunc < 0:
        sdf_trunc = 4.0 * voxel_size

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    # tsdf_depths is a (N, H, W) tensor
    for idx, view in enumerate(tqdm(views, desc="[>] TSDF Fusion", ncols=80)):
        ref_depth = tsdf_depths[idx].cuda() # (H, W)
        h, w = ref_depth.shape
        ref_depth[view.alpha_mask.squeeze() < 0.5] = 0

        ref_depth[ref_depth > max_depth] = 0
        ref_depth = ref_depth.cpu().numpy()

        pose = np.identity(4)
        pose[:3,:3] = view.R.transpose(-1,-2)
        pose[:3, 3] = view.T

        image_stem = view.image_name.rsplit('.', 1)[0]
        color = o3d.io.read_image(str(render_dir / f"{image_stem}.png"))
        depth = o3d.geometry.Image((ref_depth * 1000).astype(np.uint16))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1000., depth_trunc=max_depth, convert_rgb_to_intensity=False)
        
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, view.Fx, view.Fy, view.Cx, view.Cy)
        volume.integrate(rgbd, intrinsic, pose)

    return volume

def post_process_mesh_dtu(scene, mesh, source_path):
    print("[>] Post-processing mesh for DTU dataset...")
    train_cameras = scene.getTrainCameras()
    points = []
    for cam in train_cameras:
        c2w = (cam.world_view_transform.T).inverse()
        points.append(c2w[:3,3].cpu().numpy())
    points = np.array(points)

    # Taking the biggest connected component
    mesh_tri = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        vertex_colors=np.asarray(mesh.vertex_colors))
    cleaned_mesh_tri = _find_largest_connected_component(mesh_tri)
    cleaned_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(cleaned_mesh_tri.vertices),
        o3d.utility.Vector3iVector(cleaned_mesh_tri.faces)
    )
    vertex_colors = np.asarray(cleaned_mesh_tri.visual.vertex_colors)[:, :3] / 255.0
    cleaned_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors.astype(np.float64))

    # transform to world
    cam_file = f"{source_path}/cameras.npz"
    scale_mat = np.identity(4)
    if os.path.exists(cam_file):
        camera_param = dict(np.load(cam_file))
        scale_mat = camera_param['scale_mat_0']

    vertices = np.asarray(cleaned_mesh.vertices)
    vertices = vertices * scale_mat[0,0] + scale_mat[:3,3][None]
    cleaned_mesh.vertices = o3d.utility.Vector3dVector(vertices)

    return cleaned_mesh

def _find_largest_connected_component(mesh):
    # 获取顶点和面数据
    faces = mesh.faces
    vertices = mesh.vertices

    # 创建邻接表
    adjacency_list = [[] for _ in range(len(vertices))]
    for face in faces:
        for i in range(3):
            adjacency_list[face[i]].append(face[(i + 1) % 3])
            adjacency_list[face[i]].append(face[(i + 2) % 3])

    # 标记访问过的顶点
    visited = np.zeros(len(vertices), dtype=bool)

    def bfs(start_vertex):
        queue = deque([start_vertex])
        visited[start_vertex] = True
        component = []
        while queue:
            vertex = queue.popleft()
            component.append(vertex)
            for neighbor in adjacency_list[vertex]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        return component

    # 查找所有连接组件并记录大小
    largest_component = []
    for vertex in range(len(vertices)):
        if not visited[vertex]:
            component = bfs(vertex)
            if len(component) > len(largest_component):
                largest_component = component

    # 提取最大连接组件的顶点和面
    largest_component = set(largest_component)
    component_faces = [face for face in faces if all(v in largest_component for v in face)]
    vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(largest_component)}
    component_vertices = vertices[list(largest_component)]
    component_faces = np.array([[vertex_map[v] for v in face] for face in component_faces])

    # 创建新的网格
    largest_component_mesh = trimesh.Trimesh(vertices=component_vertices, faces=component_faces)
    return largest_component_mesh