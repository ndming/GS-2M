import os
import json
import bpy
import math
import mathutils

from argparse import ArgumentParser
from pathlib import Path

BG_K = ()

def search_for_max_iter(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)

def prepare_blender_scene(ply_file, ref_cam_info):
    # Delete all existing objects in the scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Load the mesh
    bpy.ops.wm.ply_import(filepath=str(ply_file))
    mesh = bpy.context.selected_objects[0]

    # Material
    mat = bpy.data.materials.new(name="DiffuseMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Metallic'].default_value = 0.0

    # Assign material to mesh
    if mesh.data.materials:
        mesh.data.materials[0] = mat
    else:
        mesh.data.materials.append(mat)

    w = ref_cam_info["width"]
    h = ref_cam_info["height"]

    # Configure render settings
    scene = bpy.context.scene
    # scene.render.engine = 'BLENDER_EEVEE_NEXT'
    # scene.eevee.use_gtao = True
    scene.render.engine = 'CYCLES'
    prefs = bpy.context.preferences
    cycles_prefs = prefs.addons['cycles'].preferences
    cycles_prefs.compute_device_type = 'OPTIX' # or 'CUDA', 'HIP', 'METAL'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = 512
    scene.cycles.noise_threshold = 0.01
    scene.cycles.use_denoising = True

    # Background
    if not bpy.data.worlds:
        bpy.ops.world.new()
    world = bpy.data.worlds[0]
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    bg_light = nodes.new(type='ShaderNodeBackground')
    bg_light.inputs[0].default_value = (0, 0, 0, 1)
    bg_node = nodes.new(type='ShaderNodeBackground')
    mix_shader = nodes.new(type='ShaderNodeMixShader')
    light_path = nodes.new(type='ShaderNodeLightPath')
    world_output = nodes.new(type='ShaderNodeOutputWorld')
    links.new(light_path.outputs['Is Camera Ray'], mix_shader.inputs['Fac'])
    links.new(bg_light.outputs['Background'], mix_shader.inputs[1])  # Lighting
    links.new(bg_node.outputs['Background'], mix_shader.inputs[2])  # Visible BG
    links.new(mix_shader.outputs['Shader'], world_output.inputs['Surface'])

    # Read ref camera info
    R = ref_cam_info["rotation"]
    T = ref_cam_info["position"]
    c2w = mathutils.Matrix(R).to_4x4()
    c2w.translation = mathutils.Vector(T)
    c2w = c2w @ mathutils.Matrix([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    sensor_w = 36.0 # mm (default in Blender)
    sensor_h = 24.0 # mm (default in Blender)

    fx = ref_cam_info["fx"] * sensor_w / w
    fy = ref_cam_info["fy"] * sensor_h / h

    # Camera
    camera_data = bpy.data.cameras.new(name='Camera')
    camera = bpy.data.objects.new('Camera', camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera

    # Camera intrinsics
    camera.data.lens = (fx + fy) / 2.0
    if (fx / fy) > (sensor_w / sensor_h):
        camera.data.sensor_fit = 'HORIZONTAL'
    else:
        camera.data.sensor_fit = 'VERTICAL'

    camera_z_axis = c2w.to_3x3() @ mathutils.Vector((0, 0, 1))
    rotation_quat = mathutils.Vector((0, 0, 1)).rotation_difference(camera_z_axis)

    # Add an area light with direction following the camera's z-axis
    area_data = bpy.data.lights.new(name="Area", type='AREA')
    area_object = bpy.data.objects.new(name="Area", object_data=area_data)
    bpy.context.scene.collection.objects.link(area_object)
    area_data.shape = 'DISK'
    area_data.size = 5.0
    area_object.rotation_mode = 'QUATERNION'
    area_object.rotation_quaternion = rotation_quat
    area_object.location = camera.location + camera_z_axis * 2.0

    return scene, camera, mesh, mat, bg_node, area_data

def export_anim(scene, mesh, mp4_file):
    scene.render.resolution_x = 600
    scene.render.resolution_y = 600
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.filepath = str(mp4_file)

    num_frames = 120
    scene.frame_start = 1
    scene.frame_end = num_frames
    scene.frame_current = 1

    # Create rotation animation for object
    mesh.rotation_mode = 'XYZ'
    start_rot = mesh.rotation_euler.copy()
    for frame in range(1, num_frames + 1):
        scene.frame_set(frame)

        # Oscillate rotation using sine waves
        angle_x = 0.4 * math.sin(2 * math.pi * frame / num_frames * 1)
        angle_y = 0.6 * math.sin(2 * math.pi * frame / num_frames * 2)

        mesh.rotation_euler = start_rot.copy()
        mesh.rotation_euler.x += angle_x
        mesh.rotation_euler.y += angle_y
        
        # Keyframe both axes
        mesh.keyframe_insert(data_path="rotation_euler", index=0)  # X-axis
        mesh.keyframe_insert(data_path="rotation_euler", index=1)  # Y-axis

    # Render animation directly to MP4
    bpy.ops.render.render(animation=True, write_still=False)

def render_images(scene, camera, out_dir, views, resolution_scale, view_idx):
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'

    if view_idx >= 0:
        views = [views[view_idx]]

    for cam_info in views:
        stem = cam_info["img_name"].split(".")[0]
        w = cam_info["width"]
        h = cam_info["height"]

        R = cam_info["rotation"]
        T = cam_info["position"]
        c2w = mathutils.Matrix(R).to_4x4()
        c2w.translation = mathutils.Vector(T)
        c2w = c2w @ mathutils.Matrix([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        camera.matrix_world = c2w

        scene.render.resolution_x = w // resolution_scale
        scene.render.resolution_y = h // resolution_scale
        scene.render.filepath = str(out_dir / f"{stem}.png")

        bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    parser = ArgumentParser(description="Visualize DTU dataset")
    parser.add_argument("--model", "-m", required=True, type=str)
    parser.add_argument("--res_factor", "-r", default=1, type=int)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--label", default="ours", type=str)
    parser.add_argument("--label_mesh", default="post", type=str)
    parser.add_argument("--skip_rendering", action='store_true')
    parser.add_argument("--render_view_idx", default=-1, type=int)
    parser.add_argument("--skip_animating", action='store_true')
    parser.add_argument("--anim_ref_view", default=0, type=int)
    parser.add_argument("--anim_debug", action='store_true')
    parser.add_argument("--anim_trans", nargs=3, type=float, default=[-0.25, 0.0, 1.5])
    args = parser.parse_args()

    model_dir = Path(args.model).resolve()
    loaded_iter = search_for_max_iter(str(model_dir / "point_cloud")) if args.iteration == -1 else args.iteration
    ply_file = model_dir / "train" / f"{args.label}_{loaded_iter}" / "meshes" / f"tsdf_{args.label_mesh}.ply"

    out_dir = model_dir / "train" / f"{args.label}_{loaded_iter}" / "visuals"
    os.makedirs(out_dir, exist_ok=True)

    # Read camera poses
    camera_file = model_dir / "cameras.json"
    with open(camera_file, 'r') as f:
        views = json.load(f) # an array of camera dicts

    scene, camera, mesh, mat, bg_node, light = prepare_blender_scene(ply_file, views[0])
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    mat_nodes = mat.node_tree.nodes

    if not args.skip_animating:
        ref_view = views[args.anim_ref_view]
        mesh.location += mathutils.Vector(args.anim_trans)

        R = ref_view["rotation"]
        T = ref_view["position"]
        c2w = mathutils.Matrix(R).to_4x4()
        c2w.translation = mathutils.Vector(T)
        c2w = c2w @ mathutils.Matrix([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        camera.matrix_world = c2w

        bg_node.inputs[0].default_value = (0.010, 0.012, 0.014, 1.0)
        bsdf.inputs['Base Color'].default_value = (0.7, 0.7, 0.7, 1)
        light.energy = 50.0
        if args.anim_debug:
            scene.render.film_transparent = False
            scene.render.image_settings.file_format = 'PNG'
            scene.render.image_settings.color_mode = 'RGB'
            scene.render.image_settings.color_depth = '8'
            scene.render.resolution_x = 600
            scene.render.resolution_y = 600
            scene.render.filepath = str(out_dir / "anim_debug_k.png")
            bpy.ops.render.render(write_still=True)
        else:
            mp4_file = out_dir / f"anim_k.mp4"
            export_anim(scene, mesh, mp4_file)
            print(f"MP4 saved to: {mp4_file}")

        bg_node.inputs[0].default_value = (0.968, 0.968, 0.991, 1.0)
        vcol_node = mat_nodes.new('ShaderNodeVertexColor')
        vcol_node.layer_name = "Col"
        mat.node_tree.links.new(vcol_node.outputs['Color'], bsdf.inputs['Base Color'])
        light.energy = 75.0
        if args.anim_debug:
            scene.render.filepath = str(out_dir / "anim_debug_w.png")
            bpy.ops.render.render(write_still=True)
        else:
            mp4_file = out_dir / f"anim_w.mp4"
            export_anim(scene, mesh, mp4_file)
            print(f"MP4 saved to: {mp4_file}")

        mesh.location -= mathutils.Vector(args.anim_trans)

    if not args.skip_rendering:
        for link in list(bsdf.inputs['Base Color'].links):
            mat.node_tree.links.remove(link)
        bsdf.inputs['Base Color'].default_value = (0.7, 0.7, 0.7, 1)
        light.energy = 25.0
        render_images(scene, camera, out_dir, views, args.res_factor, args.render_view_idx)
        if args.render_view_idx >= 0:
            print(f"Rendered view {args.render_view_idx} to: {out_dir}")
        else:
            print(f"Rendered {len(views)} stills to: {out_dir}")