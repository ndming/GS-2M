import os
import json
import bpy
import math
import mathutils

from argparse import ArgumentParser
from pathlib import Path
from PIL import Image

trans_configs = {
    24: [-2.0, 0.05, 1.2],
    37: [-1.6, 0.2, 0.6],
    40: [-1.2, 0.1, 0.7],
    55: [-1.2, 0.1, 0.8],
    63: [-0.8, 0.2, 0.6],
    65: [-1.0, 0.0, 0.6],
    69: [-1.2, 0.0, 0.8],
    83: [-1.0, 0.1, 0.6],
    97: [-0.7, 0.1, 0.3],
    105: [-0.3, -0.02, 0.1],
    106: [-1.0, -0.0, 0.8],
    110: [-1.2, 0.1, 0.8],
    114: [-1.2, 0.0, 0.8],
    118: [-1.2, 0.0, 0.8],
    122: [-1.2, 0.0, 0.8],
}

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
    setup_background(scene)

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
    area_light = bpy.data.lights.new(name="Area", type='AREA')
    area_object = bpy.data.objects.new(name="Area", object_data=area_light)
    bpy.context.scene.collection.objects.link(area_object)
    area_light.shape = 'DISK'
    area_light.size = 5.0
    area_object.rotation_mode = 'QUATERNION'
    area_object.rotation_quaternion = rotation_quat
    area_object.location = camera.location + camera_z_axis * 2.0

    return scene, camera, mesh, mat, area_light

def setup_background(scene, color=(0.0, 0.0, 0.0, 1.0)):
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
    bg_light.inputs[1].default_value = 0.0  # Strength = 0, no lighting

    bg_node = nodes.new(type='ShaderNodeBackground')
    bg_node.inputs[0].default_value = color
    bg_node.inputs[1].default_value = 1.0

    mix_shader = nodes.new(type='ShaderNodeMixShader')
    light_path = nodes.new(type='ShaderNodeLightPath')
    world_output = nodes.new(type='ShaderNodeOutputWorld')
    links.new(light_path.outputs['Is Camera Ray'], mix_shader.inputs['Fac'])
    links.new(bg_light.outputs['Background'], mix_shader.inputs[1])  # Lighting
    links.new(bg_node.outputs['Background'], mix_shader.inputs[2])  # Visible BG
    links.new(mix_shader.outputs['Shader'], world_output.inputs['Surface'])

def export_anim(scene, mesh, frame_dir):
    scene.render.resolution_x = 600
    scene.render.resolution_y = 600
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'

    num_frames = 120

    # Create rotation animation for object
    mesh.rotation_mode = 'XYZ'
    start_rot = mesh.rotation_euler.copy()
    for frame in range(num_frames):
        # Oscillate rotation using sine waves
        angle_x = 0.4 * math.sin(2 * math.pi * frame / num_frames * 1)
        angle_y = 0.6 * math.sin(2 * math.pi * frame / num_frames * 2)

        mesh.rotation_euler = start_rot.copy()
        mesh.rotation_euler.x += angle_x
        mesh.rotation_euler.y += angle_y

        scene.render.filepath = str(frame_dir / f"{frame:03d}.png")
        bpy.ops.render.render(write_still=True)


def convert_gif(frame_dir, gif_file):
    frames = [Image.open(frame).convert("RGBA") for frame in sorted(frame_dir.glob("*.png")) if frame.is_file()]
    cleaned_frames = []
    for frame in frames:
        new_data = []
        for r, g, b, a in frame.getdata():
            if a == 0:
                new_data.append((0, 0, 0, 0))
            else:
                new_data.append((r, g, b, 255))
        frame.putdata(new_data)
        cleaned_frames.append(frame)

    palette_source = cleaned_frames[0].convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=255)
    palette = palette_source.getpalette()

    try:
        trans_index = next(i for i in range(256) if palette[i*3:i*3+3] == [0, 0, 0])
    except StopIteration:
        trans_index = 0

    paletted_images = []
    for frame in cleaned_frames:
        alpha = frame.getchannel("A")
        pal_img = frame.convert("RGB").quantize(palette=palette_source)
        transparent_mask = alpha.point(lambda a: 255 if a == 0 else 0)
        pal_img.paste(trans_index, mask=transparent_mask)
        paletted_images.append(pal_img)
    
    paletted_images[0].save(
        gif_file, save_all=True, append_images=paletted_images[1:], optimize=True,
        duration=int(1000 / 24), loop=0, transparency=trans_index, disposal=2)

def convert_webp(frame_dir, webp_file):
    frames = [Image.open(frame).convert("RGBA") for frame in sorted(frame_dir.glob("*.png")) if frame.is_file()]
    frames[0].save(
        webp_file, save_all=True, append_images=frames[1:], format='WEBP',
        duration=int(1000 / 24), loop=0, transparency=0, disposal=2)

def render_images(scene, camera, out_dir, views, resolution_scale, view_idx):
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'

    render_dir = out_dir / "views"
    os.makedirs(render_dir, exist_ok=True)

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
        scene.render.filepath = str(render_dir / f"{stem}.png")

        bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    parser = ArgumentParser(description="Visualize DTU dataset")
    parser.add_argument("--model", "-m", required=True, type=str)
    parser.add_argument("--res_factor", "-r", default=1, type=int)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--label", default="ours", type=str)
    parser.add_argument("--label_mesh", default="post", type=str)
    parser.add_argument("--rendering", action='store_true')
    parser.add_argument("--render_view_idx", default=-1, type=int)
    parser.add_argument("--animation", action='store_true')
    parser.add_argument("--debug_anim", action='store_true')
    parser.add_argument("--still", action='store_true')
    args = parser.parse_args()

    model_dir = Path(args.model).resolve()
    loaded_iter = search_for_max_iter(str(model_dir / "point_cloud")) if args.iteration == -1 else args.iteration
    ply_file = model_dir / "train" / f"{args.label}_{loaded_iter}" / "mesh" / f"tsdf_{args.label_mesh}.ply"
    scanID = int(model_dir.name[4:])

    out_dir = model_dir / "train" / f"{args.label}_{loaded_iter}" / "visual"
    os.makedirs(out_dir, exist_ok=True)

    # Read camera poses
    camera_file = model_dir / "cameras.json"
    with open(camera_file, 'r') as f:
        views = json.load(f) # an array of camera dicts

    scene, camera, mesh, mat, light = prepare_blender_scene(ply_file, views[0])
    bsdf = mat.node_tree.nodes.get('Principled BSDF')

    if args.still:
        scene.render.film_transparent = True
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGBA'
        scene.render.image_settings.color_depth = '8'
        scene.render.resolution_x = 600
        scene.render.resolution_y = 600
        scene.render.filepath = str(out_dir / "still.png")

        bsdf.inputs['Base Color'].default_value = (0.7, 0.7, 0.7, 1)
        light.energy = 75.0

        ref_view = views[23]
        mesh.location += mathutils.Vector(trans_configs[scanID])

        R = ref_view["rotation"]
        T = ref_view["position"]
        c2w = mathutils.Matrix(R).to_4x4()
        c2w.translation = mathutils.Vector(T)
        c2w = c2w @ mathutils.Matrix([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        camera.matrix_world = c2w

        bpy.ops.render.render(write_still=True)
        mesh.location -= mathutils.Vector(trans_configs[scanID])

    if args.animation:
        ref_view = views[23]
        mesh.location += mathutils.Vector(trans_configs[scanID])

        R = ref_view["rotation"]
        T = ref_view["position"]
        c2w = mathutils.Matrix(R).to_4x4()
        c2w.translation = mathutils.Vector(T)
        c2w = c2w @ mathutils.Matrix([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        camera.matrix_world = c2w

        bsdf.inputs['Base Color'].default_value = (0.7, 0.7, 0.7, 1)
        light.energy = 50.0
        if args.debug_anim:
            scene.render.film_transparent = True
            scene.render.image_settings.file_format = 'PNG'
            scene.render.image_settings.color_mode = 'RGBA'
            scene.render.image_settings.color_depth = '8'
            scene.render.resolution_x = 600
            scene.render.resolution_y = 600
            scene.render.filepath = str(out_dir / "anim_debug.png")
            bpy.ops.render.render(write_still=True)
        else:
            frame_dir = out_dir / "frames"
            os.makedirs(frame_dir, exist_ok=True)
            export_anim(scene, mesh, frame_dir)

            gif_file = out_dir / f"anim.gif"
            convert_gif(frame_dir, gif_file)
            print(f"GIF saved to: {gif_file}")

            webp_file = out_dir / f"anim.webp"
            convert_webp(frame_dir, webp_file)
            print(f"WEBP saved to: {webp_file}")

        mesh.location -= mathutils.Vector(trans_configs[scanID])

    if args.rendering:
        for link in list(bsdf.inputs['Base Color'].links):
            mat.node_tree.links.remove(link)
        bsdf.inputs['Base Color'].default_value = (0.7, 0.7, 0.7, 1)
        light.energy = 25.0
        render_images(scene, camera, out_dir, views, args.res_factor, args.render_view_idx)
        if args.render_view_idx >= 0:
            print(f"Rendered view {args.render_view_idx} to: {out_dir}")
        else:
            print(f"Rendered {len(views)} stills to: {out_dir}")