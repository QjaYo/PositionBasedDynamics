import json
import shutil
import subprocess
from pathlib import Path


BLENDER_VIEWER_SCRIPT = r"""
import json
import math
from pathlib import Path

import bpy
from mathutils import Vector

CONFIG = __CONFIG__


def y_up_to_blender(v):
    x, y, z = v
    return Vector((x, -z, y))


def obj_frame_number(path):
    return int(path.stem.split("_")[-1])


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def set_render_settings(scene):
    scene.render.resolution_x = CONFIG["resolution_x"]
    scene.render.resolution_y = CONFIG["resolution_y"]
    scene.render.fps = CONFIG["fps"]
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        scene.eevee.taa_render_samples = CONFIG["samples"]
        scene.eevee.taa_samples = CONFIG["samples"]
    except Exception:
        try:
            scene.render.engine = "BLENDER_EEVEE"
        except Exception:
            pass


def setup_world():
    bpy.context.scene.world = bpy.data.worlds.new("World") if bpy.context.scene.world is None else bpy.context.scene.world
    bpy.context.scene.world.color = CONFIG["background_color"]


def set_viewport_shading():
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.shading.type = "RENDERED"
                    space.shading.use_scene_world_render = True
                    space.shading.use_scene_lights_render = True


def collect_frame_paths(obj_dir):
    paths = sorted(obj_dir.glob("frame_*.obj"), key=obj_frame_number)
    selected = []
    for path in paths:
        frame = obj_frame_number(path)
        if frame < CONFIG["start_frame"]:
            continue
        if CONFIG["end_frame"] is not None and frame > CONFIG["end_frame"]:
            continue
        if (frame - CONFIG["start_frame"]) % CONFIG["frame_step"] != 0:
            continue
        selected.append(path)
    if not selected:
        raise RuntimeError(f"No OBJ frames found in {obj_dir}")
    return selected


def parse_obj(path):
    vertices = []
    faces = []
    face_materials = []
    material_name = "bunny_gray"

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                vertices.append(y_up_to_blender((float(parts[1]), float(parts[2]), float(parts[3]))))
            elif line.startswith("usemtl "):
                material_name = line.split(maxsplit=1)[1].strip()
            elif line.startswith("f "):
                face = []
                for part in line.split()[1:]:
                    token = part.split("/")[0]
                    if token:
                        face.append(int(token) - 1)
                if len(face) >= 3:
                    faces.append(face)
                    face_materials.append(material_name)
    return vertices, faces, face_materials


def collect_bounds(frame_paths):
    min_v = Vector((float("inf"), float("inf"), float("inf")))
    max_v = Vector((float("-inf"), float("-inf"), float("-inf")))
    for path in frame_paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.startswith("v "):
                    continue
                parts = line.split()
                v = y_up_to_blender((float(parts[1]), float(parts[2]), float(parts[3])))
                min_v.x = min(min_v.x, v.x)
                min_v.y = min(min_v.y, v.y)
                min_v.z = min(min_v.z, v.z)
                max_v.x = max(max_v.x, v.x)
                max_v.y = max(max_v.y, v.y)
                max_v.z = max(max_v.z, v.z)
    return min_v, max_v


def roller_pair_centers():
    center = CONFIG["roller_center"]
    offset = CONFIG["roller_radius"] + 0.5 * CONFIG["roller_gap"]
    if CONFIG.get("roller_pair_axis", "y") == "x":
        first = (center[0] - offset, center[1], center[2])
        second = (center[0] + offset, center[1], center[2])
    else:
        first = (center[0], center[1] + offset, center[2])
        second = (center[0], center[1] - offset, center[2])
    return first, second


def roller_camera_bounds():
    center = CONFIG["roller_center"]
    radius = CONFIG["roller_radius"]
    gap = CONFIG["roller_gap"]
    length = CONFIG["roller_length"]
    padding = CONFIG.get("camera_bounds_padding", 1.0)
    if CONFIG.get("roller_pair_axis", "y") == "x":
        half_x = (2.0 * radius + 0.5 * gap) * padding
        half_y = max(0.16, 2.2 * radius) * padding
    else:
        half_x = max(0.16, 2.2 * radius) * padding
        half_y = (2.0 * radius + 0.5 * gap) * padding
    half_z = 0.5 * length * padding
    sim_min = (center[0] - half_x, center[1] - half_y, center[2] - half_z)
    sim_max = (center[0] + half_x, center[1] + half_y, center[2] + half_z)
    return y_up_to_blender(sim_min), y_up_to_blender(sim_max)


def choose_camera_bounds(frame_paths):
    if CONFIG.get("rollers", False) and CONFIG.get("camera_target_mode") == "rollers":
        return roller_camera_bounds()
    bounds_paths = frame_paths[:1] if CONFIG.get("camera_bounds_mode") == "first_frame" else frame_paths
    return collect_bounds(bounds_paths)


def make_material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = color
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.95
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = 0.0
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.0
        elif "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = 0.0
    return mat


def add_floor(material):
    y = CONFIG["floor_y"]
    sim_vertices = [
        (-1.0, y, -1.0),
        (1.0, y, -1.0),
        (1.0, y, 1.0),
        (-1.0, y, 1.0),
    ]
    vertices = [tuple(y_up_to_blender(v)) for v in sim_vertices]
    faces = [(0, 1, 2), (0, 2, 3)]
    mesh = bpy.data.meshes.new("FloorMesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("Floor", mesh)
    bpy.context.collection.objects.link(obj)
    mesh.materials.append(material)
    return obj


def look_at(obj, target):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_camera_and_lights(min_v, max_v):
    center = (min_v + max_v) * 0.5
    diagonal = max((max_v - min_v).length, 1e-3)

    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = center + Vector((
        CONFIG.get("camera_side_scale", 0.35) * diagonal,
        -CONFIG["camera_distance_scale"] * diagonal,
        CONFIG["camera_height_scale"] * diagonal,
    ))
    camera.data.lens = CONFIG.get("camera_lens", 55)
    look_at(camera, center)
    bpy.context.scene.camera = camera

    point_data = bpy.data.lights.new("Point", type="POINT")
    point = bpy.data.objects.new("Point", point_data)
    bpy.context.collection.objects.link(point)
    point.location = y_up_to_blender((0.5, 1.0, 0.5))
    point.data.energy = CONFIG["point_light_energy"]
    point.data.shadow_soft_size = 0.45 * diagonal




def create_roller_mesh(name, center, omega):
    radius = CONFIG["roller_radius"]
    length = CONFIG["roller_length"]
    segments = CONFIG["roller_segments"]
    rings = max(4, int(CONFIG["roller_stripe_count"] * 2))
    stripe_count = CONFIG["roller_stripe_count"]
    stripe_twist = CONFIG["roller_stripe_twist"]

    vertices = []
    for ring in range(rings + 1):
        z = -0.5 * length + length * ring / rings
        for seg in range(segments):
            theta = 2.0 * math.pi * seg / segments
            sim_v = (radius * math.cos(theta), radius * math.sin(theta), z)
            vertices.append(tuple(y_up_to_blender(sim_v)))

    faces = []
    material_indices = []
    for ring in range(rings):
        for seg in range(segments):
            a = ring * segments + seg
            b = ring * segments + (seg + 1) % segments
            c = (ring + 1) * segments + (seg + 1) % segments
            d = (ring + 1) * segments + seg
            faces.append((a, b, c, d))
            stripe = int((seg / segments) * stripe_count + (ring / rings) * stripe_twist) % 2
            material_indices.append(stripe)

    back_center = len(vertices)
    vertices.append(tuple(y_up_to_blender((0.0, 0.0, -0.5 * length))))
    front_center = len(vertices)
    vertices.append(tuple(y_up_to_blender((0.0, 0.0, 0.5 * length))))
    front_ring = rings * segments
    for seg in range(segments):
        next_seg = (seg + 1) % segments
        stripe = int((seg / segments) * stripe_count) % 2
        faces.append((back_center, next_seg, seg))
        material_indices.append(stripe)
        faces.append((front_center, front_ring + seg, front_ring + next_seg))
        material_indices.append(stripe)

    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.location = y_up_to_blender(center)
    obj["roller_omega"] = float(omega)

    yellow = bpy.data.materials.get("roller_yellow") or make_material("roller_yellow", CONFIG["roller_yellow"])
    black = bpy.data.materials.get("roller_black") or make_material("roller_black", CONFIG["roller_black"])
    mesh.materials.append(yellow)
    mesh.materials.append(black)
    for poly, material_index in zip(mesh.polygons, material_indices):
        poly.material_index = material_index
    return obj


def add_rollers():
    if not CONFIG.get("rollers", False):
        return []
    first, second = roller_pair_centers()
    return [
        create_roller_mesh("UpperRoller", first, CONFIG.get("roller_upper_omega_sign", -1.0) * CONFIG["roller_angular_speed"]),
        create_roller_mesh("LowerRoller", second, CONFIG.get("roller_lower_omega_sign", 1.0) * CONFIG["roller_angular_speed"]),
    ]


def animate_rollers(rollers, end_frame):
    if not rollers:
        return
    duration = end_frame / max(1, CONFIG["fps"])
    for obj in rollers:
        obj.rotation_euler[1] = 0.0
        obj.keyframe_insert(data_path="rotation_euler", frame=0)
        obj.rotation_euler[1] = float(obj["roller_omega"]) * duration
        obj.keyframe_insert(data_path="rotation_euler", frame=end_frame)


def load_frame_object(path, frame_index, gray, red, collection):
    vertices, faces, face_materials = parse_obj(path)
    mesh = bpy.data.meshes.new(f"Frame_{frame_index:04d}Mesh")
    mesh.from_pydata([tuple(v) for v in vertices], [], faces)
    mesh.update()
    mesh.materials.append(gray)
    mesh.materials.append(red)

    for poly, material_name in zip(mesh.polygons, face_materials):
        poly.material_index = 1 if material_name == "pull_marker_red" else 0
        poly.use_smooth = True

    obj = bpy.data.objects.new(f"Frame_{frame_index:04d}", mesh)
    collection.objects.link(obj)
    return obj


def key_visibility(obj, frame, visible):
    bpy.context.scene.frame_set(frame)
    obj.hide_viewport = not visible
    obj.hide_render = not visible
    obj.keyframe_insert(data_path="hide_viewport", frame=frame)
    obj.keyframe_insert(data_path="hide_render", frame=frame)


def set_constant_keys(obj):
    action = obj.animation_data.action if obj.animation_data else None
    fcurves = getattr(action, "fcurves", None)
    if not fcurves:
        return
    for fcurve in fcurves:
        for key in fcurve.keyframe_points:
            key.interpolation = "CONSTANT"


def animate_single_frame_visibility(obj, frame, start_frame, end_frame):
    key_visibility(obj, start_frame, frame == start_frame)
    if frame != start_frame:
        key_visibility(obj, frame, True)
    if frame < end_frame:
        key_visibility(obj, frame + 1, False)
    set_constant_keys(obj)


def main():
    obj_dir = Path(CONFIG["obj_dir"])
    output_blend = Path(CONFIG["output_blend"])
    output_blend.parent.mkdir(parents=True, exist_ok=True)

    clear_scene()
    scene = bpy.context.scene
    scene.frame_start = 0
    set_render_settings(scene)
    setup_world()

    frame_paths = collect_frame_paths(obj_dir)
    scene.frame_end = len(frame_paths) - 1

    min_v, max_v = choose_camera_bounds(frame_paths)

    gray = make_material("bunny_gray", CONFIG["bunny_color"])
    red = make_material("pull_marker_red", (1.0, 0.0, 0.0, 1.0))
    floor_mat = make_material("floor_gray", CONFIG["floor_color"])
    add_floor(floor_mat)
    setup_camera_and_lights(min_v, max_v)
    rollers = add_rollers()
    animate_rollers(rollers, len(frame_paths) - 1)

    frames_collection = bpy.data.collections.new("FrameMeshes")
    bpy.context.scene.collection.children.link(frames_collection)

    for frame_index, path in enumerate(frame_paths):
        obj = load_frame_object(path, frame_index, gray, red, frames_collection)
        animate_single_frame_visibility(obj, frame_index, 0, len(frame_paths) - 1)
        if frame_index % max(1, CONFIG["fps"]) == 0 or frame_index == len(frame_paths) - 1:
            print(f"[blend-viewer] loaded {frame_index + 1}/{len(frame_paths)}", flush=True)

    scene.frame_set(0)
    set_viewport_shading()
    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend))
    print(f"[blend-viewer] wrote {output_blend}")


main()
"""


def resolve_executable(path):
    resolved = shutil.which(path) if path else None
    if resolved:
        return resolved
    if path and Path(path).exists():
        return path
    raise FileNotFoundError(f"Executable not found: {path}")


def choose_obj_dir(config):
    marked_dir = Path(config.marked_obj_dir)
    if marked_dir.exists() and any(marked_dir.glob("frame_*.obj")):
        return marked_dir
    return Path(config.obj_dir)


def read_metadata(obj_dir):
    metadata_path = obj_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def read_fps(config, obj_dir):
    if config.video_fps is not None:
        return int(config.video_fps)
    return int(read_metadata(obj_dir).get("fps", config.fps))


def read_roller_settings(config, obj_dir):
    rollers = read_metadata(obj_dir).get("rollers", {})
    return {
        "center": tuple(rollers.get("center", config.roller_center)),
        "pair_axis": rollers.get("pair_axis", config.roller_pair_axis),
    }


def write_blender_script(output_root, blender_config):
    script = BLENDER_VIEWER_SCRIPT.replace("__CONFIG__", repr(blender_config))
    script_path = output_root / "_create_animation_blend_viewer.py"
    script_path.write_text(script, encoding="utf-8")
    return script_path


def export_animation_blend_viewer(config):
    obj_dir = choose_obj_dir(config)
    output_blend = Path(config.blend_path)
    output_root = Path(config.output_path)
    blender_bin = resolve_executable(config.blender_bin)
    fps = read_fps(config, obj_dir)
    roller_settings = read_roller_settings(config, obj_dir)

    blender_config = {
        "obj_dir": str(obj_dir.resolve()),
        "output_blend": str(output_blend.resolve()),
        "resolution_x": config.resolution_x,
        "resolution_y": config.resolution_y,
        "fps": fps,
        "samples": config.samples,
        "start_frame": config.start_frame,
        "end_frame": config.end_frame,
        "frame_step": config.frame_step,
        "camera_distance_scale": config.camera_distance_scale,
        "camera_height_scale": config.camera_height_scale,
        "camera_side_scale": config.camera_side_scale,
        "camera_lens": config.camera_lens,
        "camera_bounds_mode": config.camera_bounds_mode,
        "camera_target_mode": config.camera_target_mode,
        "camera_bounds_padding": config.camera_bounds_padding,
        "bunny_color": config.bunny_color,
        "floor_color": config.floor_color,
        "background_color": config.background_color,
        "point_light_energy": config.point_light_energy,
        "floor_y": config.floor_y,
        "rollers": config.render_rollers,
        "roller_center": roller_settings["center"],
        "roller_pair_axis": roller_settings["pair_axis"],
        "roller_radius": config.roller_radius,
        "roller_gap": config.roller_gap,
        "roller_length": config.roller_length,
        "roller_angular_speed": config.roller_angular_speed,
        "roller_upper_omega_sign": config.roller_upper_omega_sign,
        "roller_lower_omega_sign": config.roller_lower_omega_sign,
        "roller_segments": config.roller_segments,
        "roller_stripe_count": config.roller_stripe_count,
        "roller_stripe_twist": config.roller_stripe_twist,
        "roller_yellow": config.roller_yellow,
        "roller_black": config.roller_black,
    }
    script_path = write_blender_script(output_root, blender_config)
    cmd = [blender_bin, "--background", "--python", str(script_path)]
    print("[blend-viewer] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return output_blend


def main():
    from main_offline import build_config

    export_animation_blend_viewer(build_config())


if __name__ == "__main__":
    main()
