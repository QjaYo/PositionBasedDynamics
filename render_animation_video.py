import json
import shutil
import subprocess
from pathlib import Path

BLENDER_RENDER_SCRIPT = r'''
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
    stem = path.stem
    return int(stem.split("_")[-1])


def collect_bounds(paths):
    min_v = Vector((float("inf"), float("inf"), float("inf")))
    max_v = Vector((float("-inf"), float("-inf"), float("-inf")))
    for path in paths:
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
    camera_bounds_path = CONFIG.get("camera_bounds_path")
    if camera_bounds_path:
        return collect_bounds([Path(camera_bounds_path)])
    bounds_paths = frame_paths[:1] if CONFIG.get("camera_bounds_mode") == "first_frame" else frame_paths
    return collect_bounds(bounds_paths)


def look_at(obj, target):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


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
    except Exception:
        try:
            scene.render.engine = "BLENDER_EEVEE"
        except Exception:
            pass


def setup_world():
    bpy.context.scene.world = bpy.data.worlds.new("World") if bpy.context.scene.world is None else bpy.context.scene.world
    bpy.context.scene.world.color = CONFIG["background_color"]


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
        ( 1.0, y, -1.0),
        ( 1.0, y,  1.0),
        (-1.0, y,  1.0),
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


def set_roller_frame(rollers, frame):
    if not rollers:
        return
    t = frame / max(1, CONFIG["fps"])
    for obj in rollers:
        obj.rotation_euler[1] = float(obj["roller_omega"]) * t


def load_obj_mesh(path):
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

    mesh = bpy.data.meshes.new(path.stem + "Mesh")
    mesh.from_pydata([tuple(v) for v in vertices], [], faces)
    mesh.update()

    obj = bpy.data.objects.new(path.stem, mesh)
    bpy.context.collection.objects.link(obj)

    gray = bpy.data.materials.get("bunny_gray") or make_material("bunny_gray", CONFIG["bunny_color"])
    red = bpy.data.materials.get("pull_marker_red") or make_material("pull_marker_red", (1.0, 0.0, 0.0, 1.0))
    mesh.materials.append(gray)
    mesh.materials.append(red)

    for poly, name in zip(mesh.polygons, face_materials):
        poly.material_index = 1 if name == "pull_marker_red" else 0
        poly.use_smooth = True

    return obj


def delete_objects(objects):
    for obj in objects:
        bpy.data.objects.remove(obj, do_unlink=True)


def main():
    obj_dir = Path(CONFIG["obj_dir"])
    png_dir = Path(CONFIG["png_dir"])
    png_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(obj_dir.glob("frame_*.obj"), key=obj_frame_number)
    frame_paths = [
        p for p in frame_paths
        if obj_frame_number(p) >= CONFIG["start_frame"]
        and (CONFIG["end_frame"] is None or obj_frame_number(p) <= CONFIG["end_frame"])
        and ((obj_frame_number(p) - CONFIG["start_frame"]) % CONFIG["frame_step"] == 0)
    ]
    if not frame_paths:
        raise RuntimeError(f"No OBJ frames found in {obj_dir}")

    clear_scene()
    set_render_settings(bpy.context.scene)
    setup_world()
    min_v, max_v = choose_camera_bounds(frame_paths)
    setup_camera_and_lights(min_v, max_v)
    floor_mat = make_material("floor_gray", CONFIG["floor_color"])
    add_floor(floor_mat)
    rollers = add_rollers()

    for out_idx, path in enumerate(frame_paths):
        source_frame = obj_frame_number(path)
        render_frame = source_frame if CONFIG.get("use_source_frame_time", False) else out_idx
        output_frame = source_frame if CONFIG.get("preserve_frame_numbers", False) else out_idx
        bpy.context.scene.frame_set(render_frame)
        set_roller_frame(rollers, render_frame)
        obj = load_obj_mesh(path)
        bpy.context.scene.render.filepath = str(png_dir / f"frame_{output_frame:04d}.png")
        bpy.ops.render.render(write_still=True)
        delete_objects([obj])
        if out_idx % max(1, CONFIG["fps"]) == 0 or out_idx == len(frame_paths) - 1:
            print(f"[blender] rendered {out_idx + 1}/{len(frame_paths)}")


if __name__ == "__main__":
    main()
'''


def resolve_executable(path_or_name):
    path = Path(path_or_name)
    if path.exists():
        return str(path)
    resolved = shutil.which(path_or_name)
    if resolved:
        return resolved
    raise FileNotFoundError(f"Executable not found: {path_or_name}")


def prepare_png_dir(png_dir, overwrite):
    png_dir.mkdir(parents=True, exist_ok=True)
    existing = list(png_dir.glob("frame_*.png"))
    if existing and not overwrite:
        raise FileExistsError(f"{png_dir} already has frame_*.png files.")
    for path in existing:
        path.unlink()


def read_metadata(obj_dir):
    metadata_path = obj_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def read_fps(obj_dir, fps_override):
    if fps_override is not None:
        return fps_override
    return int(read_metadata(obj_dir).get("fps", 30))


def read_roller_settings(config, obj_dir):
    rollers = read_metadata(obj_dir).get("rollers", {})
    return {
        "center": tuple(rollers.get("center", config.roller_center)),
        "pair_axis": rollers.get("pair_axis", config.roller_pair_axis),
    }


def write_blender_script(png_dir, config):
    script = BLENDER_RENDER_SCRIPT.replace("__CONFIG__", repr(config))
    script_path = png_dir / "_render_sequence_blender.py"
    script_path.write_text(script, encoding="utf-8")
    return script_path


def run_blender(blender_bin, script_path):
    cmd = [blender_bin, "--background", "--python", str(script_path)]
    print("[video] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_ffmpeg(ffmpeg_bin, png_dir, output_mp4, fps, overwrite):
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    if output_mp4.exists() and not overwrite:
        raise FileExistsError(f"{output_mp4} already exists.")
    cmd = [
        ffmpeg_bin,
        "-y" if overwrite else "-n",
        "-framerate", str(fps),
        "-i", str(png_dir / "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_mp4),
    ]
    print("[video] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def render_preview_frame(config, frame):
    obj_dir = Path(config.obj_dir)
    png_dir = Path(config.frame_preview_dir)
    output_png = png_dir / f"frame_{int(frame):04d}.png"
    frame_path = obj_dir / f"frame_{int(frame):04d}.obj"
    if not frame_path.exists():
        raise FileNotFoundError(f"Preview frame OBJ not found: {frame_path}")
    if output_png.exists() and not config.overwrite:
        raise FileExistsError(f"{output_png} already exists.")

    png_dir.mkdir(parents=True, exist_ok=True)
    if output_png.exists():
        output_png.unlink()

    fps = read_fps(obj_dir, config.video_fps)
    blender_bin = resolve_executable(config.blender_bin)
    camera_bounds_path = None
    first_frame_path = obj_dir / "frame_0000.obj"
    if config.camera_bounds_mode == "first_frame" and first_frame_path.exists():
        camera_bounds_path = str(first_frame_path.resolve())
    roller_settings = read_roller_settings(config, obj_dir)
    blender_config = {
        "obj_dir": str(obj_dir.resolve()),
        "png_dir": str(png_dir.resolve()),
        "resolution_x": config.resolution_x,
        "resolution_y": config.resolution_y,
        "fps": fps,
        "samples": config.samples,
        "start_frame": int(frame),
        "end_frame": int(frame),
        "frame_step": 1,
        "preserve_frame_numbers": True,
        "use_source_frame_time": True,
        "camera_distance_scale": config.camera_distance_scale,
        "camera_height_scale": config.camera_height_scale,
        "camera_side_scale": config.camera_side_scale,
        "camera_lens": config.camera_lens,
        "camera_bounds_mode": config.camera_bounds_mode,
        "camera_target_mode": config.camera_target_mode,
        "camera_bounds_padding": config.camera_bounds_padding,
        "camera_bounds_path": camera_bounds_path,
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
    script_path = write_blender_script(png_dir, blender_config)
    run_blender(blender_bin, script_path)
    print(f"[preview] wrote {output_png}")
    return output_png


def render_animation_video(config):
    obj_dir = Path(config.marked_obj_dir)
    png_dir = Path(config.png_dir)
    output_mp4 = Path(config.mp4_path)
    fps = read_fps(obj_dir, config.video_fps)

    blender_bin = resolve_executable(config.blender_bin)
    ffmpeg_bin = resolve_executable(config.ffmpeg_bin)
    prepare_png_dir(png_dir, config.overwrite)
    roller_settings = read_roller_settings(config, obj_dir)

    blender_config = {
        "obj_dir": str(obj_dir.resolve()),
        "png_dir": str(png_dir.resolve()),
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
    script_path = write_blender_script(png_dir, blender_config)
    run_blender(blender_bin, script_path)
    run_ffmpeg(ffmpeg_bin, png_dir, output_mp4, fps, config.overwrite)
    print(f"[video] wrote {output_mp4}")


def main():
    from main_offline import build_config

    render_animation_video(build_config())


if __name__ == "__main__":
    main()
