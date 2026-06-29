import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from main_offline import build_config
import offline_animation_export as offline
from scene import create_scene


# ── Anchor preview config ────────────────────────────────────────────────────
# None이면 bbox diagonal 기준 자동값 사용.
MARKER_RADIUS = None
MARKER_RADIUS_SCALE = 0.026

BLENDER_PREVIEW_SCRIPT = r'''
import json
from pathlib import Path

import bpy
from mathutils import Vector


CONFIG = __CONFIG__


def y_up_to_blender(v):
    x, y, z = v
    return Vector((x, -z, y))


def look_at(obj, target):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


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


def set_render_settings(scene):
    scene.render.resolution_x = CONFIG["resolution_x"]
    scene.render.resolution_y = CONFIG["resolution_y"]
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        scene.eevee.taa_render_samples = CONFIG["samples"]
        if hasattr(scene.eevee, "use_gtao"):
            scene.eevee.use_gtao = True
            scene.eevee.gtao_distance = 2.0
            scene.eevee.gtao_factor = 1.25
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
    camera.location = center + Vector((0.35 * diagonal, -CONFIG["camera_distance_scale"] * diagonal, CONFIG["camera_height_scale"] * diagonal))
    camera.data.lens = 55
    look_at(camera, center)
    bpy.context.scene.camera = camera

    point_data = bpy.data.lights.new("Point", type="POINT")
    point = bpy.data.objects.new("Point", point_data)
    bpy.context.collection.objects.link(point)
    point.location = y_up_to_blender((0.5, 1.0, 0.5))
    point.data.energy = CONFIG["point_light_energy"]
    point.data.shadow_soft_size = 0.45 * diagonal


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


def load_obj_mesh(path, mesh_material, marked_material, marked_vertices):
    vertices = []
    faces = []
    marked_vertices = set(marked_vertices)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                vertices.append(y_up_to_blender((float(parts[1]), float(parts[2]), float(parts[3]))))
            elif line.startswith("f "):
                face = []
                for part in line.split()[1:]:
                    token = part.split("/")[0]
                    if token:
                        face.append(int(token) - 1)
                if len(face) >= 3:
                    faces.append(face)

    mesh = bpy.data.meshes.new(path.stem + "Mesh")
    mesh.from_pydata([tuple(v) for v in vertices], [], faces)
    mesh.update()

    obj = bpy.data.objects.new(path.stem, mesh)
    bpy.context.collection.objects.link(obj)
    mesh.materials.append(mesh_material)
    mesh.materials.append(marked_material)

    for poly in mesh.polygons:
        poly.material_index = 1 if any(v in marked_vertices for v in poly.vertices) else 0
        poly.use_smooth = True

    return obj


def add_particle_markers(points, radius, material):
    for p in points:
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=12,
            ring_count=6,
            radius=radius,
            location=y_up_to_blender(p),
        )
        bpy.context.object.data.materials.append(material)


def main():
    clear_scene()
    scene = bpy.context.scene
    set_render_settings(scene)
    setup_world()

    mesh_mat = make_material("bunny_gray", CONFIG["bunny_color"])
    marker_mat = make_material("grabbed_particle_red", (1.0, 0.0, 0.0, 1.0))

    preview = json.loads(Path(CONFIG["preview_json"]).read_text(encoding="utf-8"))
    min_v = y_up_to_blender(preview["bbox_min"])
    max_v = y_up_to_blender(preview["bbox_max"])
    setup_camera_and_lights(min_v, max_v)
    floor_mat = make_material("floor_gray", CONFIG["floor_color"])
    add_floor(floor_mat)

    load_obj_mesh(Path(CONFIG["obj_path"]), mesh_mat, marker_mat, preview["marker_indices"])

    scene.render.filepath = CONFIG["output_png"]
    bpy.ops.render.render(write_still=True)
    print(f"[anchor-preview] wrote {CONFIG['output_png']}")


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


def write_obj(path, vertices, faces):
    with path.open("w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in faces:
            f.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def prepare_output(out_dir, output_png, overwrite):
    out_dir.mkdir(parents=True, exist_ok=True)
    if output_png.exists() and not overwrite:
        raise FileExistsError(f"{output_png} already exists.")
    if output_png.exists():
        output_png.unlink()


def main():
    config = build_config()
    out_dir = Path(config.output_path) / "anchor_preview"
    output_png = out_dir / "anchor_preview.png"
    prepare_output(out_dir, output_png, config.overwrite)

    positions, edges, rest_lengths, tet_elems, rest_volumes, surface_faces = create_scene(config.asset)
    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))
    patch_radius = config.patch_radius if config.patch_radius is not None else config.auto_patch_radius_ratio * bbox_diag

    if config.fixed_anchor_indices is None:
        rng = np.random.default_rng(config.seed)
        patches = offline.choose_pull_patches(
            positions,
            surface_faces,
            rng,
            patch_radius,
            config.random_anchor_min_distance_ratio * bbox_diag,
            config.min_patch_size,
        )
    else:
        patches = offline.choose_fixed_pull_patches(
            positions,
            surface_faces,
            config.fixed_anchor_indices,
            patch_radius,
            config.min_patch_size,
            config.pull_horizontal,
        )

    group_a = patches["group_a"].astype(np.int32)
    group_b = patches["group_b"].astype(np.int32)
    marker_indices = np.unique(np.concatenate([group_a, group_b]))
    marker_positions = positions[marker_indices]

    marker_radius = MARKER_RADIUS
    if marker_radius is None:
        marker_radius = MARKER_RADIUS_SCALE * bbox_diag

    obj_path = out_dir / "rest_bunny.obj"
    preview_json = out_dir / "anchor_preview.json"
    blender_script = out_dir / "_render_anchor_preview_blender.py"

    write_obj(obj_path, positions, surface_faces)
    preview = {
        "asset": config.asset,
        "position_space": "rest_position_after_tetgen",
        "fixed_anchor_indices": list(config.fixed_anchor_indices) if config.fixed_anchor_indices is not None else None,
        "patch_radius": float(patch_radius),
        "group_a": group_a.astype(int).tolist(),
        "group_b": group_b.astype(int).tolist(),
        "marker_indices": marker_indices.astype(int).tolist(),
        "marker_positions": marker_positions.tolist(),
        "marker_radius": float(marker_radius),
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "point_a": patches["point_a"].tolist(),
        "point_b": patches["point_b"].tolist(),
        "direction": patches["direction"].tolist(),
    }
    preview_json.write_text(json.dumps(preview, indent=2), encoding="utf-8")
    blender_script.write_text(
        BLENDER_PREVIEW_SCRIPT.replace("__CONFIG__", repr({
            "obj_path": str(obj_path.resolve()),
            "preview_json": str(preview_json.resolve()),
            "output_png": str(output_png.resolve()),
            "resolution_x": config.resolution_x,
            "resolution_y": config.resolution_y,
            "samples": config.samples,
            "marker_radius": float(marker_radius),
            "camera_distance_scale": config.camera_distance_scale,
            "camera_height_scale": config.camera_height_scale,
            "bunny_color": config.bunny_color,
            "floor_color": config.floor_color,
            "background_color": config.background_color,
            "point_light_energy": config.point_light_energy,
            "floor_y": config.floor_y,
        })),
        encoding="utf-8",
    )

    print(f"[anchor-preview] anchors: {config.fixed_anchor_indices}")
    print(f"[anchor-preview] patch sizes: A={len(group_a)} B={len(group_b)}")
    print(f"[anchor-preview] marked vertices: {len(marker_indices)}")

    blender_bin = resolve_executable(config.blender_bin)
    cmd = [blender_bin, "--background", "--python", str(blender_script)]
    print("[anchor-preview] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
