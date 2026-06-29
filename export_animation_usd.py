import json
from pathlib import Path

import numpy as np

GRAY = (0.5, 0.5, 0.5)
RED = (1.0, 0.0, 0.0)
FLOOR_GRAY = (0.3, 0.3, 0.3)


def obj_frame_number(path):
    return int(path.stem.split("_")[-1])


def y_up_to_blender(v):
    x, y, z = v
    return (x, -z, y)


def read_metadata(obj_dir):
    metadata_path = obj_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def read_fps(obj_dir, fallback):
    if fallback is not None:
        return int(fallback)
    return int(read_metadata(obj_dir).get("fps", 30))


def read_roller_settings(config, obj_dir):
    rollers = read_metadata(obj_dir).get("rollers", {})
    return {
        "center": tuple(rollers.get("center", config.roller_center)),
        "pair_axis": rollers.get("pair_axis", config.roller_pair_axis),
    }


def choose_obj_dir(config):
    marked_dir = Path(config.marked_obj_dir)
    if marked_dir.exists() and any(marked_dir.glob("frame_*.obj")):
        return marked_dir
    return Path(config.obj_dir)


def collect_frame_paths(obj_dir, start_frame, end_frame, frame_step):
    paths = sorted(obj_dir.glob("frame_*.obj"), key=obj_frame_number)
    selected = []
    for path in paths:
        frame = obj_frame_number(path)
        if frame < start_frame:
            continue
        if end_frame is not None and frame > end_frame:
            continue
        if (frame - start_frame) % frame_step != 0:
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
                sim_v = (float(parts[1]), float(parts[2]), float(parts[3]))
                vertices.append(y_up_to_blender(sim_v))
            elif line.startswith("usemtl "):
                material_name = line.split(maxsplit=1)[1].strip()
            elif line.startswith("f "):
                face = []
                for part in line.split()[1:]:
                    token = part.split("/")[0]
                    if not token:
                        continue
                    idx = int(token)
                    if idx < 0:
                        idx = len(vertices) + idx
                    else:
                        idx -= 1
                    face.append(idx)
                if len(face) >= 3:
                    faces.append(face)
                    face_materials.append(material_name)
    return vertices, faces, face_materials


def fmt_float(value):
    return f"{value:.8g}"


def fmt_vec3(v):
    return f"({fmt_float(v[0])}, {fmt_float(v[1])}, {fmt_float(v[2])})"


def write_list(f, indent, values, formatter, items_per_line):
    prefix = " " * indent
    for i in range(0, len(values), items_per_line):
        chunk = values[i:i + items_per_line]
        suffix = "," if i + items_per_line < len(values) else ""
        f.write(prefix + ", ".join(formatter(v) for v in chunk) + suffix + "\n")


def write_int_array(f, indent, name, values, items_per_line=18):
    f.write(" " * indent + f"int[] {name} = [\n")
    write_list(f, indent + 4, values, str, items_per_line)
    f.write(" " * indent + "]\n")


def write_point_array(f, indent, name, values, type_name="point3f", items_per_line=3):
    f.write(" " * indent + f"{type_name}[] {name} = [\n")
    write_list(f, indent + 4, values, fmt_vec3, items_per_line)
    f.write(" " * indent + "]\n")


def write_color_primvar(f, colors):
    f.write("        color3f[] primvars:displayColor = [\n")
    write_list(f, 12, colors, fmt_vec3, 4)
    f.write("        ] (\n")
    f.write('            interpolation = "uniform"\n')
    f.write("        )\n")


def write_visibility_samples(f, frame, start_frame, end_frame):
    samples = {start_frame: "invisible", frame: "inherited"}
    if frame < end_frame:
        samples[frame + 1] = "invisible"
    f.write("        token visibility.timeSamples = {\n")
    for time in sorted(samples):
        f.write(f'            {time}: "{samples[time]}",\n')
    f.write("        }\n")


def write_mesh_prim(
    f,
    name,
    vertices,
    faces,
    face_materials,
    frame=None,
    start_frame=0,
    end_frame=0,
    material_colors=None,
):
    counts = [len(face) for face in faces]
    indices = [idx for face in faces for idx in face]
    default_colors = {
        "bunny_gray": GRAY,
        "pull_marker_red": RED,
        "floor_gray": FLOOR_GRAY,
    }
    if material_colors:
        default_colors.update(material_colors)
    colors = [default_colors.get(material, GRAY) for material in face_materials]

    f.write(f'    def Mesh "{name}"\n')
    f.write("    {\n")
    f.write('        uniform token subdivisionScheme = "none"\n')
    write_point_array(f, 8, "points", vertices)
    write_int_array(f, 8, "faceVertexCounts", counts)
    write_int_array(f, 8, "faceVertexIndices", indices)
    write_color_primvar(f, colors)
    if frame is not None:
        write_visibility_samples(f, frame, start_frame, end_frame)
    f.write("    }\n")


def floor_mesh(floor_y):
    sim_vertices = [
        (-1.0, floor_y, -1.0),
        (1.0, floor_y, -1.0),
        (1.0, floor_y, 1.0),
        (-1.0, floor_y, 1.0),
    ]
    vertices = [y_up_to_blender(v) for v in sim_vertices]
    faces = [(0, 1, 2), (0, 2, 3)]
    return vertices, faces, ["floor_gray", "floor_gray"]


def roller_mesh(center, radius, length, segments, stripe_count, stripe_twist):
    rings = max(4, int(stripe_count * 2))
    vertices = []
    for ring in range(rings + 1):
        z = -0.5 * length + length * ring / rings
        for seg in range(segments):
            theta = 2.0 * np.pi * seg / segments
            sim_v = (
                center[0] + radius * np.cos(theta),
                center[1] + radius * np.sin(theta),
                center[2] + z,
            )
            vertices.append(y_up_to_blender(sim_v))

    faces = []
    materials = []
    for ring in range(rings):
        for seg in range(segments):
            a = ring * segments + seg
            b = ring * segments + (seg + 1) % segments
            c = (ring + 1) * segments + (seg + 1) % segments
            d = (ring + 1) * segments + seg
            faces.append((a, b, c, d))
            stripe = int((seg / segments) * stripe_count + (ring / rings) * stripe_twist) % 2
            materials.append("roller_black" if stripe else "roller_yellow")

    back_center = len(vertices)
    vertices.append(y_up_to_blender((center[0], center[1], center[2] - 0.5 * length)))
    front_center = len(vertices)
    vertices.append(y_up_to_blender((center[0], center[1], center[2] + 0.5 * length)))
    front_ring = rings * segments
    for seg in range(segments):
        next_seg = (seg + 1) % segments
        stripe = int((seg / segments) * stripe_count) % 2
        material = "roller_black" if stripe else "roller_yellow"
        faces.append((back_center, next_seg, seg))
        materials.append(material)
        faces.append((front_center, front_ring + seg, front_ring + next_seg))
        materials.append(material)
    return vertices, faces, materials


def roller_specs(config):
    if config is None:
        return []
    center = config["center"]
    radius = config["radius"]
    offset = radius + 0.5 * config["gap"]
    if config.get("pair_axis", "y") == "x":
        upper = (center[0] - offset, center[1], center[2])
        lower = (center[0] + offset, center[1], center[2])
    else:
        upper = (center[0], center[1] + offset, center[2])
        lower = (center[0], center[1] - offset, center[2])
    return [("UpperRoller", upper), ("LowerRoller", lower)]


def write_usd_sequence(
    obj_dir,
    output_usd,
    fps=30,
    start_frame=0,
    end_frame=None,
    frame_step=1,
    floor_y=None,
    rollers_config=None,
):
    obj_dir = Path(obj_dir)
    output_usd = Path(output_usd)
    frame_paths = collect_frame_paths(obj_dir, start_frame, end_frame, frame_step)
    timeline_start = 0
    timeline_end = len(frame_paths) - 1

    output_usd.parent.mkdir(parents=True, exist_ok=True)
    with output_usd.open("w", encoding="utf-8") as f:
        f.write("#usda 1.0\n")
        f.write("(\n")
        f.write('    defaultPrim = "TearingSequence"\n')
        f.write(f"    startTimeCode = {timeline_start}\n")
        f.write(f"    endTimeCode = {timeline_end}\n")
        f.write(f"    framesPerSecond = {int(fps)}\n")
        f.write(f"    timeCodesPerSecond = {int(fps)}\n")
        f.write('    upAxis = "Z"\n')
        f.write("    metersPerUnit = 1\n")
        f.write(")\n\n")
        f.write('def Xform "TearingSequence"\n')
        f.write("{\n")

        if floor_y is not None:
            vertices, faces, materials = floor_mesh(floor_y)
            write_mesh_prim(f, "Floor", vertices, faces, materials)
            f.write("\n")

        if rollers_config is not None:
            roller_colors = {
                "roller_yellow": tuple(rollers_config["yellow"][:3]),
                "roller_black": tuple(rollers_config["black"][:3]),
            }
            for name, center in roller_specs(rollers_config):
                vertices, faces, materials = roller_mesh(
                    center,
                    rollers_config["radius"],
                    rollers_config["length"],
                    rollers_config["segments"],
                    rollers_config["stripe_count"],
                    rollers_config["stripe_twist"],
                )
                write_mesh_prim(f, name, vertices, faces, materials, material_colors=roller_colors)
                f.write("\n")

        for out_frame, path in enumerate(frame_paths):
            vertices, faces, face_materials = parse_obj(path)
            write_mesh_prim(
                f,
                f"Frame_{out_frame:04d}",
                vertices,
                faces,
                face_materials,
                frame=out_frame,
                start_frame=timeline_start,
                end_frame=timeline_end,
            )
            if out_frame != timeline_end:
                f.write("\n")
            if out_frame % max(1, fps) == 0 or out_frame == timeline_end:
                print(f"[usd] wrote mesh {out_frame + 1}/{len(frame_paths)} from {path.name}", flush=True)

        f.write("}\n")

    print(f"[usd] wrote {output_usd}")
    return output_usd


def export_animation_usd(config):
    obj_dir = choose_obj_dir(config)
    output_usd = Path(config.usd_path)
    fps = read_fps(obj_dir, config.video_fps)
    roller_settings = read_roller_settings(config, obj_dir)
    return write_usd_sequence(
        obj_dir=obj_dir,
        output_usd=output_usd,
        fps=fps,
        start_frame=config.start_frame,
        end_frame=config.end_frame,
        frame_step=config.frame_step,
        floor_y=config.floor_y,
        rollers_config=(
            {
                "center": roller_settings["center"],
                "pair_axis": roller_settings["pair_axis"],
                "radius": config.roller_radius,
                "gap": config.roller_gap,
                "length": config.roller_length,
                "segments": config.roller_segments,
                "stripe_count": config.roller_stripe_count,
                "stripe_twist": config.roller_stripe_twist,
                "yellow": config.roller_yellow,
                "black": config.roller_black,
            }
            if config.render_rollers
            else None
        ),
    )


def main():
    from main_offline import build_config

    export_animation_usd(build_config())


if __name__ == "__main__":
    main()
