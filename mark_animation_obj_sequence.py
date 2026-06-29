import json
import shutil
from pathlib import Path

def write_material(path):
    path.write_text(
        "newmtl bunny_gray\n"
        "Kd 0.5 0.5 0.5\n"
        "Ka 0.1 0.1 0.1\n"
        "newmtl pull_marker_red\n"
        "Kd 1.0 0.0 0.0\n"
        "Ka 0.25 0.0 0.0\n",
        encoding="utf-8",
    )


def face_indices(line):
    indices = []
    for part in line.split()[1:]:
        vertex_token = part.split("/")[0]
        if vertex_token:
            indices.append(int(vertex_token) - 1)
    return indices


def mark_frame(src_path, dst_path, material_name, marked_vertices):
    with src_path.open("r", encoding="utf-8") as src, dst_path.open("w", encoding="utf-8") as dst:
        dst.write(f"mtllib {material_name}\n")
        current_material = None

        for line in src:
            if line.startswith("f "):
                material = "pull_marker_red" if any(i in marked_vertices for i in face_indices(line)) else "bunny_gray"
                if material != current_material:
                    dst.write(f"usemtl {material}\n")
                    current_material = material
            elif line.startswith("usemtl ") or line.startswith("mtllib "):
                continue
            dst.write(line)


def mark_animation_obj_sequence(config):
    in_dir = Path(config.obj_dir)
    out_dir = Path(config.marked_obj_dir)
    overwrite = config.overwrite
    metadata_path = in_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("frame_*.obj"))
    if existing and not overwrite:
        raise FileExistsError(f"{out_dir} already has frame_*.obj files. Set overwrite=True to replace them.")
    for path in existing:
        path.unlink()

    material_name = "pull_markers.mtl"
    write_material(out_dir / material_name)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    marked_vertices = set(int(i) for i in metadata.get("group_a", []))
    marked_vertices.update(int(i) for i in metadata.get("group_b", []))

    frame_paths = sorted(in_dir.glob("frame_*.obj"))
    if not marked_vertices:
        print("[markers] no marked vertices in metadata; preserving OBJ materials")
        for i, src_path in enumerate(frame_paths):
            shutil.copy2(src_path, out_dir / src_path.name)
            if i % max(1, int(metadata.get("fps", 30))) == 0 or i == len(frame_paths) - 1:
                print(f"[markers] {i + 1}/{len(frame_paths)}")
        print(f"[markers] wrote {out_dir}")
        return

    for i, src_path in enumerate(frame_paths):
        mark_frame(src_path, out_dir / src_path.name, material_name, marked_vertices)
        if i % max(1, int(metadata.get("fps", 30))) == 0 or i == len(frame_paths) - 1:
            print(f"[markers] {i + 1}/{len(frame_paths)}")

    print(f"[markers] wrote {out_dir}")


def main():
    from main_offline import build_config

    mark_animation_obj_sequence(build_config())


if __name__ == "__main__":
    main()
