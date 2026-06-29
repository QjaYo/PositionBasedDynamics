from dataclasses import dataclass, field

import numpy as np

from utils import load_obj, tetrahedralize, build_edges


@dataclass
class SceneData:
    positions: np.ndarray
    inv_masses: np.ndarray
    edges: np.ndarray
    rest_lengths: np.ndarray
    tet_elems: np.ndarray
    rest_volumes: np.ndarray
    surface_faces: np.ndarray
    face_materials: list[str] = field(default_factory=list)
    bend_quads: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.int32))
    rest_bend_angles: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    groups: dict = field(default_factory=dict)


def create_scene(obj_path: str):
    """
    OBJ 파일을 로드해 기하 데이터를 반환
    반환:
        positions     : (N,3) float32 — 꼭짓점 초기 위치
        edges         : (E,2) int32   — 거리 제약용 엣지
        rest_lengths  : (E,)  float32
        tet_elems     : (M,4) int32   — 부피 제약용 사면체
        rest_volumes  : (M,)  float32
        surface_faces : (F,3) int32   — 렌더링용 표면 삼각형
    """

    # 로드 & 사면체화
    vertices, faces = load_obj(obj_path)
    tet_verts, tet_elems, surface_faces = tetrahedralize(vertices, faces)

    # 거리 제약 데이터
    edges = build_edges(tet_elems)
    p = tet_verts[edges[:, 0]]
    q = tet_verts[edges[:, 1]]
    rest_lengths = np.linalg.norm(p - q, axis=1).astype(np.float32)

    # 부피 제약 데이터
    v0 = tet_verts[tet_elems[:, 0]]
    v1 = tet_verts[tet_elems[:, 1]]
    v2 = tet_verts[tet_elems[:, 2]]
    v3 = tet_verts[tet_elems[:, 3]]
    rest_volumes = np.abs(np.einsum('ij,ij->i',
                    v1 - v0, np.cross(v2 - v0, v3 - v0)) / 6.0).astype(np.float32)

    print(f"[scene] verts={len(tet_verts)}, edges={len(edges)}, tets={len(tet_elems)}")
    print(f"[scene] rest_volumes: min={rest_volumes.min():.4e}, max={rest_volumes.max():.4e}, negative={(rest_volumes < 0).sum()}")

    return (tet_verts.astype(np.float32), edges, rest_lengths,
            tet_elems.astype(np.int32), rest_volumes, surface_faces)


def _empty_scene_data():
    return SceneData(
        positions=np.zeros((0, 3), dtype=np.float32),
        inv_masses=np.zeros(0, dtype=np.float32),
        edges=np.zeros((0, 2), dtype=np.int32),
        rest_lengths=np.zeros(0, dtype=np.float32),
        tet_elems=np.zeros((0, 4), dtype=np.int32),
        rest_volumes=np.zeros(0, dtype=np.float32),
        surface_faces=np.zeros((0, 3), dtype=np.int32),
        face_materials=[],
        bend_quads=np.zeros((0, 4), dtype=np.int32),
        rest_bend_angles=np.zeros(0, dtype=np.float32),
        groups={},
    )


def _rest_lengths(positions, edges):
    if len(edges) == 0:
        return np.zeros(0, dtype=np.float32)
    p = positions[edges[:, 0]]
    q = positions[edges[:, 1]]
    return np.linalg.norm(p - q, axis=1).astype(np.float32)


def _tet_rest_volumes(positions, tet_elems):
    if len(tet_elems) == 0:
        return np.zeros(0, dtype=np.float32)
    v0 = positions[tet_elems[:, 0]]
    v1 = positions[tet_elems[:, 1]]
    v2 = positions[tet_elems[:, 2]]
    v3 = positions[tet_elems[:, 3]]
    return np.abs(
        np.einsum("ij,ij->i", v1 - v0, np.cross(v2 - v0, v3 - v0)) / 6.0
    ).astype(np.float32)


def _mesh_edges(faces):
    edges = set()
    for a, b, c in np.asarray(faces, dtype=np.int32):
        for i, j in ((a, b), (b, c), (c, a)):
            edges.add((min(int(i), int(j)), max(int(i), int(j))))
    return np.array(sorted(edges), dtype=np.int32)


def _cloth_bending_constraints(faces):
    edge_to_face = {}
    for face_idx, face in enumerate(np.asarray(faces, dtype=np.int32)):
        for local_a, local_b in ((0, 1), (1, 2), (2, 0)):
            a = int(face[local_a])
            b = int(face[local_b])
            key = (min(a, b), max(a, b))
            third = int(face[3 - local_a - local_b])
            edge_to_face.setdefault(key, []).append((face_idx, third))

    quads = []
    for (a, b), refs in edge_to_face.items():
        if len(refs) != 2:
            continue
        quads.append((a, b, refs[0][1], refs[1][1]))

    quads = np.asarray(quads, dtype=np.int32)
    rest_angles = np.full(len(quads), np.pi, dtype=np.float32)
    return quads, rest_angles


def _cloth_grid(nx, ny, width, height, center, orientation, checker_cells=3):
    if nx < 2 or ny < 2:
        raise ValueError("cloth grid needs at least 2x2 corner vertices")

    center = np.asarray(center, dtype=np.float32)
    vertices = []
    for j in range(ny):
        v = j / max(1, ny - 1)
        for i in range(nx):
            u = i / max(1, nx - 1)
            x = (u - 0.5) * width
            h = (0.5 - v) * height
            if orientation == "vertical_xy":
                offset = np.array([x, h, 0.0], dtype=np.float32)
            elif orientation == "horizontal_xz":
                offset = np.array([x, 0.0, h], dtype=np.float32)
            else:
                raise ValueError(f"Unsupported cloth orientation: {orientation}")
            vertices.append(center + offset)

    faces = []
    materials = []
    checker_cells = max(1, int(checker_cells))
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            b = a + 1
            c = a + nx
            d = c + 1
            mid = len(vertices)
            vertices.append(0.25 * (vertices[a] + vertices[b] + vertices[c] + vertices[d]))

            # 셀마다 center vertex를 두고 십자형으로 4개의 삼각형을 만든다.
            faces.extend([(a, mid, b), (b, mid, d), (d, mid, c), (c, mid, a)])
            material = "cloth_red" if ((i // checker_cells) + (j // checker_cells)) % 2 == 0 else "cloth_gold"
            materials.extend([material, material, material, material])

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    edges = _mesh_edges(faces)
    bend_quads, rest_bend_angles = _cloth_bending_constraints(faces)
    return vertices, faces, materials, edges, _rest_lengths(vertices, edges), bend_quads, rest_bend_angles


def _box_mesh(center, half_extents):
    cx, cy, cz = np.asarray(center, dtype=np.float32)
    hx, hy, hz = np.asarray(half_extents, dtype=np.float32)
    vertices = np.array(
        [
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [3, 2, 6], [3, 6, 7],
            [1, 5, 6], [1, 6, 2],
            [0, 3, 7], [0, 7, 4],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def _append_mesh(scene, vertices, inv_mass, faces=None, face_materials=None, group_name=None):
    start = len(scene.positions)
    vertices = np.asarray(vertices, dtype=np.float32)
    count = len(vertices)
    scene.positions = np.vstack([scene.positions, vertices])
    scene.inv_masses = np.concatenate([
        scene.inv_masses,
        np.full(count, float(inv_mass), dtype=np.float32),
    ])
    if faces is not None and len(faces) > 0:
        shifted_faces = np.asarray(faces, dtype=np.int32) + start
        scene.surface_faces = np.vstack([scene.surface_faces, shifted_faces])
        if face_materials is None:
            scene.face_materials.extend(["bunny_gray"] * len(faces))
        else:
            scene.face_materials.extend(list(face_materials))
    if group_name is not None:
        scene.groups[group_name] = {
            "start": start,
            "end": start + count,
            "indices": np.arange(start, start + count, dtype=np.int32),
        }
    return start, start + count


def _append_edges(scene, edges, rest_lengths, offset):
    if len(edges) == 0:
        return
    scene.edges = np.vstack([scene.edges, np.asarray(edges, dtype=np.int32) + offset])
    scene.rest_lengths = np.concatenate([scene.rest_lengths, np.asarray(rest_lengths, dtype=np.float32)])


def _append_bends(scene, bend_quads, rest_angles, offset):
    if len(bend_quads) == 0:
        return
    scene.bend_quads = np.vstack([scene.bend_quads, np.asarray(bend_quads, dtype=np.int32) + offset])
    scene.rest_bend_angles = np.concatenate([scene.rest_bend_angles, np.asarray(rest_angles, dtype=np.float32)])


def _append_tet_object(scene, obj_path, material, inv_mass, group_name, offset=(0.0, 0.0, 0.0), align_floor_y=None):
    vertices, faces = load_obj(obj_path)
    tet_vertices, tet_elems, surface_faces = tetrahedralize(vertices, faces)
    tet_vertices = tet_vertices.astype(np.float32)
    offset = np.asarray(offset, dtype=np.float32)
    if align_floor_y is not None:
        offset = offset.copy()
        offset[1] += float(align_floor_y) - float(np.min(tet_vertices[:, 1]))
    tet_vertices = tet_vertices + offset

    start, end = _append_mesh(
        scene,
        tet_vertices,
        inv_mass=inv_mass,
        faces=surface_faces.astype(np.int32),
        face_materials=[material] * len(surface_faces),
        group_name=group_name,
    )
    shifted_tets = tet_elems.astype(np.int32) + start
    scene.tet_elems = np.vstack([scene.tet_elems, shifted_tets])
    scene.rest_volumes = np.concatenate([scene.rest_volumes, _tet_rest_volumes(scene.positions, shifted_tets)])

    edges = build_edges(tet_elems)
    _append_edges(scene, edges, _rest_lengths(tet_vertices, edges), start)
    scene.groups[group_name]["surface_faces"] = surface_faces.astype(np.int32) + start
    scene.groups[group_name]["tet_elems"] = shifted_tets
    print(f"[scene:{group_name}] verts={end - start}, edges={len(edges)}, tets={len(tet_elems)}")
    return start, end


def _append_static_obj(scene, obj_path, material, group_name, offset=(0.0, 0.0, 0.0), align_floor_y=None):
    vertices, faces = load_obj(obj_path)
    vertices = vertices.astype(np.float32)
    offset = np.asarray(offset, dtype=np.float32)
    if align_floor_y is not None:
        offset = offset.copy()
        offset[1] += float(align_floor_y) - float(np.min(vertices[:, 1]))
    vertices = vertices + offset
    start, end = _append_mesh(
        scene,
        vertices,
        inv_mass=0.0,
        faces=faces.astype(np.int32),
        face_materials=[material] * len(faces),
        group_name=group_name,
    )
    scene.groups[group_name]["surface_faces"] = faces.astype(np.int32) + start
    print(f"[scene:{group_name}] static verts={end - start}, faces={len(faces)}")
    return start, end


def _append_box(scene, center, half_extents, material, group_name):
    vertices, faces = _box_mesh(center, half_extents)
    start, end = _append_mesh(
        scene,
        vertices,
        inv_mass=0.0,
        faces=faces,
        face_materials=[material] * len(faces),
        group_name=group_name,
    )
    scene.groups[group_name]["center"] = np.asarray(center, dtype=np.float32)
    scene.groups[group_name]["half_extents"] = np.asarray(half_extents, dtype=np.float32)
    scene.groups[group_name]["local_vertices"] = vertices - np.asarray(center, dtype=np.float32)
    return start, end


def _append_cloth(scene, options, center):
    nx = int(options.get("cloth_nx", 28))
    ny = int(options.get("cloth_ny", 20))
    width = float(options.get("cloth_width", 0.28))
    height = float(options.get("cloth_height", 0.20))
    orientation = str(options.get("cloth_orientation", "vertical_xy"))
    checker_cells = int(options.get("cloth_checker_cells", 3))
    vertices, faces, materials, edges, rest_lengths, bend_quads, rest_angles = _cloth_grid(
        nx, ny, width, height, center, orientation, checker_cells
    )
    start, end = _append_mesh(
        scene,
        vertices,
        inv_mass=float(options.get("cloth_inv_mass", 1.0)),
        faces=faces,
        face_materials=materials,
        group_name="cloth",
    )
    _append_edges(scene, edges, rest_lengths, start)
    _append_bends(scene, bend_quads, rest_angles, start)
    top = np.array([start + i for i in range(nx)], dtype=np.int32)
    bottom = np.array([start + (ny - 1) * nx + i for i in range(nx)], dtype=np.int32)
    scene.groups["cloth"].update({
        "nx": nx,
        "ny": ny,
        "width": width,
        "height": height,
        "orientation": orientation,
        "checker_cells": checker_cells,
        "top_edge": top,
        "bottom_edge": bottom,
        "surface_faces": faces.astype(np.int32) + start,
        "local_vertices": vertices - np.asarray(center, dtype=np.float32),
    })
    print(f"[scene:cloth] verts={end - start}, edges={len(edges)}, bends={len(bend_quads)}")
    return start, end



def _shape_cloth_u(scene, cloth, top_attach_center, back_attach_center, low_y):
    indices = cloth["indices"]
    local = cloth["local_vertices"]
    width = max(float(cloth["width"]), 1e-8)
    height = max(float(cloth["height"]), 1e-8)
    top_attach_center = np.asarray(top_attach_center, dtype=np.float32)
    back_attach_center = np.asarray(back_attach_center, dtype=np.float32)
    low_y = float(low_y)

    top_local_y = 0.5 * height
    t = (top_local_y - local[:, 1]) / height
    t = np.clip(t, 0.0, 1.0).astype(np.float32)

    endpoint_y = (1.0 - t) * top_attach_center[1] + t * back_attach_center[1]
    endpoint_low = min(float(top_attach_center[1]), float(back_attach_center[1]))
    sag = max(0.0, endpoint_low - low_y)

    shaped = np.zeros_like(local, dtype=np.float32)
    shaped[:, 0] = top_attach_center[0] + local[:, 0]
    shaped[:, 1] = endpoint_y - sag * np.sin(np.pi * t)
    shaped[:, 2] = (1.0 - t) * top_attach_center[2] + t * back_attach_center[2]

    # Ensure the attached rows are exactly on their box attachment lines.
    nx = int(cloth["nx"])
    top_edge = cloth["top_edge"] - int(cloth["start"])
    bottom_edge = cloth["bottom_edge"] - int(cloth["start"])
    shaped[top_edge, 1] = top_attach_center[1]
    shaped[top_edge, 2] = top_attach_center[2]
    shaped[bottom_edge, 1] = back_attach_center[1]
    shaped[bottom_edge, 2] = back_attach_center[2]
    scene.positions[indices] = shaped
    return shaped


def create_offline_scene(config):
    options = getattr(config, "scenario_options", {})
    scene_kind = options.get("scene_kind", "tet")
    if scene_kind == "tet":
        positions, edges, rest_lengths, tet_elems, rest_volumes, surface_faces = create_scene(config.asset)
        return SceneData(
            positions=positions,
            inv_masses=np.ones(len(positions), dtype=np.float32),
            edges=edges,
            rest_lengths=rest_lengths,
            tet_elems=tet_elems,
            rest_volumes=rest_volumes,
            surface_faces=surface_faces,
            face_materials=["bunny_gray"] * len(surface_faces),
            groups={"deformable": {"start": 0, "end": len(positions), "indices": np.arange(len(positions), dtype=np.int32)}},
        )

    scene = _empty_scene_data()
    floor_y = float(config.floor_y)

    if scene_kind == "cloth_metal_drop":
        center = np.asarray(options.get("cloth_center", (0.0, 0.32, 0.0)), dtype=np.float32)
        _append_cloth(scene, options, center)
        box_half = np.asarray(options.get("box_half_extents", (0.08, 0.018, 0.035)), dtype=np.float32)
        cloth = scene.groups["cloth"]
        if options.get("match_box_width_to_cloth", False):
            box_half = box_half.copy()
            box_half[0] = 0.5 * float(cloth["width"])
        if options.get("cloth_initial_pose") == "u_drop":
            top_attach_y = float(options.get("u_top_y", center[1] + 0.5 * float(cloth["height"])))
            back_attach_y = float(options.get("u_back_y", top_attach_y))
            low_y = float(options.get("u_low_y", top_attach_y - 0.55 * float(cloth["height"])))
            back_z = float(center[2] + float(options.get("u_back_offset", -0.16)))
            top_attach = np.array([center[0], top_attach_y, center[2]], dtype=np.float32)
            back_attach = np.array([center[0], back_attach_y, back_z], dtype=np.float32)
            _shape_cloth_u(scene, cloth, top_attach, back_attach, low_y)
            _append_box(scene, (top_attach[0], top_attach[1] + box_half[1], top_attach[2]), box_half, "metal_dark", "static_box")
            _append_box(scene, (back_attach[0], back_attach[1] - box_half[1], back_attach[2]), box_half, "metal_dark", "dynamic_box")
        else:
            top_y = float(np.max(scene.positions[cloth["top_edge"], 1]))
            bottom_y = float(np.min(scene.positions[cloth["bottom_edge"], 1]))
            _append_box(scene, (center[0], top_y + box_half[1], center[2]), box_half, "metal_dark", "static_box")
            _append_box(scene, (center[0], bottom_y - box_half[1], center[2]), box_half, "metal_dark", "dynamic_box")
        return scene

    if scene_kind == "cloth_cover_bunny":
        bunny_asset = options.get("bunny_asset", "assets/bunny.obj")
        _append_static_obj(scene, bunny_asset, "bunny_gray", "static_bunny", align_floor_y=floor_y)
        bbox_min = np.min(scene.positions, axis=0)
        bbox_max = np.max(scene.positions, axis=0)
        center_y = bbox_max[1] + float(options.get("cloth_drop_height", 0.18))
        if options.get("cloth_center_y_mode") == "bunny_center_offset":
            center_y = 0.5 * (bbox_min[1] + bbox_max[1]) + float(options.get("cloth_center_y_offset", 0.22))
        center = np.array(
            [
                0.5 * (bbox_min[0] + bbox_max[0]),
                center_y,
                0.5 * (bbox_min[2] + bbox_max[2]),
            ],
            dtype=np.float32,
        )
        cloth_options = {**options, "cloth_orientation": "horizontal_xz"}
        if options.get("cloth_match_bunny_body_length", False):
            extent = bbox_max - bbox_min
            axis = str(options.get("cloth_body_length_axis", "x"))
            axis_index = {"x": 0, "y": 1, "z": 2}.get(axis, 0)
            body_length = float(extent[axis_index])
            cloth_options["cloth_width"] = body_length * float(options.get("cloth_body_length_scale", 1.0))
            cloth_options["cloth_height"] = body_length * float(options.get("cloth_body_depth_scale", options.get("cloth_body_length_scale", 1.0)))
        _append_cloth(scene, cloth_options, center)
        return scene

    if scene_kind == "cloth_moving_bunny":
        bunny_asset = options.get("bunny_asset", "assets/bunny.obj")
        bunny_start_offset = np.asarray(options.get("moving_bunny_start_offset", (0.0, 0.0, -0.22)), dtype=np.float32)
        _append_static_obj(
            scene,
            bunny_asset,
            "bunny_gray",
            "static_bunny",
            offset=bunny_start_offset,
            align_floor_y=floor_y,
        )

        bbox_min = np.min(scene.positions[scene.groups["static_bunny"]["indices"]], axis=0)
        bbox_max = np.max(scene.positions[scene.groups["static_bunny"]["indices"]], axis=0)
        bunny_center = 0.5 * (bbox_min + bbox_max)

        cloth_width = float(options.get("cloth_width", 0.28))
        cloth_height = float(options.get("cloth_height", 0.42))
        top_y = float(options.get("cloth_top_y", bbox_max[1] + 0.22))
        cloth_z = float(options.get("cloth_z", 0.0))
        cloth_center = np.array(
            [
                float(options.get("cloth_center_x", bunny_center[0])),
                top_y - 0.5 * cloth_height,
                cloth_z,
            ],
            dtype=np.float32,
        )
        cloth_options = {
            **options,
            "cloth_width": cloth_width,
            "cloth_height": cloth_height,
            "cloth_orientation": "vertical_xy",
        }
        _append_cloth(scene, cloth_options, cloth_center)
        cloth = scene.groups["cloth"]
        scene.inv_masses[cloth["top_edge"]] = 0.0

        box_half = np.asarray(
            options.get("top_box_half_extents", (0.5 * cloth_width, 0.018, 0.035)),
            dtype=np.float32,
        )
        if options.get("match_box_width_to_cloth", True):
            box_half = box_half.copy()
            box_half[0] = 0.5 * cloth_width
        _append_box(
            scene,
            (cloth_center[0], top_y + box_half[1], cloth_z),
            box_half,
            "metal_dark",
            "static_box",
        )
        return scene

    if scene_kind == "cloth_box_deformable_bunny":
        bunny_asset = options.get("bunny_asset", "assets/bunny.obj")
        _append_tet_object(scene, bunny_asset, "bunny_gray", 1.0, "deformable_bunny", align_floor_y=floor_y)
        bbox_min = np.min(scene.positions, axis=0)
        bbox_max = np.max(scene.positions, axis=0)
        box_half = np.asarray(options.get("box_half_extents", (0.08, 0.018, 0.035)), dtype=np.float32)
        center = np.array(
            [
                0.5 * (bbox_min[0] + bbox_max[0]),
                bbox_max[1] + float(options.get("cloth_drop_height", 0.28)),
                0.5 * (bbox_min[2] + bbox_max[2]),
            ],
            dtype=np.float32,
        )
        _append_cloth(scene, {**options, "cloth_orientation": "vertical_xy"}, center)
        cloth = scene.groups["cloth"]
        if options.get("match_box_width_to_cloth", False):
            box_half = box_half.copy()
            box_half[0] = 0.5 * float(cloth["width"])

        bunny_center = 0.5 * (bbox_min + bbox_max)
        if options.get("cloth_initial_pose") == "u_over_bunny":
            top_attach_y = float(bunny_center[1] + float(options.get("u_top_lift", 0.22)))
            back_attach_y = float(top_attach_y)
            back_z = float(bunny_center[2] + float(options.get("u_back_offset", -0.20)))
            top_attach = np.array([bunny_center[0], top_attach_y, bunny_center[2]], dtype=np.float32)
            back_attach = np.array([bunny_center[0], back_attach_y, back_z], dtype=np.float32)
            _shape_cloth_u(scene, cloth, top_attach, back_attach, float(bunny_center[1]))
            _append_box(scene, (top_attach[0], top_attach[1] + box_half[1], top_attach[2]), box_half, "metal_dark", "static_box")
            _append_box(scene, (back_attach[0], back_attach[1] - box_half[1], back_attach[2]), box_half, "metal_dark", "dynamic_box")
        else:
            top_y = float(np.max(scene.positions[cloth["top_edge"], 1]))
            bottom_y = float(np.min(scene.positions[cloth["bottom_edge"], 1]))
            _append_box(scene, (center[0], top_y + box_half[1], center[2]), box_half, "metal_dark", "static_box")
            _append_box(scene, (center[0], bottom_y - box_half[1], center[2]), box_half, "metal_dark", "dynamic_box")
        return scene

    raise ValueError(f"Unsupported offline scene_kind: {scene_kind}")
