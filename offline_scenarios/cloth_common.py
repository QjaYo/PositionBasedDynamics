import numpy as np
import taichi as ti

from engine.solver import static_collision_projection_jacobi
from engine.static_mesh_collision import StaticTriangleGrid
from engine.rigid import (
    RigidBoxes,
    apply_rigid_external_forces,
    apply_rigid_floor_velocity_response,
    attachment_projection_jacobi,
    particle_box_collision_projection_jacobi,
    rigid_corner_cloth_collision_projection_jacobi,
    predict_rigid_boxes,
    rigid_floor_projection_jacobi,
    sync_rigid_box_particles,
    update_rigid_velocities,
)


PREPARE_ACCEPTS_SCENE_DATA = True
EPSILON = 1e-8


def _as_vec3(value):
    return np.asarray(value, dtype=np.float32)


def _vector_field(dim, dtype, array):
    array = np.asarray(array)
    field = ti.Vector.field(dim, dtype=dtype, shape=len(array))
    field.from_numpy(array)
    return field


def _scalar_field(dtype, array):
    array = np.asarray(array)
    field = ti.field(dtype=dtype, shape=len(array))
    field.from_numpy(array)
    return field


@ti.kernel
def _sync_moving_static_mesh_particles(
    particles: ti.template(),
    indices: ti.template(),
    rest_positions: ti.template(),
    count: ti.i32,
    offset_field: ti.template(),
):
    offset = offset_field[None]
    for i in range(count):
        idx = indices[i]
        x = rest_positions[i] + offset
        particles.x[idx] = x
        particles.x_pred[idx] = x
        particles.v[idx] = ti.math.vec3(0.0, 0.0, 0.0)


def _per_iter_stiffness(k, solver_iters):
    k = min(max(float(k), 0.0), 1.0)
    solver_iters = max(1, int(solver_iters))
    return float(1.0 - (1.0 - k) ** (1.0 / solver_iters))


def _closest_point_on_triangle(p, a, b, c):
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab

    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w


def _triangle_normal(a, b, c):
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n)
    if norm < EPSILON:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return (n / norm).astype(np.float32)


def _build_triangle_grid(vertices, faces, thickness, cell_size):
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    tri_min = np.min(vertices[faces], axis=1) - thickness
    tri_max = np.max(vertices[faces], axis=1) + thickness
    origin = np.min(tri_min, axis=0).astype(np.float32)
    inv_cell = 1.0 / max(float(cell_size), EPSILON)
    grid = {}
    for face_idx, (mn, mx) in enumerate(zip(tri_min, tri_max)):
        lo = np.floor((mn - origin) * inv_cell).astype(np.int32)
        hi = np.floor((mx - origin) * inv_cell).astype(np.int32)
        for ix in range(int(lo[0]), int(hi[0]) + 1):
            for iy in range(int(lo[1]), int(hi[1]) + 1):
                for iz in range(int(lo[2]), int(hi[2]) + 1):
                    grid.setdefault((ix, iy, iz), []).append(face_idx)
    return {
        "vertices": vertices,
        "faces": faces,
        "origin": origin,
        "inv_cell": inv_cell,
        "grid": grid,
        "thickness": float(thickness),
    }


def _grid_candidates(grid_data, p):
    key = np.floor((p - grid_data["origin"]) * grid_data["inv_cell"]).astype(np.int32)
    candidates = []
    grid = grid_data["grid"]
    for ix in range(int(key[0]) - 1, int(key[0]) + 2):
        for iy in range(int(key[1]) - 1, int(key[1]) + 2):
            for iz in range(int(key[2]) - 1, int(key[2]) + 2):
                candidates.extend(grid.get((ix, iy, iz), ()))
    if not candidates:
        return ()
    return set(candidates)


def _generate_static_mesh_collision_constraints(state, particles):
    static_grid = state.get("static_collision_gpu_grid")
    if static_grid is not None:
        if state.get("moving_static_collision", False):
            static_grid.generate_constraints_with_offsets(
                particles,
                state["static_collision_particle_indices_field"],
                int(state["static_collision_num_particles"]),
                state["moving_bunny_prev_offset_field"],
                state["moving_bunny_current_offset_field"],
                state["static_collision_count_field"],
                state["static_collision_particle_indices_out_field"],
                state["static_collision_points_field"],
                state["static_collision_normals_field"],
                int(state["static_collision_max_constraints"]),
            )
        else:
            static_grid.generate_constraints(
                particles,
                state["static_collision_particle_indices_field"],
                int(state["static_collision_num_particles"]),
                state["static_collision_count_field"],
                state["static_collision_particle_indices_out_field"],
                state["static_collision_points_field"],
                state["static_collision_normals_field"],
                int(state["static_collision_max_constraints"]),
            )
        return

    grid_data = state.get("static_collision_grid")
    if grid_data is None:
        return

    x_pred = particles.x_pred.to_numpy()
    vertices = grid_data["vertices"]
    faces = grid_data["faces"]
    thickness = grid_data["thickness"]
    out_indices = state["static_collision_indices_np"]
    out_points = state["static_collision_points_np"]
    out_normals = state["static_collision_normals_np"]
    count = 0

    for idx in state["static_collision_particle_indices"]:
        if count >= len(out_indices):
            break
        idx = int(idx)
        p = x_pred[idx]
        best_distance = float("inf")
        best_point = None
        best_normal = None
        for face_idx in _grid_candidates(grid_data, p):
            face = faces[int(face_idx)]
            a, b, c = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            closest = _closest_point_on_triangle(p, a, b, c)
            delta = p - closest
            distance = float(np.linalg.norm(delta))
            if distance < best_distance:
                best_distance = distance
                best_point = closest
                best_normal = delta / distance if distance > EPSILON else _triangle_normal(a, b, c)

        if best_point is None or best_distance >= thickness:
            continue
        out_indices[count] = idx
        out_points[count] = (best_point + best_normal.astype(np.float32) * thickness).astype(np.float32)
        out_normals[count] = best_normal.astype(np.float32)
        count += 1

    state["static_collision_count_field"][None] = count
    if count > 0:
        state["static_collision_particle_indices_field"].from_numpy(out_indices)
        state["static_collision_points_field"].from_numpy(out_points)
        state["static_collision_normals_field"].from_numpy(out_normals)


def _barycentric_for_point(p, a, b, c):
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = float(np.dot(v2, v0))
    d21 = float(np.dot(v2, v1))
    denom = d00 * d11 - d01 * d01
    if abs(denom) < EPSILON:
        return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    bary = np.clip(np.array([u, v, w], dtype=np.float32), 0.0, 1.0)
    total = float(np.sum(bary))
    if total < EPSILON:
        return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)
    return bary / total


def _box_corner_offsets(half_extents):
    hx, hy, hz = np.asarray(half_extents, dtype=np.float32)
    return np.asarray(
        [
            [sx * hx, sy * hy, sz * hz]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float32,
    )


def _generate_rigid_corner_cloth_collision_constraints(state, particles):
    if "corner_cloth_faces" not in state:
        return

    boxes = state["rigid_boxes"]
    box_id = int(state["corner_cloth_box_id"])
    centers = boxes.pred_center.to_numpy()
    rotations = boxes.pred_rotation.to_numpy()
    half_extents = boxes.half_extents.to_numpy()
    x_pred = particles.x_pred.to_numpy()
    faces = state["corner_cloth_faces"]
    thickness = float(state["corner_cloth_collision_thickness"])

    out_box_ids = state["corner_cloth_box_ids_np"]
    out_corners = state["corner_cloth_local_corners_np"]
    out_tris = state["corner_cloth_tri_indices_np"]
    out_bary = state["corner_cloth_bary_np"]
    out_normals = state["corner_cloth_normals_np"]
    count = 0

    center = centers[box_id]
    rotation = rotations[box_id]
    for local_corner in _box_corner_offsets(half_extents[box_id]):
        if count >= len(out_box_ids):
            break
        corner = center + rotation @ local_corner
        best_distance = float("inf")
        best_face = None
        best_bary = None
        best_normal = None

        for face in faces:
            a = x_pred[int(face[0])]
            b = x_pred[int(face[1])]
            c = x_pred[int(face[2])]
            closest = _closest_point_on_triangle(corner, a, b, c)
            normal = _triangle_normal(a, b, c)
            signed_distance = float(np.dot(corner - closest, normal))
            if signed_distance < 0.0:
                normal = -normal
                signed_distance = -signed_distance
            if signed_distance < best_distance:
                best_distance = signed_distance
                best_face = face
                best_bary = _barycentric_for_point(closest, a, b, c)
                best_normal = normal

        if best_face is None or best_distance >= thickness:
            continue
        out_box_ids[count] = box_id
        out_corners[count] = local_corner
        out_tris[count] = best_face
        out_bary[count] = best_bary.astype(np.float32)
        out_normals[count] = best_normal.astype(np.float32)
        count += 1

    state["corner_cloth_count_field"][None] = count
    if count > 0:
        state["corner_cloth_box_ids_field"].from_numpy(out_box_ids)
        state["corner_cloth_local_corners_field"].from_numpy(out_corners)
        state["corner_cloth_tri_indices_field"].from_numpy(out_tris)
        state["corner_cloth_bary_field"].from_numpy(out_bary)
        state["corner_cloth_normals_field"].from_numpy(out_normals)

def _build_rigid_state(config, scene, state):
    options = getattr(config, "scenario_options", {})
    groups = scene.groups
    box_names = []
    centers = []
    halves = []
    masses = []
    velocities = []

    if "static_box" in groups:
        box = groups["static_box"]
        box_names.append("static_box")
        centers.append(box["center"])
        halves.append(box["half_extents"])
        masses.append(0.0)
        velocities.append((0.0, 0.0, 0.0))

    if "dynamic_box" in groups:
        box = groups["dynamic_box"]
        box_names.append("dynamic_box")
        centers.append(box["center"])
        halves.append(box["half_extents"])
        masses.append(float(options.get("dynamic_box_mass", 8.0)))
        velocities.append(_as_vec3(options.get("dynamic_box_initial_velocity", (0.0, 0.0, 0.0))))

    if not box_names:
        return

    boxes = RigidBoxes(centers, halves, masses, velocities=velocities)
    box_ids = {name: i for i, name in enumerate(box_names)}
    state["rigid_boxes"] = boxes
    state["rigid_box_ids"] = box_ids
    state["rigid_box_names"] = box_names
    state["dynamic_box_mass"] = float(options.get("dynamic_box_mass", 8.0)) if "dynamic_box" in box_ids else None
    state["box_collision_margin"] = float(options.get("box_collision_margin", 0.002))
    state["box_floor_margin"] = float(options.get("box_floor_margin", 0.0))
    state["box_floor_restitution"] = float(options.get("box_floor_restitution", 0.08))
    state["box_attachment_k"] = float(options.get("box_attachment_k", 1.0))
    state["box_collision_k"] = float(options.get("box_collision_k", 1.0))
    state["box_floor_k"] = float(options.get("box_floor_k", 1.0))

    visual_particle_indices = []
    visual_box_ids = []
    visual_local_vertices = []
    for name, box_id in box_ids.items():
        box = groups[name]
        visual_particle_indices.extend(box["indices"])
        visual_box_ids.extend([box_id] * len(box["indices"]))
        visual_local_vertices.extend(box["local_vertices"])

    state["box_visual_count"] = len(visual_particle_indices)
    state["box_visual_particle_indices_field"] = _scalar_field(ti.i32, np.asarray(visual_particle_indices, dtype=np.int32))
    state["box_visual_box_ids_field"] = _scalar_field(ti.i32, np.asarray(visual_box_ids, dtype=np.int32))
    state["box_visual_local_vertices_field"] = _vector_field(3, ti.f32, np.asarray(visual_local_vertices, dtype=np.float32))

    attachment_indices = []
    attachment_box_ids = []
    attachment_local_offsets = []
    if "static_box" in box_ids:
        center = groups["static_box"]["center"]
        for idx in groups["cloth"]["top_edge"]:
            attachment_indices.append(int(idx))
            attachment_box_ids.append(box_ids["static_box"])
            attachment_local_offsets.append(scene.positions[int(idx)] - center)
    if "dynamic_box" in box_ids:
        center = groups["dynamic_box"]["center"]
        for idx in groups["cloth"]["bottom_edge"]:
            attachment_indices.append(int(idx))
            attachment_box_ids.append(box_ids["dynamic_box"])
            attachment_local_offsets.append(scene.positions[int(idx)] - center)

    if attachment_indices:
        state["attachment_count"] = len(attachment_indices)
        state["attachment_particle_indices_field"] = _scalar_field(ti.i32, np.asarray(attachment_indices, dtype=np.int32))
        state["attachment_box_ids_field"] = _scalar_field(ti.i32, np.asarray(attachment_box_ids, dtype=np.int32))
        state["attachment_local_offsets_field"] = _vector_field(3, ti.f32, np.asarray(attachment_local_offsets, dtype=np.float32))

    if "deformable_bunny" in groups and "dynamic_box" in box_ids:
        bunny = groups["deformable_bunny"]
        state["box_collision_range"] = (int(bunny["start"]), int(bunny["end"]))
        state["box_collision_box_id"] = int(box_ids["dynamic_box"])

    if "dynamic_box" in box_ids and "surface_faces" in groups["cloth"]:
        attached = set(int(i) for i in groups["cloth"].get("bottom_edge", ()))
        cloth_faces = []
        for face in groups["cloth"]["surface_faces"]:
            if any(int(v) in attached for v in face):
                continue
            cloth_faces.append(face)
        state["corner_cloth_faces"] = np.asarray(cloth_faces, dtype=np.int32)
        state["corner_cloth_box_id"] = int(box_ids["dynamic_box"])
        state["corner_cloth_collision_thickness"] = float(options.get("rigid_corner_collision_thickness", state["collision_thickness"]))
        state["corner_cloth_collision_k"] = float(options.get("rigid_corner_collision_k", 1.0))
        max_corner_constraints = 8
        state["corner_cloth_box_ids_np"] = np.zeros(max_corner_constraints, dtype=np.int32)
        state["corner_cloth_local_corners_np"] = np.zeros((max_corner_constraints, 3), dtype=np.float32)
        state["corner_cloth_tri_indices_np"] = np.zeros((max_corner_constraints, 3), dtype=np.int32)
        state["corner_cloth_bary_np"] = np.zeros((max_corner_constraints, 3), dtype=np.float32)
        state["corner_cloth_normals_np"] = np.zeros((max_corner_constraints, 3), dtype=np.float32)
        state["corner_cloth_count_field"] = ti.field(dtype=ti.i32, shape=())
        state["corner_cloth_box_ids_field"] = ti.field(dtype=ti.i32, shape=max_corner_constraints)
        state["corner_cloth_local_corners_field"] = ti.Vector.field(3, dtype=ti.f32, shape=max_corner_constraints)
        state["corner_cloth_tri_indices_field"] = ti.Vector.field(3, dtype=ti.i32, shape=max_corner_constraints)
        state["corner_cloth_bary_field"] = ti.Vector.field(3, dtype=ti.f32, shape=max_corner_constraints)
        state["corner_cloth_normals_field"] = ti.Vector.field(3, dtype=ti.f32, shape=max_corner_constraints)
        state["corner_cloth_count_field"][None] = 0


def _sync_box_visuals(state, particles):
    if "rigid_boxes" not in state:
        return
    sync_rigid_box_particles(
        particles,
        state["rigid_boxes"],
        state["box_visual_particle_indices_field"],
        state["box_visual_box_ids_field"],
        state["box_visual_local_vertices_field"],
        int(state["box_visual_count"]),
    )


def _moving_bunny_offset(state, time):
    mode = state.get("moving_bunny_motion", "linear")
    if mode in {"cosine_x", "cosine_z"}:
        amplitude = float(state.get("moving_bunny_amplitude", 0.1))
        period = max(float(state.get("moving_bunny_period", 5.0)), EPSILON)
        value = amplitude * np.cos(2.0 * np.pi * float(time) / period)
        if mode == "cosine_z":
            return np.array([0.0, 0.0, value], dtype=np.float32)
        return np.array([value, 0.0, 0.0], dtype=np.float32)
    velocity = state["moving_bunny_velocity"]
    return (velocity * float(time)).astype(np.float32)


def _sync_moving_static_bunny(state, particles, time):
    if not state.get("moving_static_collision", False):
        return
    offset = _moving_bunny_offset(state, time)
    state["moving_bunny_prev_offset_field"][None] = offset
    state["moving_bunny_current_offset_field"][None] = offset
    _sync_moving_static_mesh_particles(
        particles,
        state["moving_bunny_indices_field"],
        state["moving_bunny_rest_positions_field"],
        int(state["moving_bunny_count"]),
        state["moving_bunny_current_offset_field"],
    )


def _advance_moving_static_bunny(state, particles, dt):
    if not state.get("moving_static_collision", False):
        return
    time = float(state.get("moving_bunny_time", 0.0))
    prev_offset = _moving_bunny_offset(state, time)
    current_offset = _moving_bunny_offset(state, time + float(dt))
    state["moving_bunny_prev_offset_field"][None] = prev_offset
    state["moving_bunny_current_offset_field"][None] = current_offset
    _sync_moving_static_mesh_particles(
        particles,
        state["moving_bunny_indices_field"],
        state["moving_bunny_rest_positions_field"],
        int(state["moving_bunny_count"]),
        state["moving_bunny_current_offset_field"],
    )
    state["moving_bunny_time"] = time + float(dt)


def prepare(config, scene, bbox_diag):
    options = getattr(config, "scenario_options", {})
    groups = scene.groups
    state = {
        "kind": "cloth",
        "scenario": config.scenario,
        "scene_kind": options.get("scene_kind"),
        "initialized": False,
        "cloth_indices": groups["cloth"]["indices"],
        "cloth_top_edge": groups["cloth"]["top_edge"],
        "cloth_bottom_edge": groups["cloth"]["bottom_edge"],
        "cloth_nx": int(groups["cloth"]["nx"]),
        "cloth_ny": int(groups["cloth"]["ny"]),
        "cloth_width": float(groups["cloth"]["width"]),
        "cloth_height": float(groups["cloth"]["height"]),
        "cloth_orientation": groups["cloth"]["orientation"],
        "cloth_checker_cells": int(groups["cloth"].get("checker_cells", options.get("cloth_checker_cells", 3))),
        "k_bending": float(options.get("k_bending", 0.35)),
        "collision_thickness": float(options.get("collision_thickness", 0.004)),
    }

    _build_rigid_state(config, scene, state)

    if "static_bunny" in groups:
        faces = groups["static_bunny"]["surface_faces"]
        cell_size = float(options.get("collision_cell_size", max(0.012, 4.0 * state["collision_thickness"])))
        max_tris_per_cell = int(options.get("static_collision_max_tris_per_cell", 256))
        neighbor_range = int(options.get("static_collision_neighbor_range", 1))
        state["static_collision_gpu_grid"] = StaticTriangleGrid(
            scene.positions,
            faces,
            state["collision_thickness"],
            cell_size,
            max_tris_per_cell=max_tris_per_cell,
            neighbor_range=neighbor_range,
        )
        state["static_collision_particle_indices"] = groups["cloth"]["indices"]
        state["collision_cell_size"] = cell_size
        state["static_collision_max_tris_per_cell"] = max_tris_per_cell
        state["static_collision_grid_overflow"] = int(state["static_collision_gpu_grid"].overflow)
        max_collisions = len(state["static_collision_particle_indices"])
        state["static_collision_num_particles"] = max_collisions
        state["static_collision_max_constraints"] = max_collisions
        state["static_collision_count_field"] = ti.field(dtype=ti.i32, shape=())
        state["static_collision_particle_indices_field"] = _scalar_field(
            ti.i32,
            np.asarray(state["static_collision_particle_indices"], dtype=np.int32),
        )
        state["static_collision_particle_indices_out_field"] = ti.field(dtype=ti.i32, shape=max_collisions)
        state["static_collision_points_field"] = ti.Vector.field(3, dtype=ti.f32, shape=max_collisions)
        state["static_collision_normals_field"] = ti.Vector.field(3, dtype=ti.f32, shape=max_collisions)
        state["static_collision_count_field"][None] = 0
        state["static_collision_k"] = float(options.get("static_collision_k", 1.0))

    if options.get("moving_static_bunny", False) and "static_bunny" in groups:
        bunny_indices = groups["static_bunny"]["indices"]
        state["moving_static_collision"] = True
        state["moving_bunny_time"] = 0.0
        state["moving_bunny_velocity"] = _as_vec3(options.get("moving_bunny_velocity", (0.0, 0.0, 0.0)))
        state["moving_bunny_motion"] = str(options.get("moving_bunny_motion", "linear"))
        state["moving_bunny_amplitude"] = float(options.get("moving_bunny_amplitude", 0.1))
        state["moving_bunny_period"] = float(options.get("moving_bunny_period", max(1.0, float(config.frames) / float(config.fps))))
        state["moving_bunny_indices_field"] = _scalar_field(ti.i32, np.asarray(bunny_indices, dtype=np.int32))
        state["moving_bunny_rest_positions_field"] = _vector_field(3, ti.f32, scene.positions[bunny_indices].astype(np.float32))
        state["moving_bunny_count"] = len(bunny_indices)
        state["moving_bunny_prev_offset_field"] = ti.Vector.field(3, dtype=ti.f32, shape=())
        state["moving_bunny_current_offset_field"] = ti.Vector.field(3, dtype=ti.f32, shape=())
        zero = ti.Vector([0.0, 0.0, 0.0])
        state["moving_bunny_prev_offset_field"][None] = zero
        state["moving_bunny_current_offset_field"][None] = zero

    return state


def register_solver_constraints(config, scenario_state, particles, solver):
    boxes = scenario_state.get("rigid_boxes")
    solver_iters = int(config.solver_iters)

    if "static_collision_count_field" in scenario_state:
        solver.register_jacobi(
            static_collision_projection_jacobi,
            particles,
            scenario_state["static_collision_particle_indices_out_field"],
            scenario_state["static_collision_points_field"],
            scenario_state["static_collision_normals_field"],
            scenario_state["static_collision_count_field"],
            _per_iter_stiffness(scenario_state.get("static_collision_k", 1.0), solver_iters),
        )

    if boxes is None:
        return

    solver.register_rigid_system(boxes)

    if scenario_state.get("attachment_count", 0) > 0:
        solver.register_jacobi(
            attachment_projection_jacobi,
            particles,
            boxes,
            scenario_state["attachment_particle_indices_field"],
            scenario_state["attachment_box_ids_field"],
            scenario_state["attachment_local_offsets_field"],
            int(scenario_state["attachment_count"]),
            _per_iter_stiffness(scenario_state.get("box_attachment_k", 1.0), solver_iters),
        )

    if "box_collision_range" in scenario_state:
        start, end = scenario_state["box_collision_range"]
        solver.register_jacobi(
            particle_box_collision_projection_jacobi,
            particles,
            boxes,
            int(start),
            int(end),
            int(scenario_state["box_collision_box_id"]),
            float(scenario_state.get("box_collision_margin", 0.002)),
            _per_iter_stiffness(scenario_state.get("box_collision_k", 1.0), solver_iters),
        )

    if "corner_cloth_count_field" in scenario_state:
        solver.register_jacobi(
            rigid_corner_cloth_collision_projection_jacobi,
            particles,
            boxes,
            scenario_state["corner_cloth_box_ids_field"],
            scenario_state["corner_cloth_local_corners_field"],
            scenario_state["corner_cloth_tri_indices_field"],
            scenario_state["corner_cloth_bary_field"],
            scenario_state["corner_cloth_normals_field"],
            scenario_state["corner_cloth_count_field"],
            float(scenario_state.get("corner_cloth_collision_thickness", scenario_state.get("collision_thickness", 0.004))),
            _per_iter_stiffness(scenario_state.get("corner_cloth_collision_k", 1.0), solver_iters),
        )

    solver.register_jacobi(
        rigid_floor_projection_jacobi,
        boxes,
        float(config.floor_y),
        float(scenario_state.get("box_floor_margin", 0.0)),
        _per_iter_stiffness(scenario_state.get("box_floor_k", 1.0), solver_iters),
    )


def apply_frame(config, scenario_state, particles, frame):
    if scenario_state["initialized"]:
        return
    _sync_box_visuals(scenario_state, particles)
    _sync_moving_static_bunny(scenario_state, particles, 0.0)
    scenario_state["initialized"] = True


def apply_substep(config, scenario_state, particles, dt, stage):
    boxes = scenario_state.get("rigid_boxes")
    if stage == "pre_predict":
        _advance_moving_static_bunny(scenario_state, particles, dt)
        if boxes is not None:
            if config.use_gravity:
                apply_rigid_external_forces(boxes, dt * float(getattr(config, "gravity_scale", 1.0)))
            predict_rigid_boxes(boxes, dt)
    elif stage == "post_predict":
        _generate_static_mesh_collision_constraints(scenario_state, particles)
        _generate_rigid_corner_cloth_collision_constraints(scenario_state, particles)
    elif stage == "post_velocity" and boxes is not None:
        update_rigid_velocities(boxes, dt)
        apply_rigid_floor_velocity_response(
            boxes,
            float(config.floor_y),
            float(scenario_state.get("box_floor_restitution", 0.08)),
            float(config.mu_k),
        )
        _sync_box_visuals(scenario_state, particles)


def clear(config, scenario_state, particles):
    return None


def metadata(config, scenario_state):
    data = {
        "anchor_mode": "cloth_rigid_constraints",
        "scene_kind": scenario_state["scene_kind"],
        "cloth": {
            "nx": scenario_state["cloth_nx"],
            "ny": scenario_state["cloth_ny"],
            "width": scenario_state["cloth_width"],
            "height": scenario_state["cloth_height"],
            "orientation": scenario_state["cloth_orientation"],
            "checker_cells": int(scenario_state["cloth_checker_cells"]),
            "top_edge": [int(i) for i in scenario_state["cloth_top_edge"]],
            "bottom_edge": [int(i) for i in scenario_state["cloth_bottom_edge"]],
            "k_bending": float(scenario_state["k_bending"]),
            "rest_bending_flat_angle_degrees": 180.0,
        },
        "collision_thickness": float(scenario_state["collision_thickness"]),
    }
    if "rigid_boxes" in scenario_state:
        data["rigid_boxes"] = {
            "names": list(scenario_state["rigid_box_names"]),
            "dynamic_box_mass": scenario_state.get("dynamic_box_mass"),
            "box_attachment_k": float(scenario_state.get("box_attachment_k", 1.0)),
            "box_collision_k": float(scenario_state.get("box_collision_k", 1.0)),
            "box_floor_k": float(scenario_state.get("box_floor_k", 1.0)),
            "rigid_corner_cloth_collision": "corner_cloth_count_field" in scenario_state,
        }
    if "static_collision_gpu_grid" in scenario_state:
        data["static_triangle_collision"] = {
            "enabled": True,
            "backend": "taichi_static_triangle_grid",
            "cell_size": float(scenario_state["collision_cell_size"]),
            "max_tris_per_cell": int(scenario_state["static_collision_max_tris_per_cell"]),
            "overflow": int(scenario_state["static_collision_grid_overflow"]),
        }
    elif "static_collision_grid" in scenario_state:
        data["static_triangle_collision"] = {
            "enabled": True,
            "cell_size": float(scenario_state["collision_cell_size"]),
        }
    return data


def log_lines(config, scenario_state):
    lines = [
        "anchor mode: cloth_rigid_constraints",
        f"scene kind: {scenario_state['scene_kind']}",
        f"cloth grid: {scenario_state['cloth_nx']}x{scenario_state['cloth_ny']}",
        f"cloth size: {scenario_state['cloth_width']:.4f} x {scenario_state['cloth_height']:.4f}",
        f"cloth orientation: {scenario_state['cloth_orientation']}",
        f"cloth checker cells: {scenario_state['cloth_checker_cells']}",
        f"bending k: {scenario_state['k_bending']:.4f} rest flat angle=180deg",
    ]
    if "rigid_boxes" in scenario_state:
        lines.append(f"rigid boxes: {', '.join(scenario_state['rigid_box_names'])}")
        if scenario_state.get("dynamic_box_mass") is not None:
            lines.append(f"dynamic box mass: {scenario_state['dynamic_box_mass']:.4f} inv_mass={1.0 / scenario_state['dynamic_box_mass']:.4f}")
        lines.append(f"rigid attachment k: {scenario_state.get('box_attachment_k', 1.0):.4f}")
        if "corner_cloth_count_field" in scenario_state:
            lines.append(f"rigid corner cloth collision thickness: {scenario_state.get('corner_cloth_collision_thickness', scenario_state['collision_thickness']):.4f}")
    if "static_collision_gpu_grid" in scenario_state:
        lines.append(
            "static triangle collision: taichi grid "
            f"cell={scenario_state['collision_cell_size']:.4f} "
            f"max/cell={scenario_state['static_collision_max_tris_per_cell']} "
            f"overflow={scenario_state['static_collision_grid_overflow']}"
        )
    elif "static_collision_grid" in scenario_state:
        lines.append(f"static triangle collision thickness: {scenario_state['collision_thickness']:.4f}")
    if "box_collision_range" in scenario_state:
        lines.append(f"dynamic box collision range: {scenario_state['box_collision_range']}")
    return lines
