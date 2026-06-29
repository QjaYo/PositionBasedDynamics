import numpy as np
import taichi as ti

import simulation_config as sim_config


PATCH_RADIUS = 0.015
MIN_PATCH_SIZE = 8


SCENARIO_CONFIG = {
    "output": "bunny_pull_tearing",
    "enable_tearing": True,
    "asset": "assets/bunny.obj",
    "use_gravity": True,
    "fixed_anchor_indices": (1208, 1167),  # bunny back1 / back2
    "tear_ratio": 1.2,
    "max_tears_per_call": 1000,
    "seed": 7,
    "patch_radius": PATCH_RADIUS,  # None이면 bbox diagonal 기준 자동값 사용.
    "auto_patch_radius_ratio": 0.08,
    "random_anchor_min_distance_ratio": 0.45,
    "min_patch_size": MIN_PATCH_SIZE,
    "pull_distance": 0.50,  # None이면 bbox diagonal 기준 자동값 사용.
    "auto_pull_distance_ratio": 0.45,
    "pull_horizontal": True,
    "pull_down_angle_degrees": None,  # 예: 45.0이면 좌우 + 아래 45도.
    "frames": 200,
    "fps": 30,
    "substeps": 10,
    "camera": {
        "bounds_mode": "first_frame",
        "target_mode": "bunny",
        "distance_scale": 5.0,
        "height_scale": 0.28,
        "side_scale": 0.35,
        "lens": 55,
        "bounds_padding": 1.0,
    },
}


def triangle_areas(vertices, faces):
    tri = vertices[faces]
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def sample_surface_point(vertices, faces, rng):
    areas = triangle_areas(vertices, faces)
    total_area = float(np.sum(areas))
    if total_area <= 1e-12:
        idx = int(rng.integers(0, len(faces)))
    else:
        idx = int(rng.choice(len(faces), p=areas / total_area))

    tri = vertices[faces[idx]]
    r1 = float(rng.random())
    r2 = float(rng.random())
    sr1 = np.sqrt(r1)
    weights = np.array([1.0 - sr1, sr1 * (1.0 - r2), sr1 * r2], dtype=np.float32)
    return weights @ tri, faces[idx], idx, weights


def surface_patch(surface_ids, vertices, center, radius, min_size, exclude=None):
    if exclude is None:
        eligible = surface_ids
    else:
        eligible = np.setdiff1d(surface_ids, np.asarray(exclude, dtype=np.int32), assume_unique=False)
    if len(eligible) == 0:
        return np.zeros(0, dtype=np.int32)

    distances = np.linalg.norm(vertices[eligible] - center, axis=1)
    group = eligible[distances <= radius]
    if len(group) >= min_size:
        return group.astype(np.int32)

    nearest = np.argsort(distances)[:min(min_size, len(eligible))]
    return eligible[nearest].astype(np.int32)


def choose_pull_patches(vertices, surface_faces, rng, radius, min_distance, min_size):
    surface_ids = np.unique(surface_faces.reshape(-1)).astype(np.int32)
    point_a, face_a, face_idx_a, weights_a = sample_surface_point(vertices, surface_faces, rng)

    best_point = None
    best_face = None
    best_face_idx = -1
    best_weights = None
    best_distance = -1.0
    for _ in range(256):
        point_b, face_b, face_idx_b, weights_b = sample_surface_point(vertices, surface_faces, rng)
        distance = float(np.linalg.norm(point_b - point_a))
        if distance > best_distance:
            best_point = point_b
            best_face = face_b
            best_face_idx = face_idx_b
            best_weights = weights_b
            best_distance = distance
        if distance >= min_distance:
            break

    group_a = surface_patch(surface_ids, vertices, point_a, radius, min_size)
    group_b = surface_patch(surface_ids, vertices, best_point, radius, min_size, exclude=group_a)

    direction = best_point - point_a
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-8:
        axis = int(np.argmax(np.ptp(vertices, axis=0)))
        direction = np.zeros(3, dtype=np.float32)
        direction[axis] = 1.0
    else:
        direction = direction / direction_norm

    return {
        "point_a": point_a.astype(np.float32),
        "point_b": best_point.astype(np.float32),
        "face_a": face_a.astype(np.int32),
        "face_b": best_face.astype(np.int32),
        "face_idx_a": int(face_idx_a),
        "face_idx_b": int(best_face_idx),
        "barycentric_a": weights_a.astype(np.float32),
        "barycentric_b": best_weights.astype(np.float32),
        "group_a": group_a,
        "group_b": group_b,
        "direction": direction.astype(np.float32),
        "distance": float(best_distance),
    }


def surface_face_for_vertex(surface_faces, vertex_idx):
    matches = np.nonzero(np.any(surface_faces == vertex_idx, axis=1))[0]
    if len(matches) == 0:
        face = np.array([vertex_idx, vertex_idx, vertex_idx], dtype=np.int32)
        return face, -1, np.array([1.0, 0.0, 0.0], dtype=np.float32)

    face_idx = int(matches[0])
    face = surface_faces[face_idx].astype(np.int32)
    weights = np.zeros(3, dtype=np.float32)
    local = np.nonzero(face == vertex_idx)[0]
    weights[int(local[0]) if len(local) > 0 else 0] = 1.0
    return face, face_idx, weights


def choose_fixed_pull_patches(vertices, surface_faces, anchor_indices, radius, min_size, pull_horizontal):
    surface_ids = np.unique(surface_faces.reshape(-1)).astype(np.int32)
    anchor_a, anchor_b = int(anchor_indices[0]), int(anchor_indices[1])

    point_a = vertices[anchor_a].astype(np.float32)
    point_b = vertices[anchor_b].astype(np.float32)
    group_a = surface_patch(surface_ids, vertices, point_a, radius, min_size)
    group_b = surface_patch(surface_ids, vertices, point_b, radius, min_size, exclude=group_a)

    direction = point_b - point_a
    if pull_horizontal:
        direction = direction.copy()
        direction[1] = 0.0
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-8:
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        direction = direction / direction_norm

    face_a, face_idx_a, weights_a = surface_face_for_vertex(surface_faces, anchor_a)
    face_b, face_idx_b, weights_b = surface_face_for_vertex(surface_faces, anchor_b)

    return {
        "point_a": point_a.astype(np.float32),
        "point_b": point_b.astype(np.float32),
        "face_a": face_a.astype(np.int32),
        "face_b": face_b.astype(np.int32),
        "face_idx_a": int(face_idx_a),
        "face_idx_b": int(face_idx_b),
        "barycentric_a": weights_a.astype(np.float32),
        "barycentric_b": weights_b.astype(np.float32),
        "group_a": group_a,
        "group_b": group_b,
        "direction": direction.astype(np.float32),
        "distance": float(np.linalg.norm(point_b - point_a)),
        "anchor_indices": [anchor_a, anchor_b],
    }


def compute_pull_directions(direction, pull_down_angle_degrees=None):
    if pull_down_angle_degrees is None:
        return -direction.astype(np.float32), direction.astype(np.float32)

    horizontal = direction.astype(np.float32).copy()
    horizontal[1] = 0.0
    norm = np.linalg.norm(horizontal)
    if norm < 1e-8:
        horizontal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        horizontal = horizontal / norm

    angle = np.deg2rad(float(pull_down_angle_degrees))
    down = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    horizontal_weight = np.cos(angle)
    down_weight = np.sin(angle)

    dir_a = -horizontal * horizontal_weight + down * down_weight
    dir_b = horizontal * horizontal_weight + down * down_weight
    return dir_a.astype(np.float32), dir_b.astype(np.float32)


def set_pin_group(particles, group, base_positions, offset):
    for idx, base in zip(group, base_positions):
        target = base + offset
        particles.pinned[int(idx)] = 1
        particles.pin_target[int(idx)] = ti.math.vec3(float(target[0]), float(target[1]), float(target[2]))


def clear_pin_group(particles, group):
    for idx in group:
        particles.pinned[int(idx)] = 0


def prepare(config, positions, surface_faces, bbox_diag):
    patch_radius = (
        config.patch_radius
        if config.patch_radius is not None
        else config.auto_patch_radius_ratio * bbox_diag
    )
    min_distance = config.random_anchor_min_distance_ratio * bbox_diag
    pull_distance = (
        config.pull_distance
        if config.pull_distance is not None
        else config.auto_pull_distance_ratio * bbox_diag
    )

    rng = np.random.default_rng(config.seed)
    if config.fixed_anchor_indices is None:
        patches = choose_pull_patches(
            positions,
            surface_faces,
            rng,
            patch_radius,
            min_distance,
            config.min_patch_size,
        )
        anchor_mode = "random_surface"
    else:
        patches = choose_fixed_pull_patches(
            positions,
            surface_faces,
            config.fixed_anchor_indices,
            patch_radius,
            config.min_patch_size,
            config.pull_horizontal,
        )
        anchor_mode = "fixed_particle_patch"

    group_a = patches["group_a"]
    group_b = patches["group_b"]
    direction = patches["direction"]
    pull_direction_a, pull_direction_b = compute_pull_directions(
        direction,
        config.pull_down_angle_degrees,
    )

    return {
        "kind": "pull",
        "scenario": config.scenario,
        "anchor_mode": anchor_mode,
        "patches": patches,
        "group_a": group_a,
        "group_b": group_b,
        "base_a": positions[group_a].astype(np.float32),
        "base_b": positions[group_b].astype(np.float32),
        "direction": direction,
        "pull_direction_a": pull_direction_a,
        "pull_direction_b": pull_direction_b,
        "patch_radius": float(patch_radius),
        "pull_distance": float(pull_distance),
    }


def apply_frame(config, scenario_state, particles, frame):
    alpha = frame / max(1, config.frames - 1)
    pull = scenario_state["pull_distance"] * alpha
    set_pin_group(
        particles,
        scenario_state["group_a"],
        scenario_state["base_a"],
        scenario_state["pull_direction_a"] * pull,
    )
    set_pin_group(
        particles,
        scenario_state["group_b"],
        scenario_state["base_b"],
        scenario_state["pull_direction_b"] * pull,
    )


def clear(config, scenario_state, particles):
    clear_pin_group(particles, scenario_state["group_a"])
    clear_pin_group(particles, scenario_state["group_b"])



def metadata(config, scenario_state):
    patches = scenario_state["patches"]
    group_a = scenario_state["group_a"]
    group_b = scenario_state["group_b"]
    base_a = scenario_state["base_a"]
    base_b = scenario_state["base_b"]
    return {
        "anchor_mode": scenario_state["anchor_mode"],
        "fixed_anchor_indices": list(config.fixed_anchor_indices) if config.fixed_anchor_indices is not None else None,
        "pull_horizontal": bool(config.pull_horizontal),
        "pull_down_angle_degrees": config.pull_down_angle_degrees,
        "pull_direction_a": scenario_state["pull_direction_a"].tolist(),
        "pull_direction_b": scenario_state["pull_direction_b"].tolist(),
        "patch_radius": float(scenario_state["patch_radius"]),
        "pull_distance": float(scenario_state["pull_distance"]),
        "anchor_a": {
            "rest_position": patches["point_a"].tolist(),
            "surface_face_index": int(patches["face_idx_a"]),
            "surface_face_vertices": patches["face_a"].astype(int).tolist(),
            "barycentric": patches["barycentric_a"].tolist(),
            "patch_vertex_indices": group_a.astype(int).tolist(),
            "patch_rest_positions": base_a.tolist(),
        },
        "anchor_b": {
            "rest_position": patches["point_b"].tolist(),
            "surface_face_index": int(patches["face_idx_b"]),
            "surface_face_vertices": patches["face_b"].astype(int).tolist(),
            "barycentric": patches["barycentric_b"].tolist(),
            "patch_vertex_indices": group_b.astype(int).tolist(),
            "patch_rest_positions": base_b.tolist(),
        },
        "point_a": patches["point_a"].tolist(),
        "point_b": patches["point_b"].tolist(),
        "group_a": group_a.astype(int).tolist(),
        "group_b": group_b.astype(int).tolist(),
        "direction": scenario_state["direction"].tolist(),
    }


def log_lines(config, scenario_state):
    patches = scenario_state["patches"]
    return [
        f"anchor mode: {scenario_state['anchor_mode']}",
        f"anchor points: A={patches['point_a']} B={patches['point_b']}",
        f"base horizontal direction: {scenario_state['direction']}",
        f"pull direction A: {scenario_state['pull_direction_a']}",
        f"pull direction B: {scenario_state['pull_direction_b']}",
        f"patch sizes: A={len(scenario_state['group_a'])} B={len(scenario_state['group_b'])}",
        f"pull distance: {scenario_state['pull_distance']:.4f}",
    ]
