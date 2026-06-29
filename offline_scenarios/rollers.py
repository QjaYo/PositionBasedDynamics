import numpy as np
import taichi as ti

import simulation_config as sim_config


SCENARIO_CONFIG = {
    "output": "rollers",
    "enable_tearing": False,
    "asset": sim_config.ASSET,
    "use_gravity": True,
    "seed": 7,
    "frames": 150,
    "fps": 30,
    "substeps": 10,
    "jacobi_relaxation": 0.8,
    "roller_center": (0.0, 0.0, 0.0),
    "roller_pair_axis": "x",
    "roller_center_height_ratio": 1.5,
    "bunny_start_height_ratio": 1.0,
    "roller_radius": 0.045,
    "roller_gap": 0.008,
    "roller_length": 0.24,
    "roller_angular_speed": 2.0 * np.pi / 1.5,
    "roller_upper_omega_sign": 1.0,
    "roller_lower_omega_sign": -1.0,
    "roller_contact_margin": 0.001,
    "mu_s": 2.0,
    "mu_k": 1.6,
    "roller_normal_restitution": 0.0,
    "roller_align_bunny_to_center": False,
    "roller_align_bunny_face_to_center": False,
    "bunny_start_offset": (0.015, 0.0, 0.0),
    "bunny_initial_velocity": (0.0, 0.0, 0.0),
    "roller_segments": 64,
    "roller_stripe_count": 12,
    "roller_stripe_twist": 3.0,
    "roller_yellow": (1.0, 0.82, 0.05, 1.0),
    "roller_black": (0.02, 0.02, 0.02, 1.0),
    "render_rollers": True,
    "camera": {
        "bounds_mode": "first_frame",
        "target_mode": "rollers",
        "distance_scale": 1.17,
        "height_scale": 0.0,
        "side_scale": 0.0,
        "lens": 55,
        "bounds_padding": 2.5,
    },
}


EPSILON = 1e-8


@ti.kernel
def set_initial_velocity(particles: ti.template(), vx: ti.f32, vy: ti.f32, vz: ti.f32):
    velocity = ti.math.vec3(vx, vy, vz)
    for i in particles.x:
        if particles.w[i] == 0.0:
            continue
        particles.v[i] = velocity
        particles.v_pre[i] = velocity


@ti.kernel
def project_roller_collision(
    particles: ti.template(),
    cx: ti.f32,
    cy: ti.f32,
    cz: ti.f32,
    radius: ti.f32,
    contact_margin: ti.f32,
    half_length: ti.f32,
    omega: ti.f32,
):
    for i in particles.x_pred:
        if particles.solver_w(i) == 0.0:
            continue

        p = particles.x_pred[i]
        z_rel = p.z - cz
        if ti.abs(z_rel) > half_length:
            continue

        dx = p.x - cx
        dy = p.y - cy
        radial = ti.sqrt(dx * dx + dy * dy)
        contact_radius = radius + contact_margin
        if radial >= contact_radius:
            continue

        nx = 1.0
        ny = 0.0
        if radial > EPSILON:
            nx = dx / radial
            ny = dy / radial

        n = ti.math.vec3(nx, ny, 0.0)
        r = ti.math.vec3(nx * radius, ny * radius, 0.0)
        surface_v = ti.math.vec3(-omega * r.y, omega * r.x, 0.0)
        corrected = ti.math.vec3(cx + nx * contact_radius, cy + ny * contact_radius, p.z)
        correction = corrected - p
        particles.x_pred[i] = corrected
        particles.dx_coll[i] += correction
        particles.coll_normal[i] += n
        particles.coll_surface_v[i] += surface_v
        particles.coll_count[i] += 1.0


def _as_vec3(value):
    return np.asarray(value, dtype=np.float32)


def roller_centers(config, center=None):
    center = _as_vec3(config.roller_center) if center is None else np.asarray(center, dtype=np.float32)
    offset = float(config.roller_radius + 0.5 * config.roller_gap)
    if config.roller_pair_axis == "x":
        upper = center - np.array([offset, 0.0, 0.0], dtype=np.float32)
        lower = center + np.array([offset, 0.0, 0.0], dtype=np.float32)
    else:
        upper = center + np.array([0.0, offset, 0.0], dtype=np.float32)
        lower = center - np.array([0.0, offset, 0.0], dtype=np.float32)
    return upper.astype(np.float32), lower.astype(np.float32)


def bunny_face_point(positions, surface_faces):
    surface_ids = np.unique(surface_faces.reshape(-1)).astype(np.int32)
    if len(surface_ids) == 0:
        idx = int(np.argmax(positions[:, 0]))
    else:
        surface_positions = positions[surface_ids]
        idx = int(surface_ids[int(np.argmax(surface_positions[:, 0]))])
    return idx, positions[idx].astype(np.float32)


def prepare(config, positions, surface_faces, bbox_diag):
    bbox_min = np.min(positions, axis=0)
    bbox_max = np.max(positions, axis=0)
    bunny_center = 0.5 * (bbox_min + bbox_max)
    bunny_height = float(max(bbox_max[1] - bbox_min[1], EPSILON))
    face_vertex, face_point = bunny_face_point(positions, surface_faces)

    center = _as_vec3(config.roller_center)
    center[1] = float(config.floor_y) + bunny_height * float(config.roller_center_height_ratio)

    start_offset = _as_vec3(config.bunny_start_offset)
    target_bunny_center = center + np.array(
        [0.0, bunny_height * float(config.bunny_start_height_ratio), 0.0],
        dtype=np.float32,
    )
    start_offset = start_offset + (target_bunny_center - bunny_center)
    if config.roller_align_bunny_to_center:
        start_offset = start_offset.copy()
        start_offset[1] += center[1] - bunny_center[1]
        start_offset[2] += center[2] - bunny_center[2]
    if config.roller_align_bunny_face_to_center:
        start_offset = start_offset.copy()
        start_offset[1] += center[1] - face_point[1]
        start_offset[2] += center[2] - face_point[2]

    positions += start_offset
    upper, lower = roller_centers(config, center)

    return {
        "kind": "rollers",
        "scenario": config.scenario,
        "initialized": False,
        "bunny_start_offset": start_offset.astype(np.float32),
        "bunny_height": float(bunny_height),
        "bunny_start_height_ratio": float(config.bunny_start_height_ratio),
        "bunny_initial_velocity": _as_vec3(config.bunny_initial_velocity),
        "bunny_face_vertex": int(face_vertex),
        "bunny_face_rest_position": face_point.astype(np.float32),
        "bunny_face_start_position": (face_point + start_offset).astype(np.float32),
        "roller_center": center,
        "roller_pair_axis": config.roller_pair_axis,
        "roller_center_height_ratio": float(config.roller_center_height_ratio),
        "upper_center": upper,
        "lower_center": lower,
        "upper_omega": float(config.roller_upper_omega_sign) * float(config.roller_angular_speed),
        "lower_omega": float(config.roller_lower_omega_sign) * float(config.roller_angular_speed),
        "roller_radius": float(config.roller_radius),
        "roller_gap": float(config.roller_gap),
        "roller_length": float(config.roller_length),
        "roller_contact_margin": float(config.roller_contact_margin),
        "roller_restitution": float(config.roller_normal_restitution),
    }


def apply_frame(config, scenario_state, particles, frame):
    if scenario_state["initialized"]:
        return
    v = scenario_state["bunny_initial_velocity"]
    set_initial_velocity(particles, float(v[0]), float(v[1]), float(v[2]))
    scenario_state["initialized"] = True


def _project_pair(scenario_state, particles):
    half_length = 0.5 * scenario_state["roller_length"]
    radius = scenario_state["roller_radius"]
    margin = scenario_state["roller_contact_margin"]
    upper = scenario_state["upper_center"]
    lower = scenario_state["lower_center"]
    project_roller_collision(particles, float(upper[0]), float(upper[1]), float(upper[2]), radius, margin, half_length, scenario_state["upper_omega"])
    project_roller_collision(particles, float(lower[0]), float(lower[1]), float(lower[2]), radius, margin, half_length, scenario_state["lower_omega"])


def apply_substep(config, scenario_state, particles, dt, stage):
    if stage in ("post_predict", "post_solve"):
        _project_pair(scenario_state, particles)


def clear(config, scenario_state, particles):
    return


def metadata(config, scenario_state):
    return {
        "anchor_mode": "rollers",
        "bunny_start_offset": scenario_state["bunny_start_offset"].tolist(),
        "bunny_height": float(scenario_state["bunny_height"]),
        "bunny_start_height_ratio": float(scenario_state["bunny_start_height_ratio"]),
        "bunny_initial_velocity": scenario_state["bunny_initial_velocity"].tolist(),
        "bunny_face_vertex": int(scenario_state["bunny_face_vertex"]),
        "bunny_face_rest_position": scenario_state["bunny_face_rest_position"].tolist(),
        "bunny_face_start_position": scenario_state["bunny_face_start_position"].tolist(),
        "rollers": {
            "center": scenario_state["roller_center"].tolist(),
            "pair_axis": scenario_state["roller_pair_axis"],
            "center_height_ratio": float(scenario_state["roller_center_height_ratio"]),
            "upper_center": scenario_state["upper_center"].tolist(),
            "lower_center": scenario_state["lower_center"].tolist(),
            "radius": float(scenario_state["roller_radius"]),
            "gap": float(scenario_state["roller_gap"]),
            "length": float(scenario_state["roller_length"]),
            "upper_omega": float(scenario_state["upper_omega"]),
            "lower_omega": float(scenario_state["lower_omega"]),
            "contact_margin": float(scenario_state["roller_contact_margin"]),
            "normal_restitution": float(scenario_state["roller_restitution"]),
        },
    }


def log_lines(config, scenario_state):
    return [
        "anchor mode: rollers",
        f"roller center: {scenario_state['roller_center']}",
        f"roller pair axis: {scenario_state['roller_pair_axis']}",
        f"bunny height/start ratio: {scenario_state['bunny_height']:.4f} / {scenario_state['bunny_start_height_ratio']:.4f}",
        f"roller radius/gap/length: {scenario_state['roller_radius']:.4f} / {scenario_state['roller_gap']:.4f} / {scenario_state['roller_length']:.4f}",
        f"roller omegas: upper={scenario_state['upper_omega']:.4f} lower={scenario_state['lower_omega']:.4f}",
        f"bunny start offset: {scenario_state['bunny_start_offset']}",
        f"bunny face vertex/start: {scenario_state['bunny_face_vertex']} / {scenario_state['bunny_face_start_position']}",
        f"bunny initial velocity: {scenario_state['bunny_initial_velocity']}",
    ]
