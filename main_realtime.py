import math
import taichi as ti
import numpy as np

from scene import create_scene
import simulation_config as sim_config
from renderer import Renderer
from engine.particles import Particles
from engine.integrator import apply_external_forces, predict_positions, velocityUpdate
from engine.friction import apply_floor_friction
from engine.restitution import apply_floor_restitution
from engine.solver import (
    JacobiConstraintSolver,
    distance_projection_jacobi,
    volume_projection_jacobi,
)
from engine.damping import damp_velocities
from engine.picker import pick, mouse_ray
import engine.tearing as tearing


# ── Realtime runtime controls ────────────────────────────────────────────────
FPS = 60
SUBSTEPS = 20
SOLVER_ITERS = 100
JACOBI_RELAXATION = 0.6
GRAB_PATCH_RADIUS = 0.0125
GRAB_MIN_PATCH_SIZE = 8
ENABLE_TEARING = False

ti.init(arch=ti.gpu)

if ENABLE_TEARING:
    tearing.tear_ratio = float(sim_config.TEAR_RATIO)
    tearing.max_tears_per_call = int(sim_config.MAX_TEARS_PER_CALL)


def surface_patch(surface_faces, vertices, center, radius, min_size):
    surface_ids = np.unique(surface_faces.reshape(-1)).astype(np.int32)
    if len(surface_ids) == 0:
        return np.zeros(0, dtype=np.int32)

    distances = np.linalg.norm(vertices[surface_ids] - center, axis=1)
    group = surface_ids[distances <= radius]
    if len(group) >= min_size:
        return group.astype(np.int32)

    nearest = np.argsort(distances)[:min(min_size, len(surface_ids))]
    return surface_ids[nearest].astype(np.int32)


def set_pin_group(particles, group, targets):
    for idx, target in zip(group, targets):
        i = int(idx)
        p = ti.math.vec3(float(target[0]), float(target[1]), float(target[2]))
        particles.pinned[i] = 1
        particles.pin_target[i] = p
        particles.x[i] = p
        particles.x_pred[i] = p
        particles.v[i] = ti.math.vec3(0.0, 0.0, 0.0)


def clear_pin_group(particles, group):
    for idx in group:
        i = int(idx)
        particles.pinned[i] = 0
        particles.v[i] = ti.math.vec3(0.0, 0.0, 0.0)

# ── (1-3) 초기화 ──────────────────────────────────────────────────────────────
(positions, edges, rest_lengths, tet_elems, rest_volumes, surface_faces) = create_scene(sim_config.ASSET)

n = len(positions)
w = np.ones(n, dtype=np.float32)          # inv_mass, 0 = 고정 파티클
particles = Particles(positions, w)

num_edges = len(edges)
max_edges = 2 * num_edges
edges_padded = np.zeros((max_edges, 2), dtype=np.int32)
rest_lengths_padded = np.zeros(max_edges, dtype=np.float32)
edges_padded[:num_edges] = edges
rest_lengths_padded[:num_edges] = rest_lengths
edges = edges_padded
rest_lengths = rest_lengths_padded

edges_field        = ti.Vector.field(2, dtype=ti.i32, shape=max_edges)
rest_lengths_field = ti.field(dtype=ti.f32, shape=max_edges)
tet_elems_field    = ti.Vector.field(4, dtype=ti.i32, shape=len(tet_elems))
rest_volumes_field = ti.field(dtype=ti.f32, shape=len(rest_volumes))

edges_field.from_numpy(edges)
rest_lengths_field.from_numpy(rest_lengths)
tet_elems_field.from_numpy(tet_elems)
rest_volumes_field.from_numpy(rest_volumes)

print(f"[main] solver: jacobi relaxation={JACOBI_RELAXATION}")

fps                = FPS
n_substeps         = SUBSTEPS
dt                 = 1.0 / (fps * n_substeps)
n_iter             = SOLVER_ITERS
mu_s               = sim_config.MU_S
mu_k               = sim_config.MU_K
e_restitution      = sim_config.E_RESTITUTION
k_distance         = sim_config.K_DISTANCE
k_volume           = sim_config.K_VOLUME
k_damping          = sim_config.DAMPING
# 논문 3.3절: k' = 1 - (1-k)^(1/n_iter) → n_iter에 독립적인 강성
stiffness_distance = float(1.0 - (1.0 - k_distance) ** (1.0 / n_iter))
stiffness_volume   = float(1.0 - (1.0 - k_volume)   ** (1.0 / n_iter))

solver = JacobiConstraintSolver(relaxation=JACOBI_RELAXATION)
solver.register_jacobi(
    distance_projection_jacobi,
    particles,
    edges_field,
    rest_lengths_field,
    num_edges,
    stiffness_distance,
)
solver.register_jacobi(
    volume_projection_jacobi,
    particles,
    tet_elems_field,
    rest_volumes_field,
    stiffness_volume,
)

renderer = Renderer(particles, surface_faces)

# ── (4) 메인 루프 ─────────────────────────────────────────────────────────────
sim_t      = 0.0
floor_move = sim_config.FLOOR_MOVE       # 런타임 토글 (F 키)
floor_base = sim_config.FLOOR_Y          # 진동 기준 y (바닥 드래그로 갱신됨)
floor_y    = sim_config.FLOOR_Y          # 현재 바닥 y

prev_lmb   = False
prev_f     = False
drag       = None             # None | {'kind': 'bunny'|'floor', ...}

FLOOR_DRAG_SENSITIVITY = 0.5  # 화면 정규화 1.0 = world 0.5m

while renderer.is_running():
    win    = renderer.window
    cursor = win.get_cursor_pos()
    cam_pos, cam_lookat, cam_up = renderer.get_camera_state()
    aspect = renderer.aspect

    # F 키 → floor 진동 토글 (엣지 트리거)
    f_now = win.is_pressed('f')
    if f_now and not prev_f:
        floor_move = not floor_move
    prev_f = f_now

    # LMB 입력 처리
    lmb = win.is_pressed(ti.ui.LMB)
    if lmb and not prev_lmb:
        # 클릭 시점: pick
        x_np = particles.x.to_numpy()[:particles.num_particles]
        kind, payload = pick(x_np, cursor, aspect, cam_pos, cam_lookat, cam_up, floor_y)
        if kind == 'bunny':
            idx, depth = payload
            center = x_np[int(idx)]
            group = surface_patch(
                surface_faces,
                x_np,
                center,
                GRAB_PATCH_RADIUS,
                GRAB_MIN_PATCH_SIZE,
            )
            if len(group) == 0:
                group = np.array([idx], dtype=np.int32)
            base_positions = x_np[group].astype(np.float32)
            drag = {
                'kind': 'bunny',
                'idx': int(idx),
                'depth': depth,
                'group': group,
                'base_positions': base_positions,
                'base_pick_position': center.astype(np.float32),
            }
            set_pin_group(particles, group, base_positions)
            renderer.set_marked_vertices(group)
        elif kind == 'floor':
            drag = {'kind': 'floor', 'cursor_y0': cursor[1], 'floor_y0': floor_y}
            renderer.set_marked_vertices([])
        else:
            drag = None
            renderer.set_marked_vertices([])
    elif lmb and drag is not None:
        # 드래그 중
        if drag['kind'] == 'bunny':
            origin, direction = mouse_ray(cursor, aspect, cam_pos, cam_lookat, cam_up)
            target = origin + direction * drag['depth']
            offset = target.astype(np.float32) - drag['base_pick_position']
            targets = drag['base_positions'] + offset
            set_pin_group(particles, drag['group'], targets)
        elif drag['kind'] == 'floor':
            dy = cursor[1] - drag['cursor_y0']
            floor_base = drag['floor_y0'] + dy * FLOOR_DRAG_SENSITIVITY
    elif not lmb and prev_lmb and drag is not None:
        # 릴리즈
        if drag['kind'] == 'bunny':
            clear_pin_group(particles, drag['group'])
            renderer.set_marked_vertices([])
        drag = None
    prev_lmb = lmb

    sel_label = ("None" if drag is None
                 else f"Bunny patch ({len(drag['group'])})" if drag['kind'] == 'bunny'
                 else "Floor")

    for _ in range(n_substeps):
        # 초기화
        particles.dx_coll.fill(0.0)
        particles.coll_normal.fill(0.0)
        particles.coll_surface_v.fill(0.0)
        particles.coll_count.fill(0.0)

        # 바닥 위치: 드래그 중엔 사용자 입력 직접 사용, 아니면 진동/정지
        floor_v_y = 0.0
        if drag is not None and drag['kind'] == 'floor':
            floor_y = floor_base
        elif floor_move:
            omega = 2.0 * math.pi * sim_config.FLOOR_FREQ
            floor_y = floor_base + sim_config.FLOOR_AMPLITUDE * math.sin(omega * sim_t)
            floor_v_y = sim_config.FLOOR_AMPLITUDE * omega * math.cos(omega * sim_t)
        else:
            floor_y = floor_base

        # (5) 외력 적용
        if sim_config.USE_GRAVITY:
            apply_external_forces(particles, dt)

        # (6) 댐핑
        damp_velocities(particles, k_damping)

        # (7) 예측 위치
        predict_positions(particles, dt)
        
        if ENABLE_TEARING:
            old_num_edges = num_edges
            old_num_particles = particles.num_particles
            num_edges, surface_faces, tearing_changed = tearing.apply_tearing(
                particles,
                edges,
                rest_lengths,
                num_edges,
                max_edges,
                tet_elems,
                rest_volumes,
                surface_faces,
            )
            if tearing_changed or num_edges != old_num_edges or particles.num_particles != old_num_particles:
                edges_field.from_numpy(edges)
                rest_lengths_field.from_numpy(rest_lengths)
                tet_elems_field.from_numpy(tet_elems)
                renderer.set_surface_faces(surface_faces)
                if drag is not None and drag['kind'] == 'bunny':
                    renderer.set_marked_vertices(drag['group'])

        # (8) 충돌 제약 생성
        solver.generate_collision_constraints(particles, floor_y, floor_v_y)

        # (9-11) 제약 반복
        solver.solve(n_iter)

        # velocityUpdate 직전 v 백업 (restitution이 v_pre로 사용)
        particles.v_pre.copy_from(particles.v)

        # (12-14) 속도 & 위치 업데이트
        velocityUpdate(particles, dt)

        # (15) 바닥 반발
        apply_floor_restitution(particles, e_restitution, floor_v_y)

        # (16) 바닥 마찰
        apply_floor_friction(particles, mu_s, mu_k, dt)

        sim_t += dt

    renderer.set_floor_y(floor_y)
    renderer.render(sel_label, floor_move)
