import math
import taichi as ti
import numpy as np

from scene import create_scene, FLOOR_Y, FLOOR_MOVE, FLOOR_AMPLITUDE, FLOOR_FREQ
from renderer import Renderer
from engine.particles import Particles
from engine.integrator import apply_external_forces, predict_positions, update
from engine.friction import apply_floor_friction
from engine.solver import distance_projection, volume_projection, ConstraintSolver
from engine.damping import damp_velocities
from engine.picker import pick, mouse_ray

ti.init(arch=ti.gpu)

# ── (1-3) 초기화 ──────────────────────────────────────────────────────────────
positions, edges, rest_lengths, tet_elems, rest_volumes, surface_faces = \
    create_scene("assets/bunny.obj")

n = len(positions)
w = np.ones(n, dtype=np.float32)          # inv_mass, 0 = 고정 파티클
particles = Particles(positions, w)

edges_field        = ti.Vector.field(2, dtype=ti.i32, shape=len(edges))
rest_lengths_field = ti.field(dtype=ti.f32, shape=len(rest_lengths))
tet_elems_field    = ti.Vector.field(4, dtype=ti.i32, shape=len(tet_elems))
rest_volumes_field = ti.field(dtype=ti.f32, shape=len(rest_volumes))

edges_field.from_numpy(edges)
rest_lengths_field.from_numpy(rest_lengths)
tet_elems_field.from_numpy(tet_elems)
rest_volumes_field.from_numpy(rest_volumes)

# 파티클별 제약 참여 횟수 계산 (거리/부피 분리)
n_dist = np.zeros(n, dtype=np.float32)
for e in edges:
    n_dist[e[0]] += 1
    n_dist[e[1]] += 1

n_vol = np.zeros(n, dtype=np.float32)
for t in tet_elems:
    n_vol[t[0]] += 1
    n_vol[t[1]] += 1
    n_vol[t[2]] += 1
    n_vol[t[3]] += 1

n_dist = np.maximum(n_dist, 1.0)
n_vol  = np.maximum(n_vol,  1.0)
particles.n_dist_constraints.from_numpy(n_dist)
particles.n_vol_constraints.from_numpy(n_vol)
print(f"[main] n_dist: mean={n_dist.mean():.1f}  n_vol: mean={n_vol.mean():.1f}")

fps                = 60
n_substeps         = 10
dt                 = 1.0 / (fps * n_substeps)
n_iter             = 15
mu_s               = 1.0   # 정지마찰계수
mu_k               = 1.0   # 운동마찰계수
k_distance         = 1.0   # 원래 stiffness (0~1 범위)
k_volume           = 0.8   # 원래 stiffness
k_damping          = 0.01  # 댐핑 계수 (0~1 범위)
# 논문 3.3절: k' = 1 - (1-k)^(1/n_iter) → n_iter에 독립적인 강성
stiffness_distance = float(1.0 - (1.0 - k_distance) ** (1.0 / n_iter))
stiffness_volume   = float(1.0 - (1.0 - k_volume)   ** (1.0 / n_iter))

solver = ConstraintSolver()
solver.register(distance_projection, particles, edges_field, rest_lengths_field, stiffness_distance)
solver.register(volume_projection, particles, tet_elems_field, rest_volumes_field, stiffness_volume)

renderer = Renderer(particles, surface_faces)

# ── (4) 메인 루프 ─────────────────────────────────────────────────────────────
sim_t      = 0.0
floor_move = FLOOR_MOVE       # 런타임 토글 (F 키)
floor_base = FLOOR_Y          # 진동 기준 y (바닥 드래그로 갱신됨)
floor_y    = FLOOR_Y          # 현재 바닥 y

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
        x_np = particles.x.to_numpy()
        kind, payload = pick(x_np, cursor, aspect, cam_pos, cam_lookat, cam_up, floor_y)
        if kind == 'bunny':
            idx, depth = payload
            drag = {'kind': 'bunny', 'idx': idx, 'depth': depth,
                    'orig_w': float(particles.w[idx])}
            particles.w[idx] = 0.0
        elif kind == 'floor':
            drag = {'kind': 'floor', 'cursor_y0': cursor[1], 'floor_y0': floor_y}
        else:
            drag = None
    elif lmb and drag is not None:
        # 드래그 중
        if drag['kind'] == 'bunny':
            origin, direction = mouse_ray(cursor, aspect, cam_pos, cam_lookat, cam_up)
            target = origin + direction * drag['depth']
            idx = drag['idx']
            particles.x[idx]      = ti.math.vec3(float(target[0]), float(target[1]), float(target[2]))
            particles.x_pred[idx] = ti.math.vec3(float(target[0]), float(target[1]), float(target[2]))
            particles.v[idx]      = ti.math.vec3(0.0, 0.0, 0.0)
        elif drag['kind'] == 'floor':
            dy = cursor[1] - drag['cursor_y0']
            floor_base = drag['floor_y0'] + dy * FLOOR_DRAG_SENSITIVITY
    elif not lmb and prev_lmb and drag is not None:
        # 릴리즈
        if drag['kind'] == 'bunny':
            idx = drag['idx']
            particles.w[idx] = drag['orig_w']
            particles.v[idx] = ti.math.vec3(0.0, 0.0, 0.0)
        drag = None
    prev_lmb = lmb

    sel_label = ("None" if drag is None
                 else "Bunny" if drag['kind'] == 'bunny'
                 else "Floor")

    for _ in range(n_substeps):
        # 초기화
        particles.dx_coll.fill(0.0)

        # 바닥 위치: 드래그 중엔 사용자 입력 직접 사용, 아니면 진동/정지
        if drag is not None and drag['kind'] == 'floor':
            floor_y = floor_base
        elif floor_move:
            floor_y = floor_base + FLOOR_AMPLITUDE * math.sin(2.0 * math.pi * FLOOR_FREQ * sim_t)
        else:
            floor_y = floor_base

        # (5) 외력 적용
        apply_external_forces(particles, dt)

        # (6) 댐핑
        damp_velocities(particles, k_damping)

        # (7) 예측 위치
        predict_positions(particles, dt)

        # (8) 충돌 제약 생성
        solver.generate_collision_constraints(particles, floor_y)

        # (9-11) 제약 반복
        solver.solve(n_iter)

        # (12-14) 속도 & 위치 업데이트
        update(particles, dt)

        # (16) 바닥 마찰
        apply_floor_friction(particles, mu_s, mu_k, dt)

        sim_t += dt

    renderer.set_floor_y(floor_y)
    renderer.render(sel_label, floor_move)
