import taichi as ti
import numpy as np

from scene import create_scene, FLOOR_Y
from renderer import Renderer
from engine.particles import Particles
from engine.integrator import apply_external_forces, predict_positions, update
from engine.friction import apply_floor_friction
from engine.solver import project_distance, project_volume, ConstraintSolver

ti.init(arch=ti.metal)

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
n_iter             = 10
mu_s               = 1.0   # 정지마찰계수
mu_k               = 1.0   # 운동마찰계수
k_distance         = 0.99  # 원래 stiffness (0~1 범위)
k_volume           = 1.0  # 원래 stiffness
# 논문 3.3절: k' = 1 - (1-k)^(1/n_iter) → n_iter에 독립적인 강성
stiffness_distance = float(1.0 - (1.0 - k_distance) ** (1.0 / n_iter))
stiffness_volume   = float(1.0 - (1.0 - k_volume)   ** (1.0 / n_iter))

solver = ConstraintSolver()
solver.register(project_distance, particles, edges_field, rest_lengths_field, stiffness_distance)
solver.register(project_volume, particles, tet_elems_field, rest_volumes_field, stiffness_volume)

renderer = Renderer(particles, surface_faces)

# ── (4) 메인 루프 ─────────────────────────────────────────────────────────────
while renderer.is_running():
    for _ in range(n_substeps):
        # (5) 외력 적용
        apply_external_forces(particles, dt)

        # (6) 댐핑
        # damp_velocities(particles, k_damping)

        # (7) 예측 위치
        predict_positions(particles, dt)

        # (8) 충돌 제약 생성
        solver.generate_collision_constraints(particles)

        # (9-11) 제약 반복
        solver.solve(n_iter)

        # (12-14) 속도 & 위치 업데이트
        update(particles, dt)

        # (16) 바닥 마찰
        apply_floor_friction(particles, FLOOR_Y, mu_s, mu_k)

    renderer.render()
