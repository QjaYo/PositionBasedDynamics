import taichi as ti
import numpy as np


@ti.data_oriented  # 이 클래스는 Taichi field를 멤버로 가지는 클래스임을 나타냄
class Particles:
    def __init__(self, positions, w, max_particles=None):
        """
        positions : (N, 3) numpy float32
        w         : (N,)   numpy float32, inv_mass (= 1/m), 0 = 고정 파티클
        """
        n = len(positions)
        self.num_particles = n       # 현재 active particle 수
        self.max_particles = 2 * n if max_particles is None else int(max_particles)
        if self.max_particles < n:
            raise ValueError("max_particles must be greater than or equal to len(positions)")

        self.x            = ti.Vector.field(3, dtype=ti.f32, shape=self.max_particles)  # 현재 위치
        self.v            = ti.Vector.field(3, dtype=ti.f32, shape=self.max_particles)  # 속도
        self.v_pre        = ti.Vector.field(3, dtype=ti.f32, shape=self.max_particles)  # 충돌 직전 속도 (restitution용)
        self.x_pred       = ti.Vector.field(3, dtype=ti.f32, shape=self.max_particles)  # 예측 위치
        self.dx_coll      = ti.Vector.field(3, dtype=ti.f32, shape=self.max_particles)  # 충돌로 인한 위치 보정량
        self.coll_normal  = ti.Vector.field(3, dtype=ti.f32, shape=self.max_particles)  # 충돌 법선 누적값
        self.coll_surface_v = ti.Vector.field(3, dtype=ti.f32, shape=self.max_particles)  # 충돌 표면 속도 누적값
        self.coll_count   = ti.field(dtype=ti.f32, shape=self.max_particles)  # 충돌 접촉 개수
        self.dx_constraint = ti.Vector.field(3, dtype=ti.f32, shape=self.max_particles)  # Jacobi 제약 보정 누적량
        self.constraint_count = ti.field(dtype=ti.f32, shape=self.max_particles)         # Jacobi 제약 보정 개수
        self.w            = ti.field(dtype=ti.f32, shape=self.max_particles)            # inv_mass (= 1/m), 0 = 고정/비활성
        self.pinned       = ti.field(dtype=ti.i32, shape=self.max_particles)            # 외부 고정점 여부
        self.pin_target   = ti.Vector.field(3, dtype=ti.f32, shape=self.max_particles)  # 외부 고정점 목표 위치

        x0 = np.zeros((self.max_particles, 3), dtype=np.float32)
        w0 = np.zeros(self.max_particles, dtype=np.float32)
        x0[:n] = positions.astype(np.float32)
        w0[:n] = w.astype(np.float32)

        self.x.from_numpy(x0)
        self.v.fill(0.0)
        self.v_pre.fill(0.0)
        self.x_pred.from_numpy(x0)
        self.dx_coll.fill(0.0)
        self.coll_normal.fill(0.0)
        self.coll_surface_v.fill(0.0)
        self.coll_count.fill(0.0)
        self.dx_constraint.fill(0.0)
        self.constraint_count.fill(0.0)
        self.w.from_numpy(w0)
        self.pinned.fill(0)
        self.pin_target.from_numpy(x0)

    @ti.func
    def solver_w(self, i):
        w = self.w[i]
        if self.pinned[i] != 0:
            w = 0.0
        return w
