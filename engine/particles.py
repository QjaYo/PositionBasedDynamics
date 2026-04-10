import taichi as ti


@ti.data_oriented  # 이 클래스는 Taichi field를 멤버로 가지는 클래스임을 나타냄
class Particles:
    def __init__(self, positions, w):
        """
        positions : (N, 3) numpy float32
        w         : (N,)   numpy float32, inv_mass (= 1/m), 0 = 고정 파티클
        """
        n = len(positions)
        self.num_particles = n

        self.x            = ti.Vector.field(3, dtype=ti.f32, shape=n)  # 현재 위치
        self.v            = ti.Vector.field(3, dtype=ti.f32, shape=n)  # 속도
        self.x_pred       = ti.Vector.field(3, dtype=ti.f32, shape=n)  # 예측 위치
        self.w            = ti.field(dtype=ti.f32, shape=n)            # inv_mass (= 1/m), 0 = 고정
        self.n_dist_constraints = ti.field(dtype=ti.f32, shape=n)       # 파티클당 참여 거리 제약 수
        self.n_vol_constraints  = ti.field(dtype=ti.f32, shape=n)       # 파티클당 참여 부피 제약 수

        self.x.from_numpy(positions.astype('float32'))
        self.v.fill(0.0)
        self.x_pred.fill(0.0)
        self.w.from_numpy(w.astype('float32'))
        self.n_dist_constraints.fill(0.0)
        self.n_vol_constraints.fill(0.0)
