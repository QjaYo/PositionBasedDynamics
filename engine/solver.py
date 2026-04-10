import taichi as ti

from engine.constraint import distance_constraint, volume_constraint, floor_constraint

EPSILON      = 1e-8   # 거리 제약 denom 체크 (그래디언트가 단위벡터 스케일)
EPSILON_VOL  = 1e-20  # 부피 제약 denom 체크 (그래디언트가 l^2 스케일 → denom은 l^4 ≈ 1e-9)


@ti.kernel
def project_distance(
    particles: ti.template(),
    edges: ti.template(),
    rest_lengths: ti.template(),
    stiffness: ti.f32
):
    for i in edges:
        idx0, idx1 = edges[i][0], edges[i][1]
        w0 = particles.w[idx0]
        w1 = particles.w[idx1]
        w_sum = w0 + w1
        if w_sum == 0.0:
            continue

        p0 = particles.x_pred[idx0]
        p1 = particles.x_pred[idx1]

        diff = p1 - p0
        length = diff.norm()
        if length < EPSILON:
            continue

        C, grad0, grad1 = distance_constraint(p0, p1, rest_lengths[i])

        denom = w0 * grad0.norm_sqr() + w1 * grad1.norm_sqr()
        if denom < EPSILON:
            continue

        s = C / denom
        particles.x_pred[idx0] -= stiffness * s * w0 * grad0 / particles.n_dist_constraints[idx0]
        particles.x_pred[idx1] -= stiffness * s * w1 * grad1 / particles.n_dist_constraints[idx1]


@ti.kernel
def project_volume(
    particles: ti.template(),
    tet_elems: ti.template(),
    rest_volumes: ti.template(),
    stiffness: ti.f32
):
    for i in tet_elems:
        idx0, idx1, idx2, idx3 = tet_elems[i][0], tet_elems[i][1], tet_elems[i][2], tet_elems[i][3]
        w0 = particles.w[idx0]
        w1 = particles.w[idx1]
        w2 = particles.w[idx2]
        w3 = particles.w[idx3]

        p0 = particles.x_pred[idx0]
        p1 = particles.x_pred[idx1]
        p2 = particles.x_pred[idx2]
        p3 = particles.x_pred[idx3]
        C, grad0, grad1, grad2, grad3 = volume_constraint(p0, p1, p2, p3, rest_volumes[i])

        denom = (w0 * grad0.norm_sqr() + w1 * grad1.norm_sqr() +
                 w2 * grad2.norm_sqr() + w3 * grad3.norm_sqr())
        if denom < EPSILON_VOL:
            continue
        s = C / denom
        particles.x_pred[idx0] -= stiffness * s * w0 * grad0 / particles.n_vol_constraints[idx0]
        particles.x_pred[idx1] -= stiffness * s * w1 * grad1 / particles.n_vol_constraints[idx1]
        particles.x_pred[idx2] -= stiffness * s * w2 * grad2 / particles.n_vol_constraints[idx2]
        particles.x_pred[idx3] -= stiffness * s * w3 * grad3 / particles.n_vol_constraints[idx3]


@ti.kernel
def project_floor(particles: ti.template()):
    for i in particles.x_pred:
        C, grad = floor_constraint(particles.x_pred[i])
        if C >= 0.0:
            continue
        w = particles.w[i]
        if w == 0.0:
            continue
        s = C / (w * grad.norm_sqr())
        particles.x_pred[i] -= s * w * grad


class ConstraintSolver:
    def __init__(self):
        self.internal_constraints = []
        self.collision_constraints = []

    def register(self, fn, *args):
        self.internal_constraints.append((fn, args))

    def generate_collision_constraints(self, particles):
        self.collision_constraints = [(project_floor, (particles,))]

    def solve(self, n_iter):
        for _ in range(n_iter):
            for fn, args in self.internal_constraints:
                fn(*args)
            for fn, args in self.collision_constraints:
                fn(*args)
