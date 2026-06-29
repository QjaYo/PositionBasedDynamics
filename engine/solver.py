import taichi as ti

from engine.constraint import distance_constraint, volume_constraint, collision_constraint

EPSILON_DIST = 1e-8
EPSILON_VOL = 1e-20
EPSILON_COLL = 1e-8
EPSILON_BEND = 1e-8
PI = 3.141592653589793


@ti.func
def _accumulate_constraint_correction(particles: ti.template(), idx: ti.i32, corr: ti.math.vec3):
    ti.atomic_add(particles.dx_constraint[idx][0], corr[0])
    ti.atomic_add(particles.dx_constraint[idx][1], corr[1])
    ti.atomic_add(particles.dx_constraint[idx][2], corr[2])
    ti.atomic_add(particles.constraint_count[idx], 1.0)


@ti.kernel
def distance_projection_jacobi(
    particles: ti.template(),
    edges: ti.template(),
    rest_lengths: ti.template(),
    num_edges: ti.i32,
    stiffness: ti.f32,
):
    for i in range(num_edges):
        idx0, idx1 = edges[i][0], edges[i][1]
        w0 = particles.solver_w(idx0)
        w1 = particles.solver_w(idx1)
        w_sum = w0 + w1
        if w_sum == 0.0:
            continue

        p0 = particles.x_pred[idx0]
        p1 = particles.x_pred[idx1]

        diff = p1 - p0
        length = diff.norm()
        if length < EPSILON_DIST:
            continue

        C, grad0, grad1 = distance_constraint(p0, p1, rest_lengths[i])
        denom = w0 * grad0.norm_sqr() + w1 * grad1.norm_sqr()
        if denom < EPSILON_DIST:
            continue

        s = C / denom
        if w0 != 0.0:
            _accumulate_constraint_correction(particles, idx0, -stiffness * s * w0 * grad0)
        if w1 != 0.0:
            _accumulate_constraint_correction(particles, idx1, -stiffness * s * w1 * grad1)


@ti.kernel
def volume_projection_jacobi(
    particles: ti.template(),
    tet_elems: ti.template(),
    rest_volumes: ti.template(),
    stiffness: ti.f32,
):
    for i in tet_elems:
        idx0, idx1, idx2, idx3 = tet_elems[i][0], tet_elems[i][1], tet_elems[i][2], tet_elems[i][3]
        w0 = particles.solver_w(idx0)
        w1 = particles.solver_w(idx1)
        w2 = particles.solver_w(idx2)
        w3 = particles.solver_w(idx3)

        p0 = particles.x_pred[idx0]
        p1 = particles.x_pred[idx1]
        p2 = particles.x_pred[idx2]
        p3 = particles.x_pred[idx3]
        C, grad0, grad1, grad2, grad3 = volume_constraint(p0, p1, p2, p3, rest_volumes[i])

        denom = (
            w0 * grad0.norm_sqr()
            + w1 * grad1.norm_sqr()
            + w2 * grad2.norm_sqr()
            + w3 * grad3.norm_sqr()
        )
        if denom < EPSILON_VOL:
            continue

        s = C / denom
        if w0 != 0.0:
            _accumulate_constraint_correction(particles, idx0, -stiffness * s * w0 * grad0)
        if w1 != 0.0:
            _accumulate_constraint_correction(particles, idx1, -stiffness * s * w1 * grad1)
        if w2 != 0.0:
            _accumulate_constraint_correction(particles, idx2, -stiffness * s * w2 * grad2)
        if w3 != 0.0:
            _accumulate_constraint_correction(particles, idx3, -stiffness * s * w3 * grad3)


@ti.kernel
def bending_projection_jacobi(
    particles: ti.template(),
    bend_quads: ti.template(),
    rest_angles: ti.template(),
    num_bends: ti.i32,
    stiffness: ti.f32,
):
    for i in range(num_bends):
        idx1 = bend_quads[i][0]
        idx2 = bend_quads[i][1]
        idx3 = bend_quads[i][2]
        idx4 = bend_quads[i][3]

        w1 = particles.solver_w(idx1)
        w2 = particles.solver_w(idx2)
        w3 = particles.solver_w(idx3)
        w4 = particles.solver_w(idx4)
        if w1 + w2 + w3 + w4 == 0.0:
            continue

        p1 = particles.x_pred[idx1]
        p2 = particles.x_pred[idx2]
        p3 = particles.x_pred[idx3]
        p4 = particles.x_pred[idx4]

        v2 = p2 - p1
        v3 = p3 - p1
        v4 = p4 - p1
        c23 = v2.cross(v3)
        c24 = v2.cross(v4)
        len23 = c23.norm()
        len24 = c24.norm()
        if len23 < EPSILON_BEND or len24 < EPSILON_BEND:
            continue

        n1 = c23 / len23
        n2 = c24 / len24
        d = n1.dot(n2)
        d = ti.max(-0.9999, ti.min(0.9999, d))

        # flat cloth의 rest bending angle은 pi(180도)이다.
        C = ti.acos(d) - rest_angles[i]
        denom_angle = ti.sqrt(1.0 - d * d)
        if denom_angle < EPSILON_BEND:
            continue

        q3 = (v2.cross(n2) + n1.cross(v2) * d) / len23
        q4 = (v2.cross(n1) + n2.cross(v2) * d) / len24
        q2 = -((v3.cross(n2) + n1.cross(v3) * d) / len23) - ((v4.cross(n1) + n2.cross(v4) * d) / len24)
        q1 = -q2 - q3 - q4

        denom = w1 * q1.norm_sqr() + w2 * q2.norm_sqr() + w3 * q3.norm_sqr() + w4 * q4.norm_sqr()
        if denom < EPSILON_BEND:
            continue

        scale = -stiffness * denom_angle * C / denom
        if w1 != 0.0:
            _accumulate_constraint_correction(particles, idx1, scale * w1 * q1)
        if w2 != 0.0:
            _accumulate_constraint_correction(particles, idx2, scale * w2 * q2)
        if w3 != 0.0:
            _accumulate_constraint_correction(particles, idx3, scale * w3 * q3)
        if w4 != 0.0:
            _accumulate_constraint_correction(particles, idx4, scale * w4 * q4)


@ti.kernel
def clear_jacobi_corrections(particles: ti.template()):
    for i in particles.x_pred:
        particles.dx_constraint[i] = ti.math.vec3(0.0, 0.0, 0.0)
        particles.constraint_count[i] = 0.0


@ti.kernel
def apply_jacobi_corrections(particles: ti.template(), relaxation: ti.f32):
    for i in particles.x_pred:
        count = particles.constraint_count[i]
        if count <= 0.0:
            continue
        if particles.solver_w(i) == 0.0:
            continue
        particles.x_pred[i] += relaxation * particles.dx_constraint[i]


@ti.kernel
def static_collision_projection_jacobi(
    particles: ti.template(),
    particle_indices: ti.template(),
    surface_points: ti.template(),
    surface_normals: ti.template(),
    count: ti.template(),
    stiffness: ti.f32,
):
    for i in range(count[None]):
        idx = particle_indices[i]
        w = particles.solver_w(idx)
        if w == 0.0:
            continue

        n = surface_normals[i]
        n_norm = n.norm()
        if n_norm < EPSILON_COLL:
            continue
        n = n / n_norm

        C = (particles.x_pred[idx] - surface_points[i]).dot(n)
        if C >= 0.0:
            continue

        corr = -stiffness * C * n
        _accumulate_constraint_correction(particles, idx, corr)
        particles.dx_coll[idx] += corr
        particles.coll_normal[idx] += n
        particles.coll_count[idx] += 1.0


@ti.kernel
def collision_projection(particles: ti.template(), floor_y: ti.f32, floor_v_y: ti.f32):
    for i in particles.x_pred:
        C, grad = collision_constraint(particles.x_pred[i], floor_y)
        if C >= 0.0:
            continue
        w = particles.solver_w(i)
        if w == 0.0:
            continue

        denom = w * grad.norm_sqr()
        if denom < EPSILON_COLL:
            continue
        s = C / denom

        correction = -s * w * grad
        particles.x_pred[i] += correction
        particles.dx_coll[i] += correction
        particles.coll_normal[i] += grad
        particles.coll_surface_v[i] += ti.math.vec3(0.0, floor_v_y, 0.0)
        particles.coll_count[i] += 1.0


class JacobiConstraintSolver:
    def __init__(self, relaxation=1.0):
        self.relaxation = float(relaxation)
        self.jacobi_constraints = []
        self.collision_constraints = []
        self.rigid_systems = []

    def register_jacobi(self, fn, *args):
        self.jacobi_constraints.append((fn, args))

    def register_rigid_system(self, rigid_system):
        self.rigid_systems.append(rigid_system)

    def generate_collision_constraints(self, particles, floor_y: float, floor_v_y: float = 0.0):
        self.collision_constraints = [(collision_projection, (particles, floor_y, float(floor_v_y)))]

    def solve(self, n_iter):
        if not self.jacobi_constraints:
            for _ in range(n_iter):
                for fn, args in self.collision_constraints:
                    fn(*args)
            return

        particles = self.jacobi_constraints[0][1][0]
        for _ in range(n_iter):
            clear_jacobi_corrections(particles)
            for rigid_system in self.rigid_systems:
                rigid_system.clear_corrections()
            for fn, args in self.jacobi_constraints:
                fn(*args)
            apply_jacobi_corrections(particles, self.relaxation)
            for rigid_system in self.rigid_systems:
                rigid_system.apply_corrections(self.relaxation)
            for fn, args in self.collision_constraints:
                fn(*args)
