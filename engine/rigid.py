import numpy as np
import taichi as ti


EPSILON = 1e-8
GRAVITY = ti.Vector([0.0, -9.8, 0.0])


@ti.func
def _skew(v):
    return ti.Matrix(
        [
            [0.0, -v.z, v.y],
            [v.z, 0.0, -v.x],
            [-v.y, v.x, 0.0],
        ]
    )


@ti.func
def _rotation_from_vector(theta):
    angle = theta.norm()
    result = ti.Matrix.identity(ti.f32, 3)
    if angle < 1e-6:
        result = result + _skew(theta)
    else:
        axis = theta / angle
        k = _skew(axis)
        result = result + ti.sin(angle) * k + (1.0 - ti.cos(angle)) * (k @ k)
    return result


@ti.func
def _orthonormalize(r):
    c0 = ti.Vector([r[0, 0], r[1, 0], r[2, 0]])
    c1 = ti.Vector([r[0, 1], r[1, 1], r[2, 1]])
    c0_norm = c0.norm()
    if c0_norm < EPSILON:
        c0 = ti.Vector([1.0, 0.0, 0.0])
    else:
        c0 = c0 / c0_norm

    c1 = c1 - c0.dot(c1) * c0
    c1_norm = c1.norm()
    if c1_norm < EPSILON:
        c1 = ti.Vector([0.0, 1.0, 0.0])
    else:
        c1 = c1 / c1_norm

    c2 = c0.cross(c1)
    return ti.Matrix(
        [
            [c0.x, c1.x, c2.x],
            [c0.y, c1.y, c2.y],
            [c0.z, c1.z, c2.z],
        ]
    )


@ti.func
def _rotation_vector_from_matrix(r):
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    c = ti.max(-1.0, ti.min(1.0, 0.5 * (trace - 1.0)))
    angle = ti.acos(c)
    theta = ti.Vector([0.0, 0.0, 0.0])
    if angle > 1e-5:
        denom = 2.0 * ti.sin(angle)
        axis = ti.Vector(
            [
                (r[2, 1] - r[1, 2]) / denom,
                (r[0, 2] - r[2, 0]) / denom,
                (r[1, 0] - r[0, 1]) / denom,
            ]
        )
        theta = axis * angle
    return theta


@ti.func
def _accumulate_particle_correction(particles: ti.template(), idx: ti.i32, corr: ti.math.vec3):
    ti.atomic_add(particles.dx_constraint[idx][0], corr[0])
    ti.atomic_add(particles.dx_constraint[idx][1], corr[1])
    ti.atomic_add(particles.dx_constraint[idx][2], corr[2])
    ti.atomic_add(particles.constraint_count[idx], 1.0)


@ti.data_oriented
class RigidBoxes:
    def __init__(self, centers, half_extents, masses, velocities=None, angular_velocities=None):
        centers = np.asarray(centers, dtype=np.float32)
        half_extents = np.asarray(half_extents, dtype=np.float32)
        masses = np.asarray(masses, dtype=np.float32)
        n = len(centers)
        if n == 0:
            raise ValueError("RigidBoxes needs at least one box")

        velocities = np.zeros((n, 3), dtype=np.float32) if velocities is None else np.asarray(velocities, dtype=np.float32)
        angular_velocities = (
            np.zeros((n, 3), dtype=np.float32)
            if angular_velocities is None
            else np.asarray(angular_velocities, dtype=np.float32)
        )

        rotations = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], n, axis=0)
        inv_masses = np.zeros(n, dtype=np.float32)
        inv_inertia_diag = np.zeros((n, 3), dtype=np.float32)
        for i, mass in enumerate(masses):
            if mass > 0.0:
                inv_masses[i] = 1.0 / mass
                hx, hy, hz = half_extents[i]
                inertia = np.array(
                    [
                        (mass / 3.0) * (hy * hy + hz * hz),
                        (mass / 3.0) * (hx * hx + hz * hz),
                        (mass / 3.0) * (hx * hx + hy * hy),
                    ],
                    dtype=np.float32,
                )
                inv_inertia_diag[i] = np.where(inertia > EPSILON, 1.0 / inertia, 0.0)

        self.num_boxes = n
        self.center = ti.Vector.field(3, dtype=ti.f32, shape=n)
        self.prev_center = ti.Vector.field(3, dtype=ti.f32, shape=n)
        self.pred_center = ti.Vector.field(3, dtype=ti.f32, shape=n)
        self.rotation = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n)
        self.prev_rotation = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n)
        self.pred_rotation = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n)
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=n)
        self.omega = ti.Vector.field(3, dtype=ti.f32, shape=n)
        self.inv_mass = ti.field(dtype=ti.f32, shape=n)
        self.inv_inertia_body_diag = ti.Vector.field(3, dtype=ti.f32, shape=n)
        self.half_extents = ti.Vector.field(3, dtype=ti.f32, shape=n)
        self.center_corr = ti.Vector.field(3, dtype=ti.f32, shape=n)
        self.theta_corr = ti.Vector.field(3, dtype=ti.f32, shape=n)

        self.center.from_numpy(centers)
        self.prev_center.from_numpy(centers)
        self.pred_center.from_numpy(centers)
        self.rotation.from_numpy(rotations)
        self.prev_rotation.from_numpy(rotations)
        self.pred_rotation.from_numpy(rotations)
        self.v.from_numpy(velocities)
        self.omega.from_numpy(angular_velocities)
        self.inv_mass.from_numpy(inv_masses)
        self.inv_inertia_body_diag.from_numpy(inv_inertia_diag)
        self.half_extents.from_numpy(half_extents)
        self.center_corr.fill(0.0)
        self.theta_corr.fill(0.0)

    @ti.func
    def inv_inertia_world_mul(self, box_id, vec):
        r = self.pred_rotation[box_id]
        local = r.transpose() @ vec
        diag = self.inv_inertia_body_diag[box_id]
        local = ti.Vector([local.x * diag.x, local.y * diag.y, local.z * diag.z])
        return r @ local

    @ti.func
    def point_velocity(self, box_id, world_point):
        r = world_point - self.pred_center[box_id]
        return self.v[box_id] + self.omega[box_id].cross(r)

    @ti.kernel
    def clear_corrections(self):
        for i in self.center_corr:
            self.center_corr[i] = ti.Vector([0.0, 0.0, 0.0])
            self.theta_corr[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def apply_corrections(self, relaxation: ti.f32):
        for i in self.pred_center:
            if self.inv_mass[i] == 0.0:
                continue
            self.pred_center[i] += relaxation * self.center_corr[i]
            theta = relaxation * self.theta_corr[i]
            if theta.norm() > EPSILON:
                self.pred_rotation[i] = _orthonormalize(_rotation_from_vector(theta) @ self.pred_rotation[i])


@ti.kernel
def apply_rigid_external_forces(boxes: ti.template(), dt: ti.f32):
    for i in boxes.v:
        if boxes.inv_mass[i] == 0.0:
            continue
        boxes.v[i] += GRAVITY * dt


@ti.kernel
def predict_rigid_boxes(boxes: ti.template(), dt: ti.f32):
    for i in boxes.center:
        boxes.prev_center[i] = boxes.center[i]
        boxes.prev_rotation[i] = boxes.rotation[i]
        if boxes.inv_mass[i] == 0.0:
            boxes.pred_center[i] = boxes.center[i]
            boxes.pred_rotation[i] = boxes.rotation[i]
            continue
        boxes.pred_center[i] = boxes.center[i] + boxes.v[i] * dt
        boxes.pred_rotation[i] = _orthonormalize(_rotation_from_vector(boxes.omega[i] * dt) @ boxes.rotation[i])


@ti.kernel
def update_rigid_velocities(boxes: ti.template(), dt: ti.f32):
    for i in boxes.center:
        if boxes.inv_mass[i] == 0.0:
            boxes.center[i] = boxes.pred_center[i]
            boxes.rotation[i] = boxes.pred_rotation[i]
            boxes.v[i] = ti.Vector([0.0, 0.0, 0.0])
            boxes.omega[i] = ti.Vector([0.0, 0.0, 0.0])
            continue

        boxes.v[i] = (boxes.pred_center[i] - boxes.center[i]) / dt
        delta_r = boxes.pred_rotation[i] @ boxes.rotation[i].transpose()
        boxes.omega[i] = _rotation_vector_from_matrix(delta_r) / dt
        boxes.center[i] = boxes.pred_center[i]
        boxes.rotation[i] = _orthonormalize(boxes.pred_rotation[i])


@ti.kernel
def sync_rigid_box_particles(
    particles: ti.template(),
    boxes: ti.template(),
    particle_indices: ti.template(),
    box_ids: ti.template(),
    local_vertices: ti.template(),
    count: ti.i32,
):
    for i in range(count):
        idx = particle_indices[i]
        box_id = box_ids[i]
        p = boxes.center[box_id] + boxes.rotation[box_id] @ local_vertices[i]
        particles.x[idx] = p
        particles.x_pred[idx] = p
        particles.pin_target[idx] = p
        particles.v[idx] = boxes.point_velocity(box_id, p)


@ti.kernel
def attachment_projection_jacobi(
    particles: ti.template(),
    boxes: ti.template(),
    particle_indices: ti.template(),
    box_ids: ti.template(),
    local_offsets: ti.template(),
    count: ti.i32,
    stiffness: ti.f32,
):
    for i in range(count):
        p_idx = particle_indices[i]
        box_id = box_ids[i]
        w_p = particles.solver_w(p_idx)
        inv_m = boxes.inv_mass[box_id]

        p = particles.x_pred[p_idx]
        anchor = boxes.pred_center[box_id] + boxes.pred_rotation[box_id] @ local_offsets[i]
        diff = p - anchor
        length = diff.norm()
        if length < EPSILON:
            continue

        n = diff / length
        r = anchor - boxes.pred_center[box_id]
        angular = boxes.inv_inertia_world_mul(box_id, r.cross(n))
        w_rigid = inv_m + angular.cross(r).dot(n)
        denom = w_p + w_rigid
        if denom < EPSILON:
            continue

        lam = stiffness * length / denom
        if w_p != 0.0:
            _accumulate_particle_correction(particles, p_idx, -w_p * lam * n)
        if inv_m != 0.0:
            center_corr = inv_m * lam * n
            theta_corr = lam * angular
            ti.atomic_add(boxes.center_corr[box_id][0], center_corr[0])
            ti.atomic_add(boxes.center_corr[box_id][1], center_corr[1])
            ti.atomic_add(boxes.center_corr[box_id][2], center_corr[2])
            ti.atomic_add(boxes.theta_corr[box_id][0], theta_corr[0])
            ti.atomic_add(boxes.theta_corr[box_id][1], theta_corr[1])
            ti.atomic_add(boxes.theta_corr[box_id][2], theta_corr[2])


@ti.kernel
def particle_box_collision_projection_jacobi(
    particles: ti.template(),
    boxes: ti.template(),
    start: ti.i32,
    end: ti.i32,
    box_id: ti.i32,
    margin: ti.f32,
    stiffness: ti.f32,
):
    for p_idx in range(start, end):
        w_p = particles.solver_w(p_idx)
        inv_m = boxes.inv_mass[box_id]
        if w_p + inv_m == 0.0:
            continue

        p = particles.x_pred[p_idx]
        center = boxes.pred_center[box_id]
        rot = boxes.pred_rotation[box_id]
        local = rot.transpose() @ (p - center)
        half = boxes.half_extents[box_id] + ti.Vector([margin, margin, margin])
        ax = ti.abs(local.x)
        ay = ti.abs(local.y)
        az = ti.abs(local.z)
        if ax > half.x or ay > half.y or az > half.z:
            continue

        px = half.x - ax
        py = half.y - ay
        pz = half.z - az
        local_n = ti.Vector([1.0, 0.0, 0.0])
        penetration = px
        if py < penetration:
            local_n = ti.Vector([0.0, 1.0, 0.0])
            penetration = py
        if pz < penetration:
            local_n = ti.Vector([0.0, 0.0, 1.0])
            penetration = pz

        if local_n.x != 0.0 and local.x < 0.0:
            local_n.x = -1.0
        if local_n.y != 0.0 and local.y < 0.0:
            local_n.y = -1.0
        if local_n.z != 0.0 and local.z < 0.0:
            local_n.z = -1.0

        n = rot @ local_n
        contact = p + n * penetration
        r = contact - center
        angular = boxes.inv_inertia_world_mul(box_id, r.cross(n))
        w_rigid = inv_m + angular.cross(r).dot(n)
        denom = w_p + w_rigid
        if denom < EPSILON:
            continue

        lam = stiffness * penetration / denom
        particle_corr = w_p * lam * n
        if w_p != 0.0:
            _accumulate_particle_correction(particles, p_idx, particle_corr)
            particles.dx_coll[p_idx] += particle_corr
            particles.coll_normal[p_idx] += n
            particles.coll_surface_v[p_idx] += boxes.point_velocity(box_id, contact)
            particles.coll_count[p_idx] += 1.0
        if inv_m != 0.0:
            center_corr = -inv_m * lam * n
            theta_corr = -lam * angular
            ti.atomic_add(boxes.center_corr[box_id][0], center_corr[0])
            ti.atomic_add(boxes.center_corr[box_id][1], center_corr[1])
            ti.atomic_add(boxes.center_corr[box_id][2], center_corr[2])
            ti.atomic_add(boxes.theta_corr[box_id][0], theta_corr[0])
            ti.atomic_add(boxes.theta_corr[box_id][1], theta_corr[1])
            ti.atomic_add(boxes.theta_corr[box_id][2], theta_corr[2])


@ti.kernel
def rigid_corner_cloth_collision_projection_jacobi(
    particles: ti.template(),
    boxes: ti.template(),
    box_ids: ti.template(),
    local_corners: ti.template(),
    tri_indices: ti.template(),
    bary_coords: ti.template(),
    normals: ti.template(),
    count: ti.template(),
    thickness: ti.f32,
    stiffness: ti.f32,
):
    for c in range(count[None]):
        box_id = box_ids[c]
        inv_m = boxes.inv_mass[box_id]
        if inv_m == 0.0:
            continue

        local = local_corners[c]
        corner = boxes.pred_center[box_id] + boxes.pred_rotation[box_id] @ local
        tri = tri_indices[c]
        bary = bary_coords[c]
        i0 = tri[0]
        i1 = tri[1]
        i2 = tri[2]
        p0 = particles.x_pred[i0]
        p1 = particles.x_pred[i1]
        p2 = particles.x_pred[i2]
        cloth_point = bary.x * p0 + bary.y * p1 + bary.z * p2

        n = normals[c]
        n_norm = n.norm()
        if n_norm < EPSILON:
            continue
        n = n / n_norm

        C = (corner - cloth_point).dot(n) - thickness
        if C >= 0.0:
            continue

        r = corner - boxes.pred_center[box_id]
        angular = boxes.inv_inertia_world_mul(box_id, r.cross(n))
        w_rigid = inv_m + angular.cross(r).dot(n)
        w0 = particles.solver_w(i0)
        w1 = particles.solver_w(i1)
        w2 = particles.solver_w(i2)
        w_cloth = w0 * bary.x * bary.x + w1 * bary.y * bary.y + w2 * bary.z * bary.z
        denom = w_rigid + w_cloth
        if denom < EPSILON:
            continue

        lam = -stiffness * C / denom
        center_corr = inv_m * lam * n
        theta_corr = lam * angular
        ti.atomic_add(boxes.center_corr[box_id][0], center_corr[0])
        ti.atomic_add(boxes.center_corr[box_id][1], center_corr[1])
        ti.atomic_add(boxes.center_corr[box_id][2], center_corr[2])
        ti.atomic_add(boxes.theta_corr[box_id][0], theta_corr[0])
        ti.atomic_add(boxes.theta_corr[box_id][1], theta_corr[1])
        ti.atomic_add(boxes.theta_corr[box_id][2], theta_corr[2])

        surface_v = boxes.point_velocity(box_id, corner)
        if w0 != 0.0:
            corr0 = -w0 * bary.x * lam * n
            _accumulate_particle_correction(particles, i0, corr0)
            particles.dx_coll[i0] += corr0
            particles.coll_normal[i0] += -n
            particles.coll_surface_v[i0] += surface_v
            particles.coll_count[i0] += 1.0
        if w1 != 0.0:
            corr1 = -w1 * bary.y * lam * n
            _accumulate_particle_correction(particles, i1, corr1)
            particles.dx_coll[i1] += corr1
            particles.coll_normal[i1] += -n
            particles.coll_surface_v[i1] += surface_v
            particles.coll_count[i1] += 1.0
        if w2 != 0.0:
            corr2 = -w2 * bary.z * lam * n
            _accumulate_particle_correction(particles, i2, corr2)
            particles.dx_coll[i2] += corr2
            particles.coll_normal[i2] += -n
            particles.coll_surface_v[i2] += surface_v
            particles.coll_count[i2] += 1.0


@ti.kernel
def rigid_floor_projection_jacobi(
    boxes: ti.template(),
    floor_y: ti.f32,
    margin: ti.f32,
    stiffness: ti.f32,
):
    for box_id in boxes.center:
        if boxes.inv_mass[box_id] == 0.0:
            continue
        half = boxes.half_extents[box_id]
        for corner in range(8):
            sx = -1.0
            sy = -1.0
            sz = -1.0
            if corner & 1:
                sx = 1.0
            if corner & 2:
                sy = 1.0
            if corner & 4:
                sz = 1.0
            local = ti.Vector([sx * half.x, sy * half.y, sz * half.z])
            point = boxes.pred_center[box_id] + boxes.pred_rotation[box_id] @ local
            penetration = floor_y + margin - point.y
            if penetration <= 0.0:
                continue

            n = ti.Vector([0.0, 1.0, 0.0])
            r = point - boxes.pred_center[box_id]
            angular = boxes.inv_inertia_world_mul(box_id, r.cross(n))
            w_rigid = boxes.inv_mass[box_id] + angular.cross(r).dot(n)
            if w_rigid < EPSILON:
                continue

            lam = stiffness * penetration / w_rigid
            center_corr = boxes.inv_mass[box_id] * lam * n
            theta_corr = lam * angular
            ti.atomic_add(boxes.center_corr[box_id][0], center_corr[0])
            ti.atomic_add(boxes.center_corr[box_id][1], center_corr[1])
            ti.atomic_add(boxes.center_corr[box_id][2], center_corr[2])
            ti.atomic_add(boxes.theta_corr[box_id][0], theta_corr[0])
            ti.atomic_add(boxes.theta_corr[box_id][1], theta_corr[1])
            ti.atomic_add(boxes.theta_corr[box_id][2], theta_corr[2])


@ti.kernel
def apply_rigid_floor_velocity_response(
    boxes: ti.template(),
    floor_y: ti.f32,
    restitution: ti.f32,
    mu_k: ti.f32,
):
    for box_id in boxes.center:
        if boxes.inv_mass[box_id] == 0.0:
            continue
        half = boxes.half_extents[box_id]
        for corner in range(8):
            sx = -1.0
            sy = -1.0
            sz = -1.0
            if corner & 1:
                sx = 1.0
            if corner & 2:
                sy = 1.0
            if corner & 4:
                sz = 1.0
            local = ti.Vector([sx * half.x, sy * half.y, sz * half.z])
            point = boxes.center[box_id] + boxes.rotation[box_id] @ local
            if point.y > floor_y + 1e-4:
                continue

            n = ti.Vector([0.0, 1.0, 0.0])
            r = point - boxes.center[box_id]
            v_point = boxes.v[box_id] + boxes.omega[box_id].cross(r)
            vn = v_point.dot(n)
            angular_n = boxes.inv_inertia_world_mul(box_id, r.cross(n))
            denom_n = boxes.inv_mass[box_id] + angular_n.cross(r).dot(n)
            if vn < 0.0 and denom_n > EPSILON:
                j = -(1.0 + restitution) * vn / denom_n
                impulse = j * n
                boxes.v[box_id] += boxes.inv_mass[box_id] * impulse
                boxes.omega[box_id] += boxes.inv_inertia_world_mul(box_id, r.cross(impulse))

            v_point = boxes.v[box_id] + boxes.omega[box_id].cross(r)
            vt = v_point - v_point.dot(n) * n
            vt_len = vt.norm()
            if vt_len > EPSILON and denom_n > EPSILON:
                tangent = vt / vt_len
                angular_t = boxes.inv_inertia_world_mul(box_id, r.cross(tangent))
                denom_t = boxes.inv_mass[box_id] + angular_t.cross(r).dot(tangent)
                if denom_t > EPSILON:
                    j_t = ti.min(mu_k * ti.abs(vn) / denom_n, vt_len / denom_t)
                    impulse_t = -j_t * tangent
                    boxes.v[box_id] += boxes.inv_mass[box_id] * impulse_t
                    boxes.omega[box_id] += boxes.inv_inertia_world_mul(box_id, r.cross(impulse_t))
