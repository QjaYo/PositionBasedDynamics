import taichi as ti

EPSILON = 1e-8


@ti.kernel
def apply_collision_restitution(
    particles: ti.template(),
    e: ti.f32,
):
    for i in particles.x:
        if particles.solver_w(i) == 0.0:
            continue
        if particles.dx_coll[i].norm() < EPSILON:
            continue

        count = particles.coll_count[i]
        if count <= 0.0:
            continue

        n = particles.coll_normal[i]
        n_norm = n.norm()
        if n_norm < EPSILON:
            continue
        n = n / n_norm

        surface_v = particles.coll_surface_v[i] / count
        v_rel_n = (particles.v_pre[i] - surface_v).dot(n)
        if v_rel_n >= 0.0:
            continue

        v_n_now = (particles.v[i] - surface_v).dot(n)
        target = -e * v_rel_n
        particles.v[i] += (target - v_n_now) * n


def apply_floor_restitution(particles, e, floor_v_y):
    apply_collision_restitution(particles, e)
