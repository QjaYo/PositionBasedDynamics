import taichi as ti

EPSILON = 1e-8


@ti.kernel
def apply_collision_friction(
    particles: ti.template(),
    mu_s: ti.f32,
    mu_k: ti.f32,
    dt: ti.f32,
):
    for i in particles.x:
        if particles.solver_w(i) == 0.0:
            continue

        dx_coll_mag = particles.dx_coll[i].norm()
        if dx_coll_mag < EPSILON:
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
        rel_v = particles.v[i] - surface_v
        rel_n = rel_v.dot(n) * n
        rel_t = rel_v - rel_n
        rel_t_norm = rel_t.norm()
        if rel_t_norm < EPSILON:
            continue

        n_impulse = dx_coll_mag / dt
        if rel_t_norm < mu_s * n_impulse:
            particles.v[i] -= rel_t
        else:
            reduction = ti.min(mu_k * n_impulse, rel_t_norm)
            particles.v[i] -= reduction * rel_t.normalized()


def apply_floor_friction(particles, mu_s, mu_k, dt):
    apply_collision_friction(particles, mu_s, mu_k, dt)
