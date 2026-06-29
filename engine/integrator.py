import taichi as ti

GRAVITY = ti.Vector([0.0, -9.8, 0.0])


@ti.kernel
def apply_external_forces(particles: ti.template(), dt: ti.f32):
    """(5) v_i <- v_i + dt * w_i * f_ext"""
    for i in particles.x:
        if particles.solver_w(i) == 0.0:
            continue
        particles.v[i] += GRAVITY * dt


@ti.kernel
def predict_positions(particles: ti.template(), dt: ti.f32):
    """(7) p_i <- x_i + dt * v_i"""
    for i in particles.x:
        if particles.pinned[i] != 0:
            particles.x_pred[i] = particles.pin_target[i]
            continue
        if particles.w[i] == 0.0:
            particles.x_pred[i] = particles.x[i]
            continue
        particles.x_pred[i] = particles.x[i] + particles.v[i] * dt


@ti.kernel
def velocityUpdate(particles: ti.template(), dt: ti.f32):
    """(12-14) v_i <- (p_i - x_i) / dt,  x_i <- p_i"""
    for i in particles.x:
        if particles.pinned[i] != 0:
            particles.x[i] = particles.pin_target[i]
            particles.x_pred[i] = particles.pin_target[i]
            particles.v[i] = ti.math.vec3(0.0, 0.0, 0.0)
            continue
        if particles.w[i] == 0.0:
            continue
        particles.v[i] = (particles.x_pred[i] - particles.x[i]) / dt
        particles.x[i] = particles.x_pred[i]
