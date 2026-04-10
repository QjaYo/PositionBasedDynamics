import taichi as ti


@ti.kernel
def damp_velocities(particles: ti.template(), k_damping: ti.f32):
    """
    논문 3.5절 dampVelocities
    강체 운동 성분은 보존하고 개별 진동(고주파)만 감쇠.
    v_i = lerp(v_i, v_cm + ω × r_i, k_damping)
    """
    n = particles.num_particles

    # (1) x_cm, (2) v_cm — 질량중심 위치/속도
    # w_i = 1/m_i 이므로 m_i = 1/w_i, w_i=0이면 고정 파티클(무한 질량) → 제외
    total_mass = 0.0
    x_cm = ti.Vector([0.0, 0.0, 0.0])
    v_cm = ti.Vector([0.0, 0.0, 0.0])
    for i in range(n):
        if particles.w[i] == 0.0:
            continue
        m = 1.0 / particles.w[i]
        total_mass += m
        x_cm += particles.x[i] * m
        v_cm += particles.v[i] * m
    x_cm /= total_mass
    v_cm /= total_mass

    # (3) L — 각운동량
    L = ti.Vector([0.0, 0.0, 0.0])
    for i in range(n):
        if particles.w[i] == 0.0:
            continue
        m = 1.0 / particles.w[i]
        r = particles.x[i] - x_cm
        L += r.cross(m * particles.v[i])

    # (4) I — 관성 텐서 (3x3 행렬)
    I = ti.Matrix([[0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0]])
    for i in range(n):
        if particles.w[i] == 0.0:
            continue
        m = 1.0 / particles.w[i]
        r = particles.x[i] - x_cm
        # r̃ (skew-symmetric matrix)
        r_tilde = ti.Matrix([[ 0.0, -r.z,  r.y],
                              [ r.z,  0.0, -r.x],
                              [-r.y,  r.x,  0.0]])
        I += m * r_tilde @ r_tilde.transpose()

    # (5) ω = I^-1 * L
    omega = I.inverse() @ L

    # (6-8) v_i = lerp(v_i, v_cm + ω × r_i, k_damping)
    for i in range(n):
        if particles.w[i] == 0.0:
            continue
        r = particles.x[i] - x_cm
        v_rigid = v_cm + omega.cross(r)
        particles.v[i] += k_damping * (v_rigid - particles.v[i])
