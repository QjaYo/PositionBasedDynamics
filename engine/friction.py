import taichi as ti

EPSILON = 1e-8


@ti.kernel
def apply_floor_friction(
    particles: ti.template(),
    floor_y: ti.f32,
    mu_s: ti.f32,
    mu_k: ti.f32
):
    """
    바닥 Coulomb 마찰 (논문 line 16)

    v_n = |v.y|  : 법선 방향 속도 크기 (바닥이 위로 밀어낸 속도)
    v_t = (v.x, v.z) : 접선 방향 속도

    정지마찰 (|v_t| <= mu_s * v_n): v_t = 0  → 완전히 멈춤
    운동마찰 (|v_t| >  mu_s * v_n): v_t 방향은 유지, 크기를 mu_k * v_n 만큼 줄임
    """
    for i in particles.x:
        if particles.w[i] == 0.0:
            continue
        if particles.x[i].y > floor_y:
            continue

        v_n = ti.abs(particles.v[i].y)
        v_t = ti.math.vec3(particles.v[i].x, 0.0, particles.v[i].z)
        v_t_norm = v_t.norm()

        if v_t_norm < EPSILON:
            continue

        if v_t_norm <= mu_s * v_n:
            # 정지마찰: 접선 속도 제거
            particles.v[i].x = 0.0
            particles.v[i].z = 0.0
        else:
            # 운동마찰: 접선 속도를 mu_k * v_n 만큼 감소
            scale = ti.max(0.0, 1.0 - mu_k * v_n / v_t_norm)
            particles.v[i].x *= scale
            particles.v[i].z *= scale
