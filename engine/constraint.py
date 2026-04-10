import taichi as ti

from scene import FLOOR_Y


@ti.func
def distance_constraint(p0: ti.math.vec3, p1: ti.math.vec3, rest_length: ti.f32):
    """
    거리 제약
    C = |p1 - p0| - d
    ∇p0 C = -(p1-p0)/|p1-p0|
    ∇p1 C =  (p1-p0)/|p1-p0|
    반환: C, ∇p0 C, ∇p1 C
    """
    diff = p1 - p0
    length = diff.norm()
    n = diff / length
    C = length - rest_length
    return C, -n, n


@ti.func
def volume_constraint(p0: ti.math.vec3, p1: ti.math.vec3, p2: ti.math.vec3, p3: ti.math.vec3, rest_volume: ti.f32):
    """
    사면체 부피 제약
    V = (1/6) * (p1-p0) · ((p2-p0) × (p3-p0))
    C = V - V0
    ∇p1 C = (1/6) * (p2-p0) × (p3-p0)
    ∇p2 C = (1/6) * (p3-p0) × (p1-p0)
    ∇p3 C = (1/6) * (p1-p0) × (p2-p0)
    ∇p0 C = -(∇p1 C + ∇p2 C + ∇p3 C)
    반환: C, ∇p0 C, ∇p1 C, ∇p2 C, ∇p3 C
    """
    grad1 = (p2 - p0).cross(p3 - p0) / 6.0
    grad2 = (p3 - p0).cross(p1 - p0) / 6.0
    grad3 = (p1 - p0).cross(p2 - p0) / 6.0
    grad0 = -(grad1 + grad2 + grad3)

    V = (p1 - p0).dot((p2 - p0).cross(p3 - p0)) / 6.0
    C = V - rest_volume
    return C, grad0, grad1, grad2, grad3


@ti.func
def floor_constraint(p: ti.math.vec3):
    """
    바닥 충돌 제약 (inequality: C < 0 일 때만 적용)
    C = p.y - FLOOR_Y
    ∇C = (0, 1, 0)
    반환: C, ∇C
    """
    C = p.y - FLOOR_Y
    grad = ti.math.vec3(0.0, 1.0, 0.0)
    return C, grad