import numpy as np
import taichi as ti


EPSILON = 1e-8


@ti.func
def _closest_point_on_triangle(p, a, b, c):
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = ab.dot(ap)
    d2 = ac.dot(ap)
    result = a

    if d1 <= 0.0 and d2 <= 0.0:
        result = a
    else:
        bp = p - b
        d3 = ab.dot(bp)
        d4 = ac.dot(bp)
        if d3 >= 0.0 and d4 <= d3:
            result = b
        else:
            vc = d1 * d4 - d3 * d2
            if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                v = d1 / (d1 - d3)
                result = a + v * ab
            else:
                cp = p - c
                d5 = ab.dot(cp)
                d6 = ac.dot(cp)
                if d6 >= 0.0 and d5 <= d6:
                    result = c
                else:
                    vb = d5 * d2 - d1 * d6
                    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                        w = d2 / (d2 - d6)
                        result = a + w * ac
                    else:
                        va = d3 * d6 - d5 * d4
                        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
                            w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
                            result = b + w * (c - b)
                        else:
                            denom = 1.0 / (va + vb + vc)
                            v = vb * denom
                            w = vc * denom
                            result = a + ab * v + ac * w
    return result


@ti.func
def _triangle_normal(a, b, c):
    n = (b - a).cross(c - a)
    norm = n.norm()
    if norm < EPSILON:
        n = ti.Vector([0.0, 1.0, 0.0])
    else:
        n = n / norm
    return n


@ti.func
def _point_inside_triangle(p, a, b, c):
    v0 = c - a
    v1 = b - a
    v2 = p - a
    dot00 = v0.dot(v0)
    dot01 = v0.dot(v1)
    dot02 = v0.dot(v2)
    dot11 = v1.dot(v1)
    dot12 = v1.dot(v2)
    denom = dot00 * dot11 - dot01 * dot01
    inside = False

    if ti.abs(denom) > EPSILON:
        inv_denom = 1.0 / denom
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        inside = u >= -EPSILON and v >= -EPSILON and u + v <= 1.0 + EPSILON

    return inside


@ti.func
def _segment_triangle_crossing(x0, x1, a, b, c):
    n = _triangle_normal(a, b, c)
    d0 = (x0 - a).dot(n)
    d1 = (x1 - a).dot(n)
    denom = d0 - d1
    hit = False
    hit_point = ti.Vector([0.0, 0.0, 0.0])
    hit_normal = ti.Vector([0.0, 1.0, 0.0])
    hit_t = 1.0e20

    if ti.abs(denom) > EPSILON and d0 * d1 <= 0.0:
        t = d0 / denom
        if t >= -EPSILON and t <= 1.0 + EPSILON:
            t_clamped = ti.min(1.0, ti.max(0.0, t))
            p = x0 + t_clamped * (x1 - x0)
            if _point_inside_triangle(p, a, b, c):
                hit = True
                hit_point = p
                hit_t = t_clamped
                if d0 >= 0.0:
                    hit_normal = n
                else:
                    hit_normal = -n

    return hit, hit_point, hit_normal, hit_t


@ti.data_oriented
class StaticTriangleGrid:
    def __init__(
        self,
        vertices,
        faces,
        thickness,
        cell_size,
        max_tris_per_cell=96,
        neighbor_range=1,
    ):
        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32)
        tri_vertices = vertices[faces]
        self.num_triangles = len(faces)
        self.thickness = float(thickness)
        self.cell_size = max(float(cell_size), EPSILON)
        self.inv_cell_size = 1.0 / self.cell_size
        self.max_tris_per_cell = int(max_tris_per_cell)
        self.neighbor_range = int(neighbor_range)

        tri_min = np.min(tri_vertices, axis=1) - self.thickness
        tri_max = np.max(tri_vertices, axis=1) + self.thickness
        origin = np.min(tri_min, axis=0).astype(np.float32)
        bounds_max = np.max(tri_max, axis=0).astype(np.float32)
        dims = np.floor((bounds_max - origin) * self.inv_cell_size).astype(np.int32) + 1
        dims = np.maximum(dims, 1).astype(np.int32)
        self.num_cells = int(dims[0] * dims[1] * dims[2])

        cell_counts_np = np.zeros(self.num_cells, dtype=np.int32)
        cell_tri_ids_np = np.full((self.num_cells, self.max_tris_per_cell), -1, dtype=np.int32)
        overflow = 0

        for tri_id, (mn, mx) in enumerate(zip(tri_min, tri_max)):
            lo = np.floor((mn - origin) * self.inv_cell_size).astype(np.int32)
            hi = np.floor((mx - origin) * self.inv_cell_size).astype(np.int32)
            lo = np.maximum(lo, 0)
            hi = np.minimum(hi, dims - 1)
            for ix in range(int(lo[0]), int(hi[0]) + 1):
                for iy in range(int(lo[1]), int(hi[1]) + 1):
                    for iz in range(int(lo[2]), int(hi[2]) + 1):
                        cell = int((ix * dims[1] + iy) * dims[2] + iz)
                        slot = int(cell_counts_np[cell])
                        if slot < self.max_tris_per_cell:
                            cell_tri_ids_np[cell, slot] = tri_id
                            cell_counts_np[cell] += 1
                        else:
                            overflow += 1

        self.overflow = int(overflow)
        self.origin = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.dims = ti.Vector.field(3, dtype=ti.i32, shape=())
        self.tri_a = ti.Vector.field(3, dtype=ti.f32, shape=self.num_triangles)
        self.tri_b = ti.Vector.field(3, dtype=ti.f32, shape=self.num_triangles)
        self.tri_c = ti.Vector.field(3, dtype=ti.f32, shape=self.num_triangles)
        self.cell_counts = ti.field(dtype=ti.i32, shape=self.num_cells)
        self.cell_tri_ids = ti.field(dtype=ti.i32, shape=(self.num_cells, self.max_tris_per_cell))

        self.origin[None] = ti.Vector([float(origin[0]), float(origin[1]), float(origin[2])])
        self.dims[None] = ti.Vector([int(dims[0]), int(dims[1]), int(dims[2])])
        self.tri_a.from_numpy(tri_vertices[:, 0, :].astype(np.float32))
        self.tri_b.from_numpy(tri_vertices[:, 1, :].astype(np.float32))
        self.tri_c.from_numpy(tri_vertices[:, 2, :].astype(np.float32))
        self.cell_counts.from_numpy(cell_counts_np)
        self.cell_tri_ids.from_numpy(cell_tri_ids_np)

    @ti.kernel
    def generate_constraints(
        self,
        particles: ti.template(),
        particle_indices: ti.template(),
        num_particles: ti.i32,
        out_count: ti.template(),
        out_particle_indices: ti.template(),
        out_surface_points: ti.template(),
        out_surface_normals: ti.template(),
        max_constraints: ti.i32,
    ):
        out_count[None] = 0
        origin = self.origin[None]
        dims = self.dims[None]
        for local_idx in range(num_particles):
            p_idx = particle_indices[local_idx]
            if particles.solver_w(p_idx) == 0.0:
                continue

            x0 = particles.x[p_idx]
            p = particles.x_pred[p_idx]
            seg_min = ti.Vector([
                ti.min(x0.x, p.x),
                ti.min(x0.y, p.y),
                ti.min(x0.z, p.z),
            ]) - self.thickness
            seg_max = ti.Vector([
                ti.max(x0.x, p.x),
                ti.max(x0.y, p.y),
                ti.max(x0.z, p.z),
            ]) + self.thickness
            lo = ti.floor((seg_min - origin) * self.inv_cell_size, ti.i32)
            hi = ti.floor((seg_max - origin) * self.inv_cell_size, ti.i32)
            lo = ti.Vector([
                ti.max(0, ti.min(lo.x, dims.x - 1)),
                ti.max(0, ti.min(lo.y, dims.y - 1)),
                ti.max(0, ti.min(lo.z, dims.z - 1)),
            ])
            hi = ti.Vector([
                ti.max(0, ti.min(hi.x, dims.x - 1)),
                ti.max(0, ti.min(hi.y, dims.y - 1)),
                ti.max(0, ti.min(hi.z, dims.z - 1)),
            ])

            hit_found = False
            best_hit_t = 1.0e20
            best_hit_point = ti.Vector([0.0, 0.0, 0.0])
            best_hit_normal = ti.Vector([0.0, 1.0, 0.0])
            best_dist = 1.0e20
            best_point = ti.Vector([0.0, 0.0, 0.0])
            best_normal = ti.Vector([0.0, 1.0, 0.0])
            found = False

            for ix in range(lo.x, hi.x + 1):
                for iy in range(lo.y, hi.y + 1):
                    for iz in range(lo.z, hi.z + 1):
                        cell = (ix * dims.y + iy) * dims.z + iz
                        count = self.cell_counts[cell]
                        for slot in range(count):
                            tri_id = self.cell_tri_ids[cell, slot]
                            a = self.tri_a[tri_id]
                            b = self.tri_b[tri_id]
                            c = self.tri_c[tri_id]

                            hit, hit_point, hit_normal, hit_t = _segment_triangle_crossing(x0, p, a, b, c)
                            if hit and hit_t < best_hit_t:
                                hit_found = True
                                best_hit_t = hit_t
                                best_hit_point = hit_point
                                best_hit_normal = hit_normal

                            closest = _closest_point_on_triangle(p, a, b, c)
                            delta = p - closest
                            dist = delta.norm()
                            if dist < best_dist:
                                normal = _triangle_normal(a, b, c)
                                if dist > EPSILON:
                                    normal = delta / dist
                                else:
                                    side = (x0 - a).dot(normal)
                                    if side < 0.0:
                                        normal = -normal
                                best_dist = dist
                                best_point = closest
                                best_normal = normal
                                found = True

            if hit_found or (found and best_dist < self.thickness):
                out_idx = ti.atomic_add(out_count[None], 1)
                if out_idx < max_constraints:
                    out_particle_indices[out_idx] = p_idx
                    if hit_found:
                        out_surface_normals[out_idx] = best_hit_normal
                        out_surface_points[out_idx] = best_hit_point + best_hit_normal * self.thickness
                    else:
                        out_surface_normals[out_idx] = best_normal
                        out_surface_points[out_idx] = best_point + best_normal * self.thickness

    @ti.kernel
    def generate_constraints_with_offsets(
        self,
        particles: ti.template(),
        particle_indices: ti.template(),
        num_particles: ti.i32,
        prev_offset_field: ti.template(),
        current_offset_field: ti.template(),
        out_count: ti.template(),
        out_particle_indices: ti.template(),
        out_surface_points: ti.template(),
        out_surface_normals: ti.template(),
        max_constraints: ti.i32,
    ):
        out_count[None] = 0
        origin = self.origin[None]
        dims = self.dims[None]
        prev_offset = prev_offset_field[None]
        current_offset = current_offset_field[None]
        for local_idx in range(num_particles):
            p_idx = particle_indices[local_idx]
            if particles.solver_w(p_idx) == 0.0:
                continue

            x0 = particles.x[p_idx] - prev_offset
            p = particles.x_pred[p_idx] - current_offset
            seg_min = ti.Vector([
                ti.min(x0.x, p.x),
                ti.min(x0.y, p.y),
                ti.min(x0.z, p.z),
            ]) - self.thickness
            seg_max = ti.Vector([
                ti.max(x0.x, p.x),
                ti.max(x0.y, p.y),
                ti.max(x0.z, p.z),
            ]) + self.thickness
            lo = ti.floor((seg_min - origin) * self.inv_cell_size, ti.i32)
            hi = ti.floor((seg_max - origin) * self.inv_cell_size, ti.i32)
            lo = ti.Vector([
                ti.max(0, ti.min(lo.x, dims.x - 1)),
                ti.max(0, ti.min(lo.y, dims.y - 1)),
                ti.max(0, ti.min(lo.z, dims.z - 1)),
            ])
            hi = ti.Vector([
                ti.max(0, ti.min(hi.x, dims.x - 1)),
                ti.max(0, ti.min(hi.y, dims.y - 1)),
                ti.max(0, ti.min(hi.z, dims.z - 1)),
            ])

            hit_found = False
            best_hit_t = 1.0e20
            best_hit_point = ti.Vector([0.0, 0.0, 0.0])
            best_hit_normal = ti.Vector([0.0, 1.0, 0.0])
            best_dist = 1.0e20
            best_point = ti.Vector([0.0, 0.0, 0.0])
            best_normal = ti.Vector([0.0, 1.0, 0.0])
            found = False

            for ix in range(lo.x, hi.x + 1):
                for iy in range(lo.y, hi.y + 1):
                    for iz in range(lo.z, hi.z + 1):
                        cell = (ix * dims.y + iy) * dims.z + iz
                        count = self.cell_counts[cell]
                        for slot in range(count):
                            tri_id = self.cell_tri_ids[cell, slot]
                            a = self.tri_a[tri_id]
                            b = self.tri_b[tri_id]
                            c = self.tri_c[tri_id]

                            hit, hit_point, hit_normal, hit_t = _segment_triangle_crossing(x0, p, a, b, c)
                            if hit and hit_t < best_hit_t:
                                hit_found = True
                                best_hit_t = hit_t
                                best_hit_point = hit_point
                                best_hit_normal = hit_normal

                            closest = _closest_point_on_triangle(p, a, b, c)
                            delta = p - closest
                            dist = delta.norm()
                            if dist < best_dist:
                                normal = _triangle_normal(a, b, c)
                                if dist > EPSILON:
                                    normal = delta / dist
                                else:
                                    side = (x0 - a).dot(normal)
                                    if side < 0.0:
                                        normal = -normal
                                best_dist = dist
                                best_point = closest
                                best_normal = normal
                                found = True

            if hit_found or (found and best_dist < self.thickness):
                out_idx = ti.atomic_add(out_count[None], 1)
                if out_idx < max_constraints:
                    out_particle_indices[out_idx] = p_idx
                    if hit_found:
                        out_surface_normals[out_idx] = best_hit_normal
                        out_surface_points[out_idx] = best_hit_point + current_offset + best_hit_normal * self.thickness
                    else:
                        out_surface_normals[out_idx] = best_normal
                        out_surface_points[out_idx] = best_point + current_offset + best_normal * self.thickness
