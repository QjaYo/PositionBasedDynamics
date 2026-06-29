import numpy as np

import simulation_config as sim_config

tear_ratio = sim_config.TEAR_RATIO
epsilon = 1e-8
max_tears_per_call = sim_config.MAX_TEARS_PER_CALL
candidate_sort_limit = 10000

TET_FACE_LOCAL = (
    (0, 1, 2, 3),
    (0, 1, 3, 2),
    (0, 2, 3, 1),
    (1, 2, 3, 0),
)

def is_below_plane(normal, plane_point, point):
    norm = np.linalg.norm(normal)
    if norm < epsilon:
        return False
    normal = normal / np.linalg.norm(normal)
    return np.dot(normal, point - plane_point) < 0

def build_edges_from_tets(tet_elems):
    edge_set = set()
    for tet in tet_elems:
        i0, i1, i2, i3 = [int(v) for v in tet]
        for a, b in ((i0, i1), (i0, i2), (i0, i3), (i1, i2), (i1, i3), (i2, i3)):
            edge_set.add((min(a, b), max(a, b)))
    return np.array(sorted(edge_set), dtype=np.int32)


def rebuild_surface_faces_from_tets(tet_elems, x_ref):
    face_map = {}
    for tet in tet_elems:
        tet = [int(v) for v in tet]
        for i0, i1, i2, i_opp in TET_FACE_LOCAL:
            face = [tet[i0], tet[i1], tet[i2]]
            key = tuple(sorted(face))
            if key in face_map:
                face_map[key][0] += 1
                continue

            p0 = x_ref[face[0]]
            p1 = x_ref[face[1]]
            p2 = x_ref[face[2]]
            p_opp = x_ref[tet[i_opp]]
            normal = np.cross(p1 - p0, p2 - p0)
            centroid = (p0 + p1 + p2) / 3.0
            if np.dot(normal, p_opp - centroid) > 0.0:
                face = [face[0], face[2], face[1]]
            face_map[key] = [1, face]

    faces = [face for count, face in face_map.values() if count == 1]
    return np.asarray(faces, dtype=np.int32)


def rebuild_rest_lengths(new_edges, old_edges, old_rest_lengths, p_old, p_new, x_ref):
    rest_map = {}
    for i in range(len(old_edges)):
        a = int(old_edges[i][0])
        b = int(old_edges[i][1])
        rest_map[(min(a, b), max(a, b))] = float(old_rest_lengths[i])

    new_rest_lengths = np.zeros(len(new_edges), dtype=np.float32)
    for i in range(len(new_edges)):
        a = int(new_edges[i][0])
        b = int(new_edges[i][1])
        mapped_a = p_old if a == p_new else a
        mapped_b = p_old if b == p_new else b
        key = (min(mapped_a, mapped_b), max(mapped_a, mapped_b))
        if key in rest_map:
            new_rest_lengths[i] = rest_map[key]
        else:
            new_rest_lengths[i] = np.linalg.norm(x_ref[mapped_a] - x_ref[mapped_b])
    return new_rest_lengths

def _build_split_candidate(
    particles,
    edges,
    rest_lengths,
    num_edges,
    max_edges,
    tet_elems,
    split_particle,
    other_particle,
    x_ref,
):
    p_new = particles.num_particles
    if p_new >= particles.max_particles:
        return None

    normal = x_ref[split_particle] - x_ref[other_particle]
    normal_norm = np.linalg.norm(normal)
    if normal_norm < epsilon:
        return None
    normal = normal / normal_norm
    plane_point = x_ref[split_particle]

    incident_mask = np.any(tet_elems == split_particle, axis=1)
    incident_tets = np.nonzero(incident_mask)[0]
    if len(incident_tets) == 0:
        return None

    tet_centroids = x_ref[tet_elems[incident_tets]].mean(axis=1)
    side = (tet_centroids - plane_point) @ normal
    split_tets = incident_tets[side < 0.0]
    if len(split_tets) == 0 or len(split_tets) == len(incident_tets):
        return None

    proposed_tets = tet_elems.copy()
    for t_idx in split_tets:
        for local_idx in range(4):
            if proposed_tets[t_idx][local_idx] == split_particle:
                proposed_tets[t_idx][local_idx] = p_new

    new_edges = build_edges_from_tets(proposed_tets)
    new_num_edges = len(new_edges)
    if new_num_edges > max_edges:
        return None

    new_rest_lengths = rebuild_rest_lengths(
        new_edges,
        edges[:num_edges],
        rest_lengths[:num_edges],
        split_particle,
        p_new,
        x_ref,
    )
    return split_particle, p_new, proposed_tets, new_edges, new_rest_lengths, new_num_edges


def apply_tearing(
    particles,
    edges,
    rest_lengths,
    num_edges,
    max_edges,
    tet_elems,
    rest_volumes,
    surface_faces,
):
    changed = False
    tears_done = 0

    while tears_done < max_tears_per_call:
        if num_edges <= 0 or particles.num_particles >= particles.max_particles:
            return num_edges, surface_faces, changed

        x_pred = particles.x_pred.to_numpy()[:particles.num_particles]
        active_edges = edges[:num_edges]
        edge_a = active_edges[:, 0]
        edge_b = active_edges[:, 1]

        current_lengths = np.linalg.norm(x_pred[edge_a] - x_pred[edge_b], axis=1)
        active_rest_lengths = rest_lengths[:num_edges]
        valid = active_rest_lengths > epsilon

        strains = np.zeros(num_edges, dtype=np.float32)
        strains[valid] = current_lengths[valid] / active_rest_lengths[valid]
        candidate_indices = np.nonzero(strains > tear_ratio)[0]
        if len(candidate_indices) == 0:
            return num_edges, surface_faces, changed

        candidate_strains = strains[candidate_indices]
        if len(candidate_indices) > candidate_sort_limit:
            top_local = np.argpartition(candidate_strains, -candidate_sort_limit)[-candidate_sort_limit:]
            candidate_indices = candidate_indices[top_local]
            candidate_strains = strains[candidate_indices]
        ordered_candidates = candidate_indices[np.argsort(candidate_strains)[::-1]]
        split = None
        for tear_idx in ordered_candidates:
            p1 = int(edges[tear_idx][0])
            p2 = int(edges[tear_idx][1])

            split = _build_split_candidate(
                particles,
                edges,
                rest_lengths,
                num_edges,
                max_edges,
                tet_elems,
                p1,
                p2,
                x_pred,
            )
            if split is not None:
                break

            split = _build_split_candidate(
                particles,
                edges,
                rest_lengths,
                num_edges,
                max_edges,
                tet_elems,
                p2,
                p1,
                x_pred,
            )
            if split is not None:
                break

        if split is None:
            return num_edges, surface_faces, changed

        p_old, p_new, proposed_tets, new_edges, new_rest_lengths, new_num_edges = split

        particles.x[p_new] = particles.x[p_old]
        particles.x_pred[p_new] = particles.x_pred[p_old]
        particles.v[p_new] = particles.v[p_old]
        particles.v_pre[p_new] = particles.v_pre[p_old]
        particles.w[p_new] = particles.w[p_old]
        particles.pinned[p_new] = 0
        particles.pin_target[p_new] = particles.x_pred[p_new]
        particles.num_particles += 1

        tet_elems[:, :] = proposed_tets
        edges[:new_num_edges] = new_edges
        rest_lengths[:new_num_edges] = new_rest_lengths
        num_edges = new_num_edges
        changed = True
        tears_done += 1


    return num_edges, surface_faces, changed
