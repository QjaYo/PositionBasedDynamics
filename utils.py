import numpy as np
import tetgen
import pymeshfix


def load_obj(path: str):
    vertices, faces = [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'v':
                vertices.append([float(x) for x in parts[1:4]])
            elif parts[0] == 'f':
                idx = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                if len(idx) == 3:
                    faces.append(idx)
    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32)


def tetrahedralize(vertices, faces):
    """
    tetgen으로 표면 메시 → 사면체 메시 변환
    반환: tet_vertices (N,3), tet_elements (M,4), surface_faces (F,3)
    """
    meshfix = pymeshfix.MeshFix(vertices, faces)
    meshfix.repair()
    surface_faces = meshfix.faces
    vertices = meshfix.points

    tet = tetgen.TetGen(vertices, surface_faces)
    tet_vertices, tet_elements, _, _ = tet.tetrahedralize(order=1, mindihedral=10, minratio=1.5, maxvolume=5e-3)
    return tet_vertices, tet_elements, surface_faces


def build_edges(tet_elements):
    """
    사면체 배열에서 중복 없는 엣지 목록 추출
    각 사면체는 6개의 엣지를 가짐: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
    """
    edge_set = set()
    for tet in tet_elements:
        i0, i1, i2, i3 = tet
        for a, b in [(i0,i1),(i0,i2),(i0,i3),(i1,i2),(i1,i3),(i2,i3)]:
            edge_set.add((min(a,b), max(a,b)))
    return np.array(sorted(edge_set), dtype=np.int32)
