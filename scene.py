import numpy as np

from utils import load_obj, tetrahedralize, build_edges

FLOOR_Y = 0.0333


def create_scene(obj_path: str):
    """
    OBJ 파일을 로드해 기하 데이터를 반환
    반환:
        positions     : (N,3) float32 — 꼭짓점 초기 위치
        edges         : (E,2) int32   — 거리 제약용 엣지
        rest_lengths  : (E,)  float32
        tet_elems     : (M,4) int32   — 부피 제약용 사면체
        rest_volumes  : (M,)  float32
        surface_faces : (F,3) int32   — 렌더링용 표면 삼각형
    """

    # 로드 & 사면체화
    vertices, faces = load_obj(obj_path)
    tet_verts, tet_elems, surface_faces = tetrahedralize(vertices, faces)

    # 거리 제약 데이터
    edges = build_edges(tet_elems)
    p = tet_verts[edges[:, 0]]
    q = tet_verts[edges[:, 1]]
    rest_lengths = np.linalg.norm(p - q, axis=1).astype(np.float32)

    # 부피 제약 데이터
    v0 = tet_verts[tet_elems[:, 0]]
    v1 = tet_verts[tet_elems[:, 1]]
    v2 = tet_verts[tet_elems[:, 2]]
    v3 = tet_verts[tet_elems[:, 3]]
    rest_volumes = np.abs(np.einsum('ij,ij->i',
                    v1 - v0, np.cross(v2 - v0, v3 - v0)) / 6.0).astype(np.float32)

    print(f"[scene] verts={len(tet_verts)}, edges={len(edges)}, tets={len(tet_elems)}")
    print(f"[scene] rest_volumes: min={rest_volumes.min():.4e}, max={rest_volumes.max():.4e}, negative={(rest_volumes < 0).sum()}")

    return (tet_verts.astype(np.float32), edges, rest_lengths,
            tet_elems.astype(np.int32), rest_volumes, surface_faces)
