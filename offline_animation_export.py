import json
from pathlib import Path

import numpy as np
import taichi as ti

import engine.tearing as tearing
import render_animation_video
from engine.damping import damp_velocities
from engine.friction import apply_floor_friction
from engine.integrator import apply_external_forces, predict_positions, velocityUpdate
from engine.particles import Particles
from engine.restitution import apply_floor_restitution
from engine.solver import (
    JacobiConstraintSolver,
    bending_projection_jacobi,
    distance_projection_jacobi,
    volume_projection_jacobi,
)
from scene import create_offline_scene
from offline_scenarios import (
    apply_scenario_frame,
    apply_scenario_substep,
    clear_scenario,
    prepare_scenario,
    register_scenario_solver_constraints,
    scenario_log_lines,
    scenario_metadata,
)


DEFAULT_MATERIAL = "bunny_gray"


def prepare_output_dir(out_dir, overwrite):
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("frame_*.obj"))
    if existing and not overwrite:
        raise FileExistsError(f"{out_dir} already has frame_*.obj files. Use --overwrite to replace them.")
    for path in existing:
        path.unlink()


def write_obj(path, vertices, faces, face_materials=None):
    if face_materials is None or len(face_materials) != len(faces):
        face_materials = [DEFAULT_MATERIAL] * len(faces)

    with path.open("w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        current_material = None
        for face, material in zip(faces, face_materials):
            if material != current_material:
                f.write(f"usemtl {material}\n")
                current_material = material
            f.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def _field_or_none_vector(dim, dtype, array):
    if len(array) == 0:
        return None
    field = ti.Vector.field(dim, dtype=dtype, shape=len(array))
    field.from_numpy(array)
    return field


def _field_or_none(dtype, array):
    if len(array) == 0:
        return None
    field = ti.field(dtype=dtype, shape=len(array))
    field.from_numpy(array)
    return field


def export_animation_sequence(config):
    ASSET = config.asset
    OUT_DIR = config.obj_dir
    OVERWRITE = config.overwrite
    FRAMES = config.frames
    FPS = config.fps
    SUBSTEPS = config.substeps
    SOLVER_ITERS = config.solver_iters
    ARCH = config.arch
    MAX_PARTICLE_FACTOR = config.max_particle_factor
    MAX_EDGE_FACTOR = config.max_edge_factor
    K_DISTANCE = config.k_distance
    K_VOLUME = config.k_volume
    DAMPING = config.damping
    USE_GRAVITY = config.use_gravity
    GRAVITY_SCALE = float(getattr(config, "gravity_scale", 1.0))
    FLOOR_Y = config.floor_y
    MU_S = config.mu_s
    MU_K = config.mu_k
    E_RESTITUTION = config.e_restitution
    JACOBI_RELAXATION = float(config.jacobi_relaxation)
    K_BENDING = float(getattr(config, "scenario_options", {}).get("k_bending", 0.0))
    old_tear_ratio = tearing.tear_ratio
    old_max_tears = tearing.max_tears_per_call
    tearing.tear_ratio = float(config.tear_ratio)
    tearing.max_tears_per_call = int(config.max_tears_per_call)

    ti.init(arch=ti.gpu if ARCH == "gpu" else ti.cpu)

    out_dir = Path(OUT_DIR)
    preview_dir = Path(config.frame_preview_dir)
    prepare_output_dir(out_dir, OVERWRITE)
    if config.preview_during_simulation:
        render_animation_video.prepare_png_dir(preview_dir, OVERWRITE)

    scene_data = create_offline_scene(config)
    bbox_positions = scene_data.positions.astype(np.float32)
    bbox_diag = float(np.linalg.norm(np.max(bbox_positions, axis=0) - np.min(bbox_positions, axis=0)))
    scenario_state = prepare_scenario(config, scene_data, bbox_diag)

    positions = scene_data.positions.astype(np.float32)
    inv_masses = scene_data.inv_masses.astype(np.float32)
    edges = scene_data.edges.astype(np.int32)
    rest_lengths = scene_data.rest_lengths.astype(np.float32)
    tet_elems = scene_data.tet_elems.astype(np.int32)
    rest_volumes = scene_data.rest_volumes.astype(np.float32)
    surface_faces = scene_data.surface_faces.astype(np.int32)
    face_materials = list(scene_data.face_materials)
    bend_quads = scene_data.bend_quads.astype(np.int32)
    rest_bend_angles = scene_data.rest_bend_angles.astype(np.float32)

    max_particles = max(len(positions), int(np.ceil(MAX_PARTICLE_FACTOR * len(positions))))
    particles = Particles(positions, inv_masses, max_particles=max_particles)
    initial_num_particles = particles.num_particles

    num_edges = len(edges)
    max_edges = max(num_edges, int(np.ceil(MAX_EDGE_FACTOR * max(1, num_edges))))
    edges_padded = np.zeros((max_edges, 2), dtype=np.int32)
    rest_lengths_padded = np.zeros(max_edges, dtype=np.float32)
    if num_edges > 0:
        edges_padded[:num_edges] = edges
        rest_lengths_padded[:num_edges] = rest_lengths
    edges = edges_padded
    rest_lengths = rest_lengths_padded

    edges_field = ti.Vector.field(2, dtype=ti.i32, shape=max_edges)
    rest_lengths_field = ti.field(dtype=ti.f32, shape=max_edges)
    edges_field.from_numpy(edges)
    rest_lengths_field.from_numpy(rest_lengths)

    tet_elems_field = _field_or_none_vector(4, ti.i32, tet_elems)
    rest_volumes_field = _field_or_none(ti.f32, rest_volumes)
    bend_quads_field = _field_or_none_vector(4, ti.i32, bend_quads)
    rest_bend_angles_field = _field_or_none(ti.f32, rest_bend_angles)

    stiffness_distance = float(1.0 - (1.0 - K_DISTANCE) ** (1.0 / SOLVER_ITERS))
    stiffness_volume = float(1.0 - (1.0 - K_VOLUME) ** (1.0 / SOLVER_ITERS))
    stiffness_bending = float(1.0 - (1.0 - K_BENDING) ** (1.0 / SOLVER_ITERS)) if K_BENDING > 0.0 else 0.0

    def make_solver(active_num_edges):
        solver = JacobiConstraintSolver(relaxation=JACOBI_RELAXATION)
        if active_num_edges > 0:
            solver.register_jacobi(
                distance_projection_jacobi,
                particles,
                edges_field,
                rest_lengths_field,
                int(active_num_edges),
                stiffness_distance,
            )
        if tet_elems_field is not None and rest_volumes_field is not None:
            solver.register_jacobi(
                volume_projection_jacobi,
                particles,
                tet_elems_field,
                rest_volumes_field,
                stiffness_volume,
            )
        if bend_quads_field is not None and rest_bend_angles_field is not None and stiffness_bending > 0.0:
            solver.register_jacobi(
                bending_projection_jacobi,
                particles,
                bend_quads_field,
                rest_bend_angles_field,
                int(len(bend_quads)),
                stiffness_bending,
            )
        register_scenario_solver_constraints(config, scenario_state, particles, solver)
        return solver

    solver = make_solver(num_edges)

    dt = 1.0 / (FPS * SUBSTEPS)

    metadata = {
        "asset": ASSET,
        "output": config.output,
        "output_path": config.output_path,
        "scenario": config.scenario,
        "scene_kind": getattr(config, "scenario_options", {}).get("scene_kind", "tet"),
        "enable_tearing": bool(config.enable_tearing),
        "position_space": "simulation_initial_position",
        "frames": FRAMES,
        "fps": FPS,
        "substeps": SUBSTEPS,
        "solver_iters": SOLVER_ITERS,
        "solver_mode": "jacobi",
        "preview_during_simulation": bool(config.preview_during_simulation),
        "preview_every": int(config.preview_every),
        "jacobi_relaxation": float(JACOBI_RELAXATION),
        "k_distance": float(K_DISTANCE),
        "k_volume": float(K_VOLUME),
        "k_bending": float(K_BENDING),
        "num_bending_constraints": int(len(bend_quads)),
        "seed": config.seed,
        "use_gravity": bool(USE_GRAVITY),
        "damping": float(DAMPING),
        "floor_y": float(FLOOR_Y),
        "mu_s": float(MU_S),
        "mu_k": float(MU_K),
        "e_restitution": float(E_RESTITUTION),
        "tear_ratio": float(tearing.tear_ratio),
        "max_tears_per_frame": int(tearing.max_tears_per_call),
        "constraint_normalization": False,
        "materials": sorted(set(face_materials)),
    }
    metadata.update(scenario_metadata(config, scenario_state))
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[offline] output: {out_dir}")
    if config.preview_during_simulation:
        print(f"[offline] preview: every {config.preview_every} frames -> {preview_dir}")
    print(f"[offline] scenario: {config.scenario}")
    print(f"[offline] tearing: {'ON' if config.enable_tearing else 'OFF'}")
    for line in scenario_log_lines(config, scenario_state):
        print(f"[offline] {line}")
    print(f"[offline] solver: jacobi relaxation={JACOBI_RELAXATION:.4f}")
    if len(bend_quads) > 0:
        print(f"[offline] bending constraints: {len(bend_quads)} k={K_BENDING:.4f}")
    if config.enable_tearing:
        print(f"[offline] tear ratio: {tearing.tear_ratio:.4f}")
        print(f"[offline] max tears/frame: {tearing.max_tears_per_call}")

    for frame in range(FRAMES):
        # 프레임 시작: 시나리오별 고정점/초기 상태 갱신
        apply_scenario_frame(config, scenario_state, particles, frame)
        frame_topology_changed = False

        for _ in range(SUBSTEPS):
            # 충돌 누적값 초기화: restitution/friction이 이번 substep 접촉만 보도록 비움
            particles.dx_coll.fill(0.0)
            particles.coll_normal.fill(0.0)
            particles.coll_surface_v.fill(0.0)
            particles.coll_count.fill(0.0)

            # 시나리오별 pre-predict 처리: rigid body 예측, 외부 제어 상태 갱신 등
            apply_scenario_substep(config, scenario_state, particles, dt, "pre_predict")

            # (5) 외력 적용
            if USE_GRAVITY:
                apply_external_forces(particles, dt * GRAVITY_SCALE)

            # (6) 댐핑
            if DAMPING > 0.0:
                damp_velocities(particles, DAMPING)

            # (7) 예측 위치
            predict_positions(particles, dt)

            # (8) 시나리오별 충돌 제약 생성: static mesh grid, rigid corner 후보 등
            apply_scenario_substep(config, scenario_state, particles, dt, "post_predict")

            # (8) 바닥 충돌 제약 생성
            solver.generate_collision_constraints(particles, FLOOR_Y)

            # (9-11) 거리/부피/bending/충돌/rigid 제약 반복 투영
            solver.solve(SOLVER_ITERS)

            # solver 이후 시나리오별 후처리 hook
            apply_scenario_substep(config, scenario_state, particles, dt, "post_solve")

            # velocityUpdate 직전 v 백업 (restitution이 v_pre로 사용)
            particles.v_pre.copy_from(particles.v)

            # (12-14) 속도 & 위치 업데이트
            velocityUpdate(particles, dt)

            # (15) 반발
            apply_floor_restitution(particles, E_RESTITUTION, 0.0)

            # (16) 마찰
            apply_floor_friction(particles, MU_S, MU_K, dt)

            # 시나리오별 velocity 후처리: rigid velocity response, 렌더 mesh sync 등
            apply_scenario_substep(config, scenario_state, particles, dt, "post_velocity")

        # tearing은 offline에서는 frame 단위로 적용해 topology 재구성 비용을 줄임
        old_num_edges = num_edges
        old_num_particles = particles.num_particles
        changed = False
        if config.enable_tearing:
            particles.x_pred.copy_from(particles.x)
            num_edges, surface_faces, changed = tearing.apply_tearing(
                particles,
                edges,
                rest_lengths,
                num_edges,
                max_edges,
                tet_elems,
                rest_volumes,
                surface_faces,
            )

        # topology가 바뀐 경우 Taichi field와 solver 등록 정보를 새 edge/tet 배열로 동기화
        if changed or num_edges != old_num_edges or particles.num_particles != old_num_particles:
            frame_topology_changed = True
            edges_field.from_numpy(edges)
            rest_lengths_field.from_numpy(rest_lengths)
            if tet_elems_field is not None:
                tet_elems_field.from_numpy(tet_elems)
            solver = make_solver(num_edges)

        # 현재 프레임 geometry를 OBJ로 저장
        vertices = particles.x.to_numpy()[:particles.num_particles]
        if frame_topology_changed:
            surface_faces = tearing.rebuild_surface_faces_from_tets(tet_elems, vertices)
            face_materials = [DEFAULT_MATERIAL] * len(surface_faces)
        write_obj(out_dir / f"frame_{frame:04d}.obj", vertices, surface_faces, face_materials)

        # 긴 offline run 중간 확인용 preview frame 저장
        if config.preview_during_simulation and config.preview_every > 0 and frame % config.preview_every == 0:
            render_animation_video.render_preview_frame(config, frame)

        # 진행 상황 로그
        print(
            f"[offline] frame {frame + 1}/{FRAMES} "
            f"tears={particles.num_particles - initial_num_particles} "
            f"particles={particles.num_particles} edges={num_edges} faces={len(surface_faces)}",
            flush=True,
        )

    clear_scenario(config, scenario_state, particles)
    tearing.tear_ratio = old_tear_ratio
    tearing.max_tears_per_call = old_max_tears
    print("[offline] done")


def main():
    from main_offline import build_config

    export_animation_sequence(build_config())


if __name__ == "__main__":
    main()
