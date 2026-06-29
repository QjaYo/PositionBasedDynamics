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
    distance_projection_jacobi,
    volume_projection_jacobi,
)
from scene import create_scene
from offline_scenarios import (
    apply_scenario_frame,
    apply_scenario_substep,
    clear_scenario,
    prepare_scenario,
    scenario_log_lines,
    scenario_metadata,
)



def prepare_output_dir(out_dir, overwrite):
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("frame_*.obj"))
    if existing and not overwrite:
        raise FileExistsError(f"{out_dir} already has frame_*.obj files. Use --overwrite to replace them.")
    for path in existing:
        path.unlink()


def write_obj(path, vertices, faces):
    with path.open("w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in faces:
            f.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")



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
    FLOOR_Y = config.floor_y
    MU_S = config.mu_s
    MU_K = config.mu_k
    E_RESTITUTION = config.e_restitution
    JACOBI_RELAXATION = float(config.jacobi_relaxation)
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

    positions, edges, rest_lengths, tet_elems, rest_volumes, surface_faces = create_scene(ASSET)
    bbox_diag = float(np.linalg.norm(np.max(positions, axis=0) - np.min(positions, axis=0)))
    scenario_state = prepare_scenario(config, positions, surface_faces, bbox_diag)

    max_particles = max(len(positions), int(np.ceil(MAX_PARTICLE_FACTOR * len(positions))))
    particles = Particles(positions, np.ones(len(positions), dtype=np.float32), max_particles=max_particles)
    initial_num_particles = particles.num_particles

    num_edges = len(edges)
    max_edges = max(num_edges, int(np.ceil(MAX_EDGE_FACTOR * num_edges)))
    edges_padded = np.zeros((max_edges, 2), dtype=np.int32)
    rest_lengths_padded = np.zeros(max_edges, dtype=np.float32)
    edges_padded[:num_edges] = edges
    rest_lengths_padded[:num_edges] = rest_lengths
    edges = edges_padded
    rest_lengths = rest_lengths_padded

    edges_field = ti.Vector.field(2, dtype=ti.i32, shape=max_edges)
    rest_lengths_field = ti.field(dtype=ti.f32, shape=max_edges)
    tet_elems_field = ti.Vector.field(4, dtype=ti.i32, shape=len(tet_elems))
    rest_volumes_field = ti.field(dtype=ti.f32, shape=len(rest_volumes))

    edges_field.from_numpy(edges)
    rest_lengths_field.from_numpy(rest_lengths)
    tet_elems_field.from_numpy(tet_elems)
    rest_volumes_field.from_numpy(rest_volumes)

    stiffness_distance = float(1.0 - (1.0 - K_DISTANCE) ** (1.0 / SOLVER_ITERS))
    stiffness_volume = float(1.0 - (1.0 - K_VOLUME) ** (1.0 / SOLVER_ITERS))

    def make_solver(active_num_edges):
        solver = JacobiConstraintSolver(relaxation=JACOBI_RELAXATION)
        solver.register_jacobi(
            distance_projection_jacobi,
            particles,
            edges_field,
            rest_lengths_field,
            int(active_num_edges),
            stiffness_distance,
        )
        solver.register_jacobi(
            volume_projection_jacobi,
            particles,
            tet_elems_field,
            rest_volumes_field,
            stiffness_volume,
        )
        return solver

    solver = make_solver(num_edges)

    dt = 1.0 / (FPS * SUBSTEPS)

    metadata = {
        "asset": ASSET,
        "output": config.output,
        "output_path": config.output_path,
        "scenario": config.scenario,
        "enable_tearing": bool(config.enable_tearing),
        "position_space": "rest_position_after_tetgen",
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
        "seed": config.seed,
        "use_gravity": bool(USE_GRAVITY),
        "damping": float(DAMPING),
        "floor_y": float(FLOOR_Y),
        "mu_s": float(MU_S),
        "mu_k": float(MU_K),
        "e_restitution": float(E_RESTITUTION),
        "tear_ratio": float(tearing.tear_ratio),
        "max_tears_per_substep": int(tearing.max_tears_per_call),
        "constraint_normalization": False,
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
    if config.enable_tearing:
        print(f"[offline] tear ratio: {tearing.tear_ratio:.4f}")
        print(f"[offline] max tears/frame: {tearing.max_tears_per_call}")

    for frame in range(FRAMES):
        apply_scenario_frame(config, scenario_state, particles, frame)
        frame_topology_changed = False

        for _ in range(SUBSTEPS):
            particles.dx_coll.fill(0.0)
            particles.coll_normal.fill(0.0)
            particles.coll_surface_v.fill(0.0)
            particles.coll_count.fill(0.0)
            if USE_GRAVITY:
                apply_external_forces(particles, dt)
            if DAMPING > 0.0:
                damp_velocities(particles, DAMPING)
            predict_positions(particles, dt)
            apply_scenario_substep(config, scenario_state, particles, dt, "post_predict")

            solver.generate_collision_constraints(particles, FLOOR_Y)
            solver.solve(SOLVER_ITERS)
            apply_scenario_substep(config, scenario_state, particles, dt, "post_solve")
            particles.v_pre.copy_from(particles.v)
            velocityUpdate(particles, dt)
            apply_floor_restitution(particles, E_RESTITUTION, 0.0)
            apply_floor_friction(particles, MU_S, MU_K, dt)
            apply_scenario_substep(config, scenario_state, particles, dt, "post_velocity")

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
        if changed or num_edges != old_num_edges or particles.num_particles != old_num_particles:
            frame_topology_changed = True
            edges_field.from_numpy(edges)
            rest_lengths_field.from_numpy(rest_lengths)
            tet_elems_field.from_numpy(tet_elems)
            solver = make_solver(num_edges)

        vertices = particles.x.to_numpy()[:particles.num_particles]
        if frame_topology_changed:
            surface_faces = tearing.rebuild_surface_faces_from_tets(tet_elems, vertices)
        write_obj(out_dir / f"frame_{frame:04d}.obj", vertices, surface_faces)
        if config.preview_during_simulation and config.preview_every > 0 and frame % config.preview_every == 0:
            render_animation_video.render_preview_frame(config, frame)
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
