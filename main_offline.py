import argparse
from dataclasses import dataclass

import simulation_config as sim_config
import mark_animation_obj_sequence
import offline_animation_export
import render_animation_video
import export_animation_usd
import export_animation_blend_viewer
from offline_scenarios import get_scenario_config


# ── Pipeline switches ────────────────────────────────────────────────────────
RUN_SIMULATION = True
RUN_MARKING = True
RUN_VIDEO = True
RUN_USD = True
RUN_BLEND_VIEWER = False

# ── Scenario ─────────────────────────────────────────────────────────────────
SCENARIO = "bunny_pull"

# ── Output ───────────────────────────────────────────────────────────────────
# None이면 선택된 시나리오의 SCENARIO_CONFIG 값을 사용.
OUTPUT = None
OUTPUT_PATH = None
OVERWRITE = True

# ── Runtime / render backend ─────────────────────────────────────────────────
ARCH = "gpu"  # "gpu" | "cpu"
BLENDER_BIN = "/home/kyumin/Development/blender-5.1.2-linux-x64/blender"
FFMPEG_BIN = "ffmpeg"

# ── Simulation length override ───────────────────────────────────────────────
# None이면 선택된 시나리오의 SCENARIO_CONFIG 값을 사용.
FPS = None
FRAMES = None
DURATION_SECONDS = None

# ── Offline solver defaults ──────────────────────────────────────────────────
SOLVER_ITERS = 100
JACOBI_RELAXATION = 0.75
DEFAULT_MIN_PATCH_SIZE = 8

# ── Simulation progress preview ──────────────────────────────────────────────
PREVIEW_DURING_SIMULATION = True
PREVIEW_EVERY = 50

# ── Video render ─────────────────────────────────────────────────────────────
RESOLUTION_X = 1280
RESOLUTION_Y = 720
VIDEO_FPS = None  # None이면 simulation metadata의 FPS 사용.
SAMPLES = 16
START_FRAME = 0
END_FRAME = None
FRAME_STEP = 1
BUNNY_COLOR = (0.5, 0.5, 0.5, 1.0)
FLOOR_COLOR = (0.3, 0.3, 0.3, 1.0)
BACKGROUND_COLOR = (0.02, 0.02, 0.02)
POINT_LIGHT_ENERGY = 45.0

# ── Capacity ─────────────────────────────────────────────────────────────────
MAX_PARTICLE_FACTOR = 4.0
MAX_EDGE_FACTOR = 4.0

CLI_ARGS = None


@dataclass(frozen=True)
class OfflineConfig:
    output: str
    output_path: str
    obj_dir: str
    marked_obj_dir: str
    png_dir: str
    frame_preview_dir: str
    mp4_path: str
    preview_during_simulation: bool
    preview_every: int
    usd_path: str
    blend_path: str
    overwrite: bool
    scenario: str
    scenario_options: dict
    enable_tearing: bool
    asset: str
    fixed_anchor_indices: tuple[int, int] | None
    seed: int
    patch_radius: float | None
    auto_patch_radius_ratio: float
    random_anchor_min_distance_ratio: float
    min_patch_size: int
    pull_distance: float | None
    auto_pull_distance_ratio: float
    pull_horizontal: bool
    pull_down_angle_degrees: float | None
    render_rollers: bool
    roller_center: tuple[float, float, float]
    roller_pair_axis: str
    roller_center_height_ratio: float
    bunny_start_height_ratio: float
    roller_radius: float
    roller_gap: float
    roller_length: float
    roller_angular_speed: float
    roller_upper_omega_sign: float
    roller_lower_omega_sign: float
    roller_contact_margin: float
    roller_normal_restitution: float
    roller_align_bunny_to_center: bool
    roller_align_bunny_face_to_center: bool
    bunny_start_offset: tuple[float, float, float]
    bunny_initial_velocity: tuple[float, float, float]
    roller_segments: int
    roller_stripe_count: int
    roller_stripe_twist: float
    roller_yellow: tuple[float, float, float, float]
    roller_black: tuple[float, float, float, float]
    frames: int
    fps: int
    substeps: int
    solver_iters: int
    jacobi_relaxation: float
    arch: str
    use_gravity: bool
    gravity_scale: float
    damping: float
    floor_y: float
    mu_s: float
    mu_k: float
    e_restitution: float
    k_distance: float
    k_volume: float
    max_particle_factor: float
    max_edge_factor: float
    tear_ratio: float
    max_tears_per_call: int
    blender_bin: str
    ffmpeg_bin: str
    resolution_x: int
    resolution_y: int
    video_fps: int | None
    samples: int
    start_frame: int
    end_frame: int | None
    frame_step: int
    camera_bounds_mode: str
    camera_target_mode: str
    camera_distance_scale: float
    camera_height_scale: float
    camera_side_scale: float
    camera_orbit_degrees: float | None
    camera_lens: float
    camera_bounds_padding: float
    bunny_color: tuple[float, float, float, float]
    floor_color: tuple[float, float, float, float]
    background_color: tuple[float, float, float]
    point_light_energy: float


def _tuple(value):
    return tuple(value)


def build_config(args=None):
    args = CLI_ARGS if args is None else args
    scenario_name = args.scenario if args is not None and args.scenario is not None else SCENARIO
    scenario_config = get_scenario_config(scenario_name)
    camera = scenario_config.get("camera", {})

    scenario_overridden = args is not None and args.scenario is not None
    output_override = args.output if args is not None and args.output is not None else (None if scenario_overridden else OUTPUT)
    output_path_override = args.output_path if args is not None and args.output_path is not None else (None if scenario_overridden else OUTPUT_PATH)
    output = output_override if output_override is not None else scenario_config.get("output", scenario_config.get("run_name", scenario_name))
    output_path = output_path_override if output_path_override is not None else f"outputs/{output}"
    scenario_fps = int(scenario_config.get("fps", 30))
    fps = int(FPS if FPS is not None else scenario_fps)
    if DURATION_SECONDS is not None:
        frames = max(1, int(round(float(DURATION_SECONDS) * fps)))
    else:
        frames = int(FRAMES if FRAMES is not None else scenario_config.get("frames", 200))

    obj_dir = f"{output_path}/obj"
    marked_obj_dir = f"{output_path}/obj_marked"
    png_dir = f"{output_path}/png"
    frame_preview_dir = f"{output_path}/frame_preview"
    mp4_path = f"{output_path}/animation.mp4"
    usd_path = f"{output_path}/animation.usd"
    blend_path = f"{output_path}/animation_viewer.blend"

    return OfflineConfig(
        output=output,
        output_path=output_path,
        obj_dir=obj_dir,
        marked_obj_dir=marked_obj_dir,
        png_dir=png_dir,
        frame_preview_dir=frame_preview_dir,
        mp4_path=mp4_path,
        preview_during_simulation=PREVIEW_DURING_SIMULATION,
        preview_every=PREVIEW_EVERY,
        usd_path=usd_path,
        blend_path=blend_path,
        overwrite=OVERWRITE,
        scenario=scenario_name,
        scenario_options=scenario_config,
        enable_tearing=bool(scenario_config.get("enable_tearing", False)),
        asset=scenario_config.get("asset", sim_config.ASSET),
        fixed_anchor_indices=scenario_config.get("fixed_anchor_indices"),
        seed=int(scenario_config.get("seed", 0)),
        patch_radius=scenario_config.get("patch_radius"),
        auto_patch_radius_ratio=float(scenario_config.get("auto_patch_radius_ratio", 0.08)),
        random_anchor_min_distance_ratio=float(scenario_config.get("random_anchor_min_distance_ratio", 0.45)),
        min_patch_size=int(scenario_config.get("min_patch_size", DEFAULT_MIN_PATCH_SIZE)),
        pull_distance=scenario_config.get("pull_distance"),
        auto_pull_distance_ratio=float(scenario_config.get("auto_pull_distance_ratio", 0.45)),
        pull_horizontal=bool(scenario_config.get("pull_horizontal", True)),
        pull_down_angle_degrees=scenario_config.get("pull_down_angle_degrees"),
        render_rollers=bool(scenario_config.get("render_rollers", False)),
        roller_center=_tuple(scenario_config.get("roller_center", (0.0, 0.0, 0.0))),
        roller_pair_axis=str(scenario_config.get("roller_pair_axis", "y")),
        roller_center_height_ratio=float(scenario_config.get("roller_center_height_ratio", 0.0)),
        bunny_start_height_ratio=float(scenario_config.get("bunny_start_height_ratio", 0.0)),
        roller_radius=float(scenario_config.get("roller_radius", 0.0)),
        roller_gap=float(scenario_config.get("roller_gap", 0.0)),
        roller_length=float(scenario_config.get("roller_length", 0.0)),
        roller_angular_speed=float(scenario_config.get("roller_angular_speed", 0.0)),
        roller_upper_omega_sign=float(scenario_config.get("roller_upper_omega_sign", -1.0)),
        roller_lower_omega_sign=float(scenario_config.get("roller_lower_omega_sign", 1.0)),
        roller_contact_margin=float(scenario_config.get("roller_contact_margin", 0.0)),
        roller_normal_restitution=float(scenario_config.get("roller_normal_restitution", 0.0)),
        roller_align_bunny_to_center=bool(scenario_config.get("roller_align_bunny_to_center", False)),
        roller_align_bunny_face_to_center=bool(scenario_config.get("roller_align_bunny_face_to_center", False)),
        bunny_start_offset=_tuple(scenario_config.get("bunny_start_offset", (0.0, 0.0, 0.0))),
        bunny_initial_velocity=_tuple(scenario_config.get("bunny_initial_velocity", (0.0, 0.0, 0.0))),
        roller_segments=int(scenario_config.get("roller_segments", 32)),
        roller_stripe_count=int(scenario_config.get("roller_stripe_count", 8)),
        roller_stripe_twist=float(scenario_config.get("roller_stripe_twist", 0.0)),
        roller_yellow=_tuple(scenario_config.get("roller_yellow", (1.0, 0.82, 0.05, 1.0))),
        roller_black=_tuple(scenario_config.get("roller_black", (0.02, 0.02, 0.02, 1.0))),
        frames=frames,
        fps=fps,
        substeps=int(scenario_config.get("substeps", 10)),
        solver_iters=int(scenario_config.get("solver_iters", SOLVER_ITERS)),
        jacobi_relaxation=float(scenario_config.get("jacobi_relaxation", JACOBI_RELAXATION)),
        arch=ARCH,
        use_gravity=bool(scenario_config.get("use_gravity", sim_config.USE_GRAVITY)),
        gravity_scale=float(scenario_config.get("gravity_scale", 1.0)),
        damping=float(scenario_config.get("damping", sim_config.DAMPING)),
        floor_y=float(scenario_config.get("floor_y", sim_config.FLOOR_Y)),
        mu_s=float(scenario_config.get("mu_s", sim_config.MU_S)),
        mu_k=float(scenario_config.get("mu_k", sim_config.MU_K)),
        e_restitution=float(scenario_config.get("e_restitution", sim_config.E_RESTITUTION)),
        k_distance=float(scenario_config.get("k_distance", sim_config.K_DISTANCE)),
        k_volume=float(scenario_config.get("k_volume", sim_config.K_VOLUME)),
        max_particle_factor=float(scenario_config.get("max_particle_factor", MAX_PARTICLE_FACTOR)),
        max_edge_factor=float(scenario_config.get("max_edge_factor", MAX_EDGE_FACTOR)),
        tear_ratio=float(scenario_config.get("tear_ratio", sim_config.TEAR_RATIO)),
        max_tears_per_call=int(scenario_config.get("max_tears_per_call", sim_config.MAX_TEARS_PER_CALL)),
        blender_bin=BLENDER_BIN,
        ffmpeg_bin=FFMPEG_BIN,
        resolution_x=RESOLUTION_X,
        resolution_y=RESOLUTION_Y,
        video_fps=VIDEO_FPS,
        samples=SAMPLES,
        start_frame=START_FRAME,
        end_frame=END_FRAME,
        frame_step=FRAME_STEP,
        camera_bounds_mode=camera.get("bounds_mode", "first_frame"),
        camera_target_mode=camera.get("target_mode", "bunny"),
        camera_distance_scale=float(camera.get("distance_scale", 5.0)),
        camera_height_scale=float(camera.get("height_scale", 0.28)),
        camera_side_scale=float(camera.get("side_scale", 0.35)),
        camera_orbit_degrees=None if camera.get("orbit_degrees") is None else float(camera.get("orbit_degrees")),
        camera_lens=float(camera.get("lens", 55)),
        camera_bounds_padding=float(camera.get("bounds_padding", 1.0)),
        bunny_color=BUNNY_COLOR,
        floor_color=FLOOR_COLOR,
        background_color=BACKGROUND_COLOR,
        point_light_energy=POINT_LIGHT_ENERGY,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run the offline PBD animation pipeline.")
    parser.add_argument("--scenario", choices=("bunny_pull", "spot_pull", "pull_tearing", "pull_only", "roller", "rollers", "cloth_metal_drop", "cloth_cover_bunny", "cloth_box_bunny", "moving_bunny"), help="Offline scenario to run.")
    parser.add_argument("--output", help="Output folder name under outputs/ unless --output-path is set.")
    parser.add_argument("--output-path", help="Explicit output directory path.")
    return parser.parse_args()


def main():
    global CLI_ARGS
    CLI_ARGS = parse_args()
    config = build_config(CLI_ARGS)
    print(f"[main-offline] output: {config.output}")
    print(f"[main-offline] scenario: {config.scenario}")
    print(f"[main-offline] output path: {config.output_path}")

    # 1. 시뮬레이션: scenario 설정으로 PBD를 돌리고 OBJ frame sequence를 저장
    if RUN_SIMULATION:
        offline_animation_export.export_animation_sequence(config)

    # 2. 마킹: anchor/특수 표시가 필요한 경우 별도 OBJ sequence 생성
    if RUN_MARKING:
        mark_animation_obj_sequence.mark_animation_obj_sequence(config)

    # 3. 렌더링: OBJ sequence를 PNG frame과 MP4 영상으로 변환
    if RUN_VIDEO:
        render_animation_video.render_animation_video(config)

    # 4. USD export: Blender/usdview에서 frame 단위로 볼 수 있는 scene export
    if RUN_USD:
        export_animation_usd.export_animation_usd(config)

    # 5. Blender viewer export: 필요할 때만 .blend viewer 파일 생성
    if RUN_BLEND_VIEWER:
        export_animation_blend_viewer.export_animation_blend_viewer(config)

    print("[main-offline] done")


if __name__ == "__main__":
    main()
