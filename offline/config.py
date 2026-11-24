from dataclasses import dataclass

import simulation_config as sim_config
from .scenarios import get_scenario_config


# ── Scenario ─────────────────────────────────────────────────────────────────
SCENARIO = "bunny_pull"

# ── Output ───────────────────────────────────────────────────────────────────
# None이면 선택된 시나리오의 SCENARIO_CONFIG 값을 사용.
OUTPUT = None
OUTPUT_PATH = None
OVERWRITE = True

ARCH = "gpu"  # "gpu" | "cpu"

# ── Simulation length override ───────────────────────────────────────────────
# None이면 선택된 시나리오의 SCENARIO_CONFIG 값을 사용.
FPS = None
FRAMES = None
DURATION_SECONDS = None

# ── Offline solver defaults ──────────────────────────────────────────────────
SOLVER_ITERS = 100
JACOBI_RELAXATION = 0.75
DEFAULT_MIN_PATCH_SIZE = 8

# ── Capacity ─────────────────────────────────────────────────────────────────
MAX_PARTICLE_FACTOR = 4.0
MAX_EDGE_FACTOR = 4.0


@dataclass(frozen=True)
class OfflineConfig:
    output: str
    output_path: str
    obj_dir: str
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


def build_config(args=None):
    scenario_name = args.scenario if args is not None and args.scenario is not None else SCENARIO
    scenario_config = get_scenario_config(scenario_name)

    scenario_overridden = args is not None and args.scenario is not None
    output_override = args.output if args is not None and args.output is not None else (None if scenario_overridden else OUTPUT)
    output_path_override = args.output_path if args is not None and args.output_path is not None else (None if scenario_overridden else OUTPUT_PATH)
    output = output_override if output_override is not None else scenario_config.get("output", scenario_config.get("run_name", scenario_name))
    output_path = output_path_override if output_path_override is not None else f"outputs/{output}"
    scenario_fps = int(scenario_config.get("fps", 30))
    fps = int(FPS if FPS is not None else scenario_fps)
    frame_override = getattr(args, "frames", None)
    if frame_override is not None:
        frames = frame_override
    elif DURATION_SECONDS is not None:
        frames = max(1, int(round(float(DURATION_SECONDS) * fps)))
    else:
        frames = int(FRAMES if FRAMES is not None else scenario_config.get("frames", 200))

    obj_dir = f"{output_path}/obj"

    return OfflineConfig(
        output=output,
        output_path=output_path,
        obj_dir=obj_dir,
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
        roller_center=tuple(scenario_config.get("roller_center", (0.0, 0.0, 0.0))),
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
        bunny_start_offset=tuple(scenario_config.get("bunny_start_offset", (0.0, 0.0, 0.0))),
        bunny_initial_velocity=tuple(scenario_config.get("bunny_initial_velocity", (0.0, 0.0, 0.0))),
        frames=frames,
        fps=fps,
        substeps=int(scenario_config.get("substeps", 10)),
        solver_iters=int(scenario_config.get("solver_iters", SOLVER_ITERS)),
        jacobi_relaxation=float(scenario_config.get("jacobi_relaxation", JACOBI_RELAXATION)),
        arch=getattr(args, "arch", None) or ARCH,
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
    )
