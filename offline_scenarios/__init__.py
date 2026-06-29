from copy import deepcopy

from . import bunny_pull, cloth_box_bunny, cloth_common, cloth_cover_bunny, cloth_metal_drop, moving_bunny, pull_tearing, rollers, spot_pull


_SCENARIOS = {
    "bunny_pull": bunny_pull,
    "pull_tearing": bunny_pull,
    "pull_only": bunny_pull,
    "spot_pull": spot_pull,
    "roller": rollers,
    "rollers": rollers,
    "cloth_metal_drop": cloth_metal_drop,
    "cloth_cover_bunny": cloth_cover_bunny,
    "cloth_box_bunny": cloth_box_bunny,
    "moving_bunny": moving_bunny,
}
_SCENARIO_CONFIG_OVERRIDES = {
    "pull_only": {
        "output": "bunny_pull_only",
        "enable_tearing": False,
    },
}
_KIND_MODULES = {
    "pull": pull_tearing,
    "rollers": rollers,
    "cloth": cloth_common,
}


def _deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _module_for_name(name):
    try:
        return _SCENARIOS[name]
    except KeyError as exc:
        names = ", ".join(sorted(_SCENARIOS))
        raise ValueError(f"Unsupported offline scenario: {name}. Available: {names}") from exc


def get_scenario_config(name):
    module = _module_for_name(name)
    config = deepcopy(getattr(module, "SCENARIO_CONFIG", {}))
    _deep_update(config, deepcopy(_SCENARIO_CONFIG_OVERRIDES.get(name, {})))
    return config


def _module_for_config(config):
    return _module_for_name(config.scenario)


def _module_for_state(scenario_state):
    try:
        return _KIND_MODULES[scenario_state["kind"]]
    except KeyError as exc:
        raise ValueError(f"Unsupported scenario state: {scenario_state['kind']}") from exc


def prepare_scenario(config, scene_or_positions, surface_faces_or_bbox_diag, bbox_diag=None):
    module = _module_for_config(config)
    if hasattr(scene_or_positions, "positions"):
        scene = scene_or_positions
        bbox = surface_faces_or_bbox_diag if bbox_diag is None else bbox_diag
        if getattr(module, "PREPARE_ACCEPTS_SCENE_DATA", False):
            return module.prepare(config, scene, bbox)
        return module.prepare(config, scene.positions, scene.surface_faces, bbox)
    return module.prepare(config, scene_or_positions, surface_faces_or_bbox_diag, bbox_diag)


def apply_scenario_frame(config, scenario_state, particles, frame):
    return _module_for_state(scenario_state).apply_frame(config, scenario_state, particles, frame)


def clear_scenario(config, scenario_state, particles):
    return _module_for_state(scenario_state).clear(config, scenario_state, particles)


def apply_scenario_substep(config, scenario_state, particles, dt, stage):
    fn = getattr(_module_for_state(scenario_state), "apply_substep", None)
    if fn is not None:
        return fn(config, scenario_state, particles, dt, stage)
    return None


def register_scenario_solver_constraints(config, scenario_state, particles, solver):
    fn = getattr(_module_for_state(scenario_state), "register_solver_constraints", None)
    if fn is not None:
        return fn(config, scenario_state, particles, solver)
    return None


def scenario_metadata(config, scenario_state):
    fn = getattr(_module_for_state(scenario_state), "metadata", None)
    if fn is None:
        return {}
    return fn(config, scenario_state)


def scenario_log_lines(config, scenario_state):
    fn = getattr(_module_for_state(scenario_state), "log_lines", None)
    if fn is None:
        return []
    return fn(config, scenario_state)
