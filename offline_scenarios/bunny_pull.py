from .pull_tearing import *


PATCH_RADIUS = 0.015
MIN_PATCH_SIZE = 8


SCENARIO_CONFIG = {
    "output": "bunny_pull_tearing",
    "enable_tearing": True,
    "asset": "assets/bunny.obj",
    "use_gravity": True,
    "fixed_anchor_indices": (1208, 1167),  # bunny back1 / back2
    "tear_ratio": 1.2,
    "max_tears_per_call": 1000,
    "seed": 7,
    "patch_radius": PATCH_RADIUS,
    "auto_patch_radius_ratio": 0.08,
    "random_anchor_min_distance_ratio": 0.45,
    "min_patch_size": MIN_PATCH_SIZE,
    "pull_distance": 0.50,
    "auto_pull_distance_ratio": 0.45,
    "pull_horizontal": True,
    "pull_down_angle_degrees": None,
    "frames": 200,
    "fps": 30,
    "substeps": 10,
    "camera": {
        "bounds_mode": "first_frame",
        "target_mode": "bunny",
        "distance_scale": 5.0,
        "height_scale": 0.28,
        "side_scale": 0.35,
        "lens": 55,
        "bounds_padding": 1.0,
    },
}
