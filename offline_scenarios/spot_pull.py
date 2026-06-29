from .pull_tearing import *


SCENARIO_CONFIG = {
    "output": "spot_pull",
    "enable_tearing": True,
    "asset": "assets/spot.obj",
    "use_gravity": True,
    "fixed_anchor_indices": (1453, 80),  # spot upper/back-side z-axis anchors after tetgen
    "tear_ratio": 1.2,
    "max_tears_per_call": 1000,
    "seed": 7,
    "patch_radius": 0.015,
    "auto_patch_radius_ratio": 0.08,
    "random_anchor_min_distance_ratio": 0.45,
    "min_patch_size": 8,
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
        "distance_scale": -4.0,
        "height_scale": 0.22,
        "side_scale": 1.6,
        "lens": 55,
        "bounds_padding": 1.0,
    },
}
