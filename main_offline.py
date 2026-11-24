"""Run a PBD simulation and save its surface mesh at each frame."""

import argparse

from offline.config import build_config
from offline.scenarios import SCENARIO_NAMES
from offline.simulation import export_animation_sequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIO_NAMES,
                        help="Simulation preset. spot_pull requires a local assets/spot.obj.")
    parser.add_argument("--output", help="Run folder name under outputs/.")
    parser.add_argument("--output-path", help="Explicit run directory.")
    parser.add_argument("--frames", type=int, help="Override the preset frame count.")
    parser.add_argument("--arch", choices=("cpu", "gpu"), help="Taichi backend.")
    args = parser.parse_args()
    if args.frames is not None and args.frames < 1:
        parser.error("--frames must be positive")

    config = build_config(args)
    print(f"[offline] scenario={config.scenario} output={config.obj_dir}")
    export_animation_sequence(config)


if __name__ == "__main__":
    main()
