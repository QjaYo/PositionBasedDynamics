from pathlib import Path

from main_offline import build_config
import render_animation_video as video


# ── Single-frame camera preview config ───────────────────────────────────────
FRAME = 0


def main():
    config = build_config()
    obj_dir = Path(config.marked_obj_dir)
    png_dir = Path(config.frame_preview_dir)
    fps = video.read_fps(obj_dir, config.video_fps)
    roller_settings = video.read_roller_settings(config, obj_dir)

    blender_bin = video.resolve_executable(config.blender_bin)
    video.prepare_png_dir(png_dir, config.overwrite)

    blender_config = {
        "obj_dir": str(obj_dir.resolve()),
        "png_dir": str(png_dir.resolve()),
        "resolution_x": config.resolution_x,
        "resolution_y": config.resolution_y,
        "fps": fps,
        "samples": config.samples,
        "start_frame": FRAME,
        "end_frame": FRAME,
        "frame_step": 1,
        "camera_distance_scale": config.camera_distance_scale,
        "camera_height_scale": config.camera_height_scale,
        "camera_side_scale": config.camera_side_scale,
        "camera_lens": config.camera_lens,
        "camera_bounds_mode": config.camera_bounds_mode,
        "camera_target_mode": config.camera_target_mode,
        "camera_bounds_padding": config.camera_bounds_padding,
        "bunny_color": config.bunny_color,
        "floor_color": config.floor_color,
        "background_color": config.background_color,
        "point_light_energy": config.point_light_energy,
        "floor_y": config.floor_y,
        "rollers": config.render_rollers,
        "roller_center": roller_settings["center"],
        "roller_pair_axis": roller_settings["pair_axis"],
        "roller_radius": config.roller_radius,
        "roller_gap": config.roller_gap,
        "roller_length": config.roller_length,
        "roller_angular_speed": config.roller_angular_speed,
        "roller_segments": config.roller_segments,
        "roller_stripe_count": config.roller_stripe_count,
        "roller_stripe_twist": config.roller_stripe_twist,
        "roller_yellow": config.roller_yellow,
        "roller_black": config.roller_black,
    }
    script_path = video.write_blender_script(png_dir, blender_config)
    video.run_blender(blender_bin, script_path)
    print(f"[frame-preview] wrote {png_dir / 'frame_0000.png'}")


if __name__ == "__main__":
    main()
