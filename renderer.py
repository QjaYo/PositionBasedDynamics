import time
import taichi as ti
import numpy as np

from scene import FLOOR_Y


class Renderer:
    def __init__(self, particles, surface_faces, title="Position-Based Dynamics", fps=120):
        self.particles = particles
        self.frame_dt  = 1.0 / fps
        self._last_time = time.time()

        # GGUI 윈도우 & 씬
        self.window = ti.ui.Window(title, res=(1024, 768))
        self.canvas = self.window.get_canvas()
        self.scene  = self.window.get_scene()
        self.camera = ti.ui.Camera()

        # 카메라 초기 위치
        self.camera.position(0.0, 0.15, 0.5)
        self.camera.lookat(0.0, 0.1, 0.0)
        self.camera.up(0.0, 1.0, 0.0)

        # 표면 삼각형 인덱스를 Taichi field에 업로드
        self.indices = ti.field(dtype=ti.i32, shape=len(surface_faces) * 3)
        self.indices.from_numpy(surface_faces.flatten().astype(np.int32))

        # 바닥 메시 (FLOOR_Y 높이의 사각형 평면, 삼각형 2개)
        floor_verts = np.array([
            [-1.0, FLOOR_Y, -1.0],
            [ 1.0, FLOOR_Y, -1.0],
            [ 1.0, FLOOR_Y,  1.0],
            [-1.0, FLOOR_Y,  1.0],
        ], dtype=np.float32)
        floor_faces = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)

        self.floor_verts = ti.Vector.field(3, dtype=ti.f32, shape=4)
        self.floor_verts.from_numpy(floor_verts)
        self.floor_indices = ti.field(dtype=ti.i32, shape=6)
        self.floor_indices.from_numpy(floor_faces)

    def render(self):
        # 60 FPS 제한
        now = time.time()
        elapsed = now - self._last_time
        if elapsed < self.frame_dt:
            time.sleep(self.frame_dt - elapsed)
        self._last_time = time.time()

        # 마우스 좌클릭 드래그로 카메라 회전
        self.camera.track_user_inputs(self.window, movement_speed=0.01, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)

        # 조명
        self.scene.ambient_light((0.7, 0.7, 0.7))
        self.scene.point_light(pos=(0.5, 1.0, 0.5), color=(1.0, 1.0, 1.0))

        # 바닥 렌더링
        self.scene.mesh(self.floor_verts, indices=self.floor_indices, color=(0.3, 0.3, 0.3))

        # 메시 렌더링
        self.scene.mesh(self.particles.x, indices=self.indices, color=(0.5, 0.5, 0.5))

        self.canvas.scene(self.scene)
        self.window.show()

    def is_running(self):
        if self.window.is_pressed(ti.ui.ESCAPE):
            return False
        return self.window.running
