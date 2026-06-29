import time
import taichi as ti
import numpy as np

import simulation_config as sim_config


class Renderer:
    def __init__(self, particles, surface_faces, title="Position-Based Dynamics", fps=120):
        self.particles = particles
        self.frame_dt  = 1.0 / fps
        self._last_time = time.time()

        # GGUI 윈도우 & 씬
        self.res = (1024, 768)
        self.window = ti.ui.Window(title, res=self.res)
        self.canvas = self.window.get_canvas()
        self.scene  = self.window.get_scene()
        self.camera = ti.ui.Camera()

        # 카메라 초기 위치
        self.camera.position(0.0, 0.15, 0.5)
        self.camera.lookat(0.0, 0.1, 0.0)
        self.camera.up(0.0, 1.0, 0.0)

        # 표면 삼각형 인덱스를 Taichi field에 업로드
        self.surface_faces = None
        self.indices = None
        self.marked_indices = None
        self.marked_vertices = set()
        self.set_surface_faces(surface_faces)

        # 바닥 메시 (사각형 평면, 삼각형 2개) — y는 set_floor_y로 갱신
        self._floor_xz = np.array([
            [-1.0, -1.0],
            [ 1.0, -1.0],
            [ 1.0,  1.0],
            [-1.0,  1.0],
        ], dtype=np.float32)
        floor_faces = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)

        self.floor_verts = ti.Vector.field(3, dtype=ti.f32, shape=4)
        self.floor_indices = ti.field(dtype=ti.i32, shape=6)
        self.floor_indices.from_numpy(floor_faces)
        self.set_floor_y(sim_config.FLOOR_Y)

    def set_floor_y(self, y: float):
        verts = np.empty((4, 3), dtype=np.float32)
        verts[:, 0] = self._floor_xz[:, 0]
        verts[:, 1] = y
        verts[:, 2] = self._floor_xz[:, 1]
        self.floor_verts.from_numpy(verts)

    def set_surface_faces(self, surface_faces):
        self.surface_faces = np.asarray(surface_faces, dtype=np.int32)
        self._update_mesh_indices()

    def set_marked_vertices(self, marked_vertices):
        self.marked_vertices = set(int(i) for i in marked_vertices)
        self._update_mesh_indices()

    def _make_index_field(self, faces):
        if len(faces) == 0:
            return None
        flat_faces = faces.flatten().astype(np.int32)
        indices = ti.field(dtype=ti.i32, shape=len(flat_faces))
        indices.from_numpy(flat_faces)
        return indices

    def _update_mesh_indices(self):
        self.indices = None
        self.marked_indices = None
        if self.surface_faces is None:
            return

        if not self.marked_vertices:
            self.indices = self._make_index_field(self.surface_faces)
            return

        marked_mask = np.array(
            [any(int(v) in self.marked_vertices for v in face) for face in self.surface_faces],
            dtype=bool,
        )
        self.indices = self._make_index_field(self.surface_faces[~marked_mask])
        self.marked_indices = self._make_index_field(self.surface_faces[marked_mask])

    def get_camera_state(self):
        return (np.array(self.camera.curr_position, dtype=np.float32),
                np.array(self.camera.curr_lookat,   dtype=np.float32),
                np.array(self.camera.curr_up,       dtype=np.float32))

    @property
    def aspect(self):
        return self.res[0] / self.res[1]

    def render(self, sel_label: str = "None", floor_move: bool = False):
        # 60 FPS 제한
        now = time.time()
        elapsed = now - self._last_time
        if elapsed < self.frame_dt:
            time.sleep(self.frame_dt - elapsed)
        self._last_time = time.time()

        # 마우스 우클릭 드래그로 카메라 회전 (LMB는 picking에 사용)
        self.camera.track_user_inputs(self.window, movement_speed=0.01, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)

        # 조명
        self.scene.ambient_light((0.7, 0.7, 0.7))
        self.scene.point_light(pos=(0.5, 1.0, 0.5), color=(1.0, 1.0, 1.0))

        # 바닥 렌더링
        self.scene.mesh(self.floor_verts, indices=self.floor_indices, color=(0.3, 0.3, 0.3))

        # 메시 렌더링
        if self.indices is not None:
            self.scene.mesh(self.particles.x, indices=self.indices, color=(0.5, 0.5, 0.5))
        if self.marked_indices is not None:
            self.scene.mesh(self.particles.x, indices=self.marked_indices, color=(1.0, 0.0, 0.0))

        # 좌상단 상태 GUI
        gui = self.window.get_gui()
        with gui.sub_window("Status", 0.01, 0.01, 0.25, 0.12):
            gui.text(f"Selected: {sel_label}")
            gui.text(f"Floor move (F): {'ON' if floor_move else 'OFF'}")

        self.canvas.scene(self.scene)
        self.window.show()

    def is_running(self):
        if self.window.is_pressed(ti.ui.ESCAPE):
            return False
        return self.window.running
