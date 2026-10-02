"""Thin adapter around gymnasium-robotics PointMaze U-Maze.

Stable-WM does not ship this environment. The adapter only exposes
state restore, rendering, and the native 0.45 XY success check. It does
not shift actions.
"""

from __future__ import annotations

import gymnasium as gym
import gymnasium_robotics
import mujoco
import numpy as np

gym.register_envs(gymnasium_robotics)

SUCCESS_RADIUS = 0.45


class PointMazeUMaze:
    """LP2 U-Maze wrapper. State is ``[x, y, vx, vy]``."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        show_goal_marker: bool = True,
        camera_id: int | None = None,
    ):
        self.width = int(width)
        self.height = int(height)
        self.camera_id = camera_id
        self.show_goal_marker = show_goal_marker
        self.env = gym.make(
            'PointMaze_UMaze-v3',
            render_mode='rgb_array',
            width=self.width,
            height=self.height,
            continuing_task=True,
            reset_target=False,
            max_episode_steps=1000,
        )
        self.env.reset(seed=0)
        self._target_site = self._site_id('target')
        self.set_goal_marker_visible(show_goal_marker)
        self._goal = np.zeros(2, dtype=np.float64)

    def close(self) -> None:
        self.env.close()

    @property
    def _point(self):
        return self.env.unwrapped.point_env

    def _site_id(self, name: str) -> int:
        return int(self._point.model.site(name).id)

    @property
    def action_space(self):
        return self.env.action_space

    def camera_names(self) -> list[tuple[int, str]]:
        model = self._point.model
        names = []
        for index in range(model.ncam):
            names.append((index, model.camera(index).name))
        return names

    def set_goal_marker_visible(self, visible: bool) -> None:
        rgba = self._point.model.site_rgba[self._target_site]
        rgba[3] = 0.7 if visible else 0.0
        self.show_goal_marker = visible

    def restore_state(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        if state.shape != (4,):
            raise ValueError(f'PointMaze state must be (4,), got {state.shape}')
        point = self._point
        qpos = point.data.qpos.copy()
        qvel = point.data.qvel.copy()
        qpos[:2] = state[:2]
        qvel[:2] = state[2:4]
        point.set_state(qpos, qvel)
        mujoco.mj_forward(point.model, point.data)

    def current_state(self) -> np.ndarray:
        point = self._point
        return np.concatenate(
            [point.data.qpos[:2], point.data.qvel[:2]]
        ).astype(np.float64)

    def set_goal_xy(self, xy: np.ndarray) -> None:
        xy = np.asarray(xy, dtype=np.float64).reshape(2)
        self._goal = xy.copy()
        self.env.unwrapped.goal = xy.copy()
        site = self._point.model.site_pos[self._target_site]
        site[0] = xy[0]
        site[1] = xy[1]
        mujoco.mj_forward(self._point.model, self._point.data)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        return self.env.step(action)

    def _apply_topdown_camera(self) -> None:
        """Match the stored frames: top-down, azimuth 180, distance 8.8.

        The gymnasium-robotics default is an oblique view (elevation -45,
        azimuth 90). That view is about MAE 40 against this dataset.
        """
        renderer = self._point.mujoco_renderer
        if renderer.viewer is None:
            self.env.render()
        cam = renderer.viewer.cam
        cam.azimuth = 180.0
        cam.elevation = -90.0
        cam.distance = 8.8
        cam.lookat[:] = np.array([0.0, 0.0, 0.0])

    def render(self) -> np.ndarray:
        point = self._point
        if self.camera_id is not None and hasattr(point, 'camera_id'):
            point.camera_id = self.camera_id
        self._apply_topdown_camera()
        frame = self.env.render()
        if frame is None:
            raise RuntimeError('PointMaze render() returned None')
        return np.ascontiguousarray(frame)

    def success(self) -> bool:
        xy = self.current_state()[:2]
        return bool(np.linalg.norm(xy - self._goal) <= SUCCESS_RADIUS)
