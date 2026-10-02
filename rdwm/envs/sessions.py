"""Per-environment restore, replay, goal, and success hooks.

Each session talks to an existing simulator. PointMaze is the only
environment constructed outside Stable-WM, via ``rdwm.envs.pointmaze``.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from rdwm.data.io import column, frame_hwc

# Registers swm/* ids.
import stable_worldmodel as swm  # noqa: F401


def _l2(left, right) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError(f'state shape {left.shape} vs {right.shape}')
    return float(np.linalg.norm(left - right))


def _row(episode: dict, index: int, *names: str) -> np.ndarray:
    return np.asarray(column(episode, *names)[index], dtype=np.float64)


def discrete_buttons(raw) -> np.ndarray:
    """Map Lance ``button_states`` onto SceneEnv's discrete pair.

    Length 2 is already ``[button_0, button_1]``. Length 4 or shape
    ``(2, 2)`` is one-hot per button. Anything else is rejected.
    """
    arr = np.asarray(raw, dtype=np.float64)
    if arr.shape == (2, 2):
        if not np.allclose(arr.sum(axis=1), 1.0):
            raise ValueError(f'button one-hot rows do not sum to 1: {arr}')
        return np.argmax(arr, axis=1).astype(np.int64)
    flat = arr.reshape(-1)
    if flat.shape == (2,):
        rounded = np.rint(flat)
        if np.max(np.abs(flat - rounded)) > 1e-5:
            raise ValueError(f'button_states are not discrete: {flat}')
        values = set(int(v) for v in rounded.tolist())
        if not values.issubset({0, 1}):
            raise ValueError(f'button_states outside {{0, 1}}: {flat}')
        return rounded.astype(np.int64)
    if flat.shape == (4,):
        return np.array(
            [int(np.argmax(flat[:2])), int(np.argmax(flat[2:]))],
            dtype=np.int64,
        )
    raise ValueError(
        f'button_states shape {arr.shape} is not a discrete pair or one-hot'
    )


def _make(env_id: str, **kwargs):
    return gym.make(env_id, max_episode_steps=1000, **kwargs)


_GYM_WRAPPERS = {
    'TimeLimit',
    'OrderEnforcing',
    'PassiveEnvChecker',
    'Autoreset',
    'HumanRendering',
    'RecordEpisodeStatistics',
}


def _core(env):
    """Drop Gymnasium bookkeeping wrappers, keep domain wrappers.

    ``MazeEnv`` is itself a wrapper. ``env.unwrapped`` skips it and lands on
    the OGBench env, which does not expose the Stable-WM ``set_state``.
    """
    seen = 0
    while type(env).__name__ in _GYM_WRAPPERS and hasattr(env, 'env'):
        env = env.env
        seen += 1
        if seen > 8:
            break
    return env


class PushTSession:
    name = 'pusht'

    def __init__(self, spec: dict):
        self.env = _make(spec['env_id'], **spec['env_kwargs'])
        self.env.reset(seed=0)
        space = self.env.action_space
        self.action_dim = int(np.prod(space.shape))
        self.action_low = np.asarray(space.low, dtype=np.float64).reshape(-1)
        self.action_high = np.asarray(space.high, dtype=np.float64).reshape(-1)

    def close(self) -> None:
        self.env.close()

    def _state_from_row(self, episode, index) -> np.ndarray:
        return _row(episode, index, 'state').reshape(-1)

    def restore(self, episode, index) -> np.ndarray:
        state = self._state_from_row(episode, index)
        if 'goal_pose' in episode:
            pose = _row(episode, index, 'goal_pose').reshape(-1)
            _core(self.env).goal_pose = pose[:3].copy()
        if 'goal_state' in episode:
            _core(self.env)._set_goal_state(
                _row(episode, index, 'goal_state').reshape(-1)
            )
        _core(self.env)._set_state(state)
        return self.state()

    def state(self) -> np.ndarray:
        return np.asarray(_core(self.env)._get_obs(), dtype=np.float64)

    def render(self) -> np.ndarray:
        return np.ascontiguousarray(_core(self.env).render())

    def step(self, action: np.ndarray) -> None:
        self.env.step(np.asarray(action, dtype=np.float32).reshape(-1))

    def set_goal(self, episode, index) -> None:
        self._goal = self._state_from_row(episode, index)
        _core(self.env)._set_goal_state(self._goal)

    def park_on_goal(self, episode, index) -> None:
        self.set_goal(episode, index)
        _core(self.env)._set_state(self._goal)

    def episode_goal_matches(self, episode, index) -> bool | None:
        """Predicate on the stored episode ``goal_state``, if that column exists."""
        if 'goal_state' not in episode:
            return None
        goal = _row(episode, index, 'goal_state').reshape(-1)
        _core(self.env)._set_goal_state(goal)
        _core(self.env)._set_state(goal)
        return self.success()

    def success(self) -> bool:
        ok, _ = _core(self.env).eval_state(
            _core(self.env).goal_state, self.state()
        )
        return bool(ok)


class TwoRoomSession:
    name = 'tworoom'

    def __init__(self, spec: dict):
        self.env = _make(spec['env_id'], **spec['env_kwargs'])
        self.env.reset(seed=0)
        space = self.env.action_space
        self.action_dim = int(np.prod(space.shape))
        self.action_low = np.asarray(space.low, dtype=np.float64).reshape(-1)
        self.action_high = np.asarray(space.high, dtype=np.float64).reshape(-1)
        self.layout_error = None

    def close(self) -> None:
        self.env.close()

    def _apply_layout(self, observation: np.ndarray) -> None:
        obs = np.asarray(observation, dtype=np.float64).reshape(-1)
        if obs.shape[0] < 10:
            raise ValueError(f'TwoRoom observation is {obs.shape}, expected 10')
        wall = float(_core(self.env).WALL_CENTER)
        centers = []
        axis = None
        for xy in obs[4:10].reshape(3, 2):
            if np.allclose(xy, 0.0):
                continue
            if abs(xy[0] - wall) <= 1e-3:
                this_axis = 1
                center = float(xy[1])
            elif abs(xy[1] - wall) <= 1e-3:
                this_axis = 0
                center = float(xy[0])
            else:
                raise RuntimeError(
                    f'door coordinate {xy.tolist()} is not on the center '
                    f'wall at {wall}'
                )
            if axis is None:
                axis = this_axis
            elif axis != this_axis:
                raise RuntimeError('door coordinates disagree on wall axis')
            centers.append(center)
        if not centers or axis is None:
            raise RuntimeError('observation has no door geometry')
        env = _core(self.env)
        current = np.asarray(
            env.variation_space['door']['position'].value, dtype=np.int64
        ).reshape(-1)
        positions = current.copy()
        positions[: len(centers)] = np.rint(centers).astype(np.int64)
        env.variation_space['wall']['axis'].set_value(int(axis))
        env.variation_space['door']['number'].set_value(int(len(centers)))
        env.variation_space['door']['position'].set_value(positions)
        env._cache_params()
        restored = np.asarray(env._get_obs(), dtype=np.float64)
        # Agent and target are written next; compare door slots only.
        self.layout_error = float(
            np.max(np.abs(restored[4:10] - obs[4:10]))
        )

    def restore(self, episode, index) -> np.ndarray:
        obs = _row(episode, index, 'observation')
        agent = _row(episode, index, 'pos_agent').reshape(-1)
        target = _row(episode, index, 'pos_target').reshape(-1)
        self._apply_layout(obs)
        _core(self.env)._set_state(agent)
        _core(self.env)._set_goal_state(target)
        return self.state()

    def state(self) -> np.ndarray:
        pos = _core(self.env).agent_position.detach().cpu().numpy()
        return np.asarray(pos, dtype=np.float64).reshape(-1)

    def render(self) -> np.ndarray:
        return np.ascontiguousarray(_core(self.env).render())

    def step(self, action: np.ndarray) -> None:
        self.env.step(np.asarray(action, dtype=np.float32).reshape(-1))

    def set_goal(self, episode, index) -> None:
        # The predicate target is this row's agent position. Callers pass
        # the future row, so the target is that future XY.
        target = _row(episode, index, 'pos_agent').reshape(-1)
        self._goal = target
        _core(self.env)._set_goal_state(target)

    def park_on_goal(self, episode, index) -> None:
        agent = _row(episode, index, 'pos_agent').reshape(-1)
        self._apply_layout(_row(episode, index, 'observation'))
        _core(self.env)._set_state(agent)
        _core(self.env)._set_goal_state(agent)
        self._goal = agent

    def success(self) -> bool:
        agent = _core(self.env).agent_position
        target = _core(self.env).target_position
        dist = float((agent - target).norm())
        return dist < 16.0


class _ManipSession:
    """Shared qpos/qvel restore for OGBench manipulation envs."""

    def __init__(self, spec: dict):
        self.env = _make(spec['env_id'], **spec['env_kwargs'])
        self.env.reset(seed=0)
        space = self.env.action_space
        self.action_dim = int(np.prod(space.shape))
        self.action_low = np.asarray(space.low, dtype=np.float64).reshape(-1)
        self.action_high = np.asarray(space.high, dtype=np.float64).reshape(-1)
        self.button_mismatch = 0

    def close(self) -> None:
        self.env.close()

    def _q(self, episode, index, kind: str) -> np.ndarray:
        return _row(episode, index, kind).reshape(-1)

    def restore(self, episode, index) -> np.ndarray:
        qpos = self._q(episode, index, 'qpos')
        qvel = self._q(episode, index, 'qvel')
        _core(self.env).set_state(qpos, qvel)
        return self.state()

    def state(self) -> np.ndarray:
        data = _core(self.env)._data
        return np.concatenate(
            [np.asarray(data.qpos, dtype=np.float64), np.asarray(data.qvel, dtype=np.float64)]
        )

    def render(self) -> np.ndarray:
        frame = _core(self.env).render()
        return np.ascontiguousarray(frame)

    def step(self, action: np.ndarray) -> None:
        self.env.step(np.asarray(action, dtype=np.float32).reshape(-1))


class CubeSession(_ManipSession):
    name = 'cube'

    def _block_pose(self, episode, index):
        pos = _row(
            episode,
            index,
            'privileged/block_0_pos',
            'privileged_block_0_pos',
        ).reshape(-1)
        quat_names = (
            'privileged/block_0_quat',
            'privileged_block_0_quat',
        )
        quat = None
        for name in quat_names:
            if name in episode:
                quat = _row(episode, index, name).reshape(-1)
                break
        return pos, quat

    def set_goal(self, episode, index) -> None:
        pos, quat = self._block_pose(episode, index)
        _core(self.env).set_target_pos(0, pos, quat)

    def park_on_goal(self, episode, index) -> None:
        self.restore(episode, index)
        self.set_goal(episode, index)

    def success(self) -> bool:
        flags = _core(self.env)._compute_successes()
        return bool(all(flags))


class SceneSession(_ManipSession):
    name = 'scene'

    def restore(self, episode, index) -> np.ndarray:
        qpos = self._q(episode, index, 'qpos')
        qvel = self._q(episode, index, 'qvel')
        buttons = discrete_buttons(
            _row(episode, index, 'button_states')
        )
        _core(self.env).set_state(
            qpos,
            qvel,
            button_state_0=int(buttons[0]),
            button_state_1=int(buttons[1]),
        )
        got = np.asarray(
            _core(self.env)._cur_button_states, dtype=np.int64
        ).reshape(-1)
        if got.shape != buttons.shape or not np.array_equal(got, buttons):
            self.button_mismatch += 1
        return self.state()

    def _scalar(self, episode, index, *names: str) -> float:
        return float(_row(episode, index, *names).reshape(-1)[0])

    def set_goal(self, episode, index) -> None:
        # This table has no privileged pose columns. object_joint_0 is the
        # free joint at qpos[14:21] (xyz + quat). Drawer and window slides
        # are qpos[23] and qpos[24].
        qpos = self._q(episode, index, 'qpos')
        if qpos.shape[0] < 25:
            raise ValueError(f'scene qpos is {qpos.shape}, expected 25')
        core = _core(self.env)
        core.set_cube_target_pos(0, qpos[14:17], qpos[17:21])
        buttons = discrete_buttons(_row(episode, index, 'button_states'))
        for button_id, value in enumerate(buttons.tolist()):
            core.set_target_button_state(button_id, int(value))
        core.set_target_drawer_pos(float(qpos[23]))
        core.set_target_window_pos(float(qpos[24]))

    def park_on_goal(self, episode, index) -> None:
        self.restore(episode, index)
        self.set_goal(episode, index)

    def success(self) -> bool:
        cubes, buttons, drawer, window = (
            _core(self.env)._compute_successes()
        )
        return bool(all(cubes) and all(buttons) and drawer and window)


class ReacherSession:
    name = 'reacher'

    def __init__(self, spec: dict):
        self.env = _make(spec['env_id'], **spec['env_kwargs'])
        self.env.reset(seed=0)
        space = self.env.action_space
        self.action_dim = int(np.prod(space.shape))
        self.action_low = np.asarray(space.low, dtype=np.float64).reshape(-1)
        self.action_high = np.asarray(space.high, dtype=np.float64).reshape(-1)

    def close(self) -> None:
        self.env.close()

    def restore(self, episode, index) -> np.ndarray:
        qpos = _row(episode, index, 'qpos').reshape(-1)
        qvel = _row(episode, index, 'qvel').reshape(-1)
        _core(self.env).set_state(qpos, qvel)
        return self.state()

    def state(self) -> np.ndarray:
        physics = _core(self.env).env.physics
        return np.concatenate(
            [
                np.asarray(physics.data.qpos, dtype=np.float64),
                np.asarray(physics.data.qvel, dtype=np.float64),
            ]
        )

    def render(self) -> np.ndarray:
        return np.ascontiguousarray(_core(self.env).render())

    def step(self, action: np.ndarray) -> None:
        self.env.step(np.asarray(action, dtype=np.float32).reshape(-1))

    def set_goal(self, episode, index) -> None:
        qpos = _row(episode, index, 'qpos').reshape(-1)
        _core(self.env).set_target_qpos(qpos)
        self._goal_qpos = qpos

    def park_on_goal(self, episode, index) -> None:
        self.set_goal(episode, index)
        qvel = _row(episode, index, 'qvel').reshape(-1)
        _core(self.env).set_state(self._goal_qpos, qvel)

    def success(self) -> bool:
        task = _core(self.env).env.task
        physics = _core(self.env).env.physics
        return task.get_termination(physics) is not None


class PointMazeSession:
    name = 'pointmaze'

    def __init__(self, spec: dict, *, width: int, height: int):
        from rdwm.envs.pointmaze import PointMazeUMaze

        self.sim = PointMazeUMaze(width=width, height=height)
        space = self.sim.action_space
        self.action_dim = int(np.prod(space.shape))
        self.action_low = np.asarray(space.low, dtype=np.float64).reshape(-1)
        self.action_high = np.asarray(space.high, dtype=np.float64).reshape(-1)
        self.camera_names = self.sim.camera_names()

    def close(self) -> None:
        self.sim.close()

    def _state_from_row(self, episode, index) -> np.ndarray:
        if 'state' in episode:
            state = _row(episode, index, 'state').reshape(-1)
            if state.shape == (4,):
                return state
        if 'proprio' in episode:
            state = _row(episode, index, 'proprio').reshape(-1)
            if state.shape == (4,):
                return state
        parts = []
        for name in ('x', 'y', 'vx', 'vy'):
            if name not in episode:
                break
            parts.append(float(_row(episode, index, name).reshape(-1)[0]))
        if len(parts) == 4:
            return np.asarray(parts, dtype=np.float64)
        qpos = _row(episode, index, 'qpos').reshape(-1)
        qvel = _row(episode, index, 'qvel').reshape(-1)
        return np.concatenate([qpos[:2], qvel[:2]])

    def restore(self, episode, index) -> np.ndarray:
        self.sim.restore_state(self._state_from_row(episode, index))
        return self.state()

    def state(self) -> np.ndarray:
        return self.sim.current_state()

    def render(self) -> np.ndarray:
        return self.sim.render()

    def step(self, action: np.ndarray) -> None:
        self.sim.step(action)

    def set_goal(self, episode, index) -> None:
        if 'desired_goal' in episode:
            xy = _row(episode, index, 'desired_goal').reshape(-1)[:2]
        else:
            xy = self._state_from_row(episode, index)[:2]
        self.sim.set_goal_xy(xy)

    def park_on_goal(self, episode, index) -> None:
        self.set_goal(episode, index)
        if 'desired_goal' in episode:
            xy = _row(episode, index, 'desired_goal').reshape(-1)[:2]
        else:
            xy = self._state_from_row(episode, index)[:2]
        self.sim.restore_state(np.array([xy[0], xy[1], 0.0, 0.0]))

    def success(self) -> bool:
        return self.sim.success()

    def set_view(self, *, show_goal_marker: bool, camera_id: int | None):
        self.sim.set_goal_marker_visible(show_goal_marker)
        self.sim.camera_id = camera_id


class AntMazeSession:
    name = 'antmaze'

    def __init__(self, spec: dict, *, maze_type: str):
        kwargs = dict(spec['env_kwargs'])
        kwargs['maze_type'] = maze_type
        self.maze_type = maze_type
        self.env = _make(spec['env_id'], **kwargs)
        # Set this before reset. MazeEnv's variation default is 0, and
        # reset recompiles the model without OGBench's pixel floor
        # encoding. Gym `in` tests sample membership, not key membership.
        maze = _core(self.env)
        encoding = maze.variation_space['floor']['pixel_encoding']
        encoding.set_init_value(1)
        encoding.set_value(1)
        self.env.reset(seed=0)
        space = self.env.action_space
        self.action_dim = int(np.prod(space.shape))
        self.action_low = np.asarray(space.low, dtype=np.float64).reshape(-1)
        self.action_high = np.asarray(space.high, dtype=np.float64).reshape(-1)
        self._inner = _core(self.env).env
        self._goal_tol = self._read_goal_tol()

    def close(self) -> None:
        self.env.close()

    def _read_goal_tol(self) -> float:
        inner = self._inner
        for name in (
            '_goal_tol',
            'goal_tol',
            '_success_threshold',
            'success_threshold',
        ):
            if hasattr(inner, name):
                return float(getattr(inner, name))
        return 0.5

    def restore(self, episode, index) -> np.ndarray:
        qpos = _row(episode, index, 'qpos').reshape(-1)
        qvel = _row(episode, index, 'qvel').reshape(-1)
        _core(self.env).set_state(qpos, qvel)
        return self.state()

    def state(self) -> np.ndarray:
        inner = self._inner
        return np.concatenate(
            [
                np.asarray(inner.data.qpos, dtype=np.float64),
                np.asarray(inner.data.qvel, dtype=np.float64),
            ]
        )

    def render(self) -> np.ndarray:
        frame = self._inner.render()
        if isinstance(frame, dict):
            frame = next(iter(frame.values()))
        return np.ascontiguousarray(frame)

    def step(self, action: np.ndarray) -> None:
        self.env.step(np.asarray(action, dtype=np.float32).reshape(-1))

    def _xy(self, episode, index) -> np.ndarray:
        return _row(episode, index, 'qpos').reshape(-1)[:2]

    def set_goal(self, episode, index) -> None:
        xy = self._xy(episode, index)
        self._goal = np.asarray(xy, dtype=np.float64).copy()
        inner = self._inner
        if hasattr(inner, 'set_target_pos'):
            inner.set_target_pos(xy)
            return
        if hasattr(inner, 'set_goal'):
            inner.set_goal(xy)
            return
        placed = False
        for name in ('_cur_goal_xy', '_goal_xy', 'goal_xy'):
            if hasattr(inner, name):
                setattr(inner, name, xy.copy())
                placed = True
                break
        if not placed:
            raise RuntimeError(
                'OGBench antmaze has no goal setter. Methods: '
                + ', '.join(sorted(dir(inner)))
            )

    def park_on_goal(self, episode, index) -> None:
        self.restore(episode, index)
        self.set_goal(episode, index)

    def success(self) -> bool:
        inner = self._inner
        if hasattr(inner, 'get_xy'):
            xy = np.asarray(inner.get_xy(), dtype=np.float64).reshape(-1)[:2]
        else:
            xy = np.asarray(inner.data.qpos[:2], dtype=np.float64)
        return bool(np.linalg.norm(xy - self._goal) <= self._goal_tol)


SESSIONS = {
    'pusht': PushTSession,
    'tworoom': TwoRoomSession,
    'cube': CubeSession,
    'reacher': ReacherSession,
    'scene': SceneSession,
}


def open_session(name: str, spec: dict, **kwargs):
    if name == 'pointmaze':
        return PointMazeSession(spec, **kwargs)
    if name == 'antmaze':
        return AntMazeSession(spec, **kwargs)
    return SESSIONS[name](spec)


def dataset_frame(episode: dict, index: int) -> np.ndarray:
    return frame_hwc(episode['pixels'], index)
