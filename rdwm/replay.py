"""Replay a held-out Lance window inside the live simulator.

The numbers printed here are the measured error distributions. Thresholds
only separate JPEG / float32 noise from a restore that is not tracking
the recorded transition. They are not widened after the fact.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from rdwm.data.io import (
    action_at,
    check_episode_boundary,
    column,
    holdout_starts,
    open_lance,
    summarize_schema,
    window_actions_finite,
)
from rdwm.data.manifest import ENVS, HOLDOUT
from rdwm.envs.sessions import dataset_frame, open_session


def _dist(values) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {'n': 0, 'median': None, 'max': None, 'p95': None, 'min': None}
    return {
        'n': int(array.size),
        'min': float(array.min()),
        'median': float(np.median(array)),
        'p95': float(np.quantile(array, 0.95)),
        'max': float(array.max()),
        'mean': float(array.mean()),
    }


def _fmt(summary: dict) -> str:
    if not summary or summary.get('median') is None:
        return '-'
    return f"{summary['median']:.3g} / {summary['max']:.3g}"


def _l2(left, right) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(left - right))


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if (
        frame.ndim == 3
        and frame.shape[0] in (1, 3, 4)
        and frame.shape[-1] not in (1, 3, 4)
    ):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.dtype != np.uint8:
        peak = float(np.nanmax(frame)) if frame.size else 0.0
        if peak <= 1.0:
            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _mae(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.mean(
            np.abs(left.astype(np.float32) - right.astype(np.float32))
        )
    )


def _jpeg_roundtrip_mae(frame: np.ndarray, quality: int = 95) -> float:
    buffer = BytesIO()
    Image.fromarray(frame).save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    decoded = np.array(Image.open(buffer).convert('RGB'))
    return _mae(frame, decoded)


def _shift_mae(left: np.ndarray, right: np.ndarray, radius: int = 2):
    """Smallest MAE over integer translations, for camera-offset checks."""
    best = (_mae(left, right), 0, 0)
    height, width = left.shape[:2]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            y0 = max(dy, 0)
            x0 = max(dx, 0)
            y1 = height + min(dy, 0)
            x1 = width + min(dx, 0)
            shifted_left = left[y0:y1, x0:x1]
            shifted_right = right[
                max(-dy, 0) : height + min(-dy, 0),
                max(-dx, 0) : width + min(-dx, 0),
            ]
            error = _mae(shifted_left, shifted_right)
            if error < best[0]:
                best = (error, dy, dx)
    return best


def _dataset_state(session, episode, index) -> np.ndarray:
    name = session.name
    if name in ('pusht', 'pointmaze'):
        return session._state_from_row(episode, index)
    if name == 'tworoom':
        return _row_vec(episode, index, 'pos_agent')
    qpos = _row_vec(episode, index, 'qpos')
    qvel = _row_vec(episode, index, 'qvel')
    return np.concatenate([qpos, qvel])


def _row_vec(episode, index, name) -> np.ndarray:
    from rdwm.data.io import column as _column

    return np.asarray(_column(episode, name)[index], dtype=np.float64).reshape(
        -1
    )


def _finite_start(episode, preferred: int, horizon: int) -> int | None:
    length = int(column(episode, 'action').shape[0])
    order = [preferred] + [
        step for step in range(0, length - horizon) if step != preferred
    ]
    for start in order:
        if window_actions_finite(episode, start, horizon):
            return start
    return None


def _roll(session, episode, start: int, actions: list[np.ndarray]):
    session.restore(episode, start)
    restore_error = _l2(
        session.state(), _dataset_state(session, episode, start)
    )
    errors = []
    for offset, action in enumerate(actions):
        session.step(action)
        errors.append(
            _l2(
                session.state(),
                _dataset_state(session, episode, start + offset + 1),
            )
        )
    return restore_error, errors


def _image_pair(session, episode, index: int) -> tuple[float, float, tuple]:
    rendered = _normalize_frame(session.render())
    stored = _normalize_frame(dataset_frame(episode, index))
    if rendered.shape != stored.shape:
        return float('inf'), float('nan'), rendered.shape
    return (
        _mae(rendered, stored),
        _jpeg_roundtrip_mae(stored),
        _shift_mae(rendered, stored),
    )


def _lock_pointmaze_view(session, episode, index: int) -> dict:
    cameras = [None] + [cam_id for cam_id, _name in session.camera_names]
    trials = []
    for visible in (True, False):
        for camera_id in cameras:
            session.set_view(show_goal_marker=visible, camera_id=camera_id)
            session.restore(episode, index)
            mae, jpeg_mae, shift = _image_pair(session, episode, index)
            trials.append(
                {
                    'show_goal_marker': visible,
                    'camera_id': camera_id,
                    'mae': mae,
                    'jpeg_mae': jpeg_mae,
                    'shift': [shift[1], shift[2]] if isinstance(shift, tuple) else None,
                }
            )
    finite = [trial for trial in trials if np.isfinite(trial['mae'])]
    finite.sort(key=lambda trial: trial['mae'])
    chosen = finite[0] if finite else trials[0]
    # Accept a view only when it lands near JPEG error. Otherwise keep the
    # default camera and let the image check fail in the open.
    jpeg_ref = chosen.get('jpeg_mae') or 1.0
    if chosen['mae'] <= max(8.0, 5.0 * float(jpeg_ref)):
        session.set_view(
            show_goal_marker=chosen['show_goal_marker'],
            camera_id=chosen['camera_id'],
        )
        chosen = {**chosen, 'accepted': True}
    else:
        session.set_view(show_goal_marker=False, camera_id=None)
        chosen = {**chosen, 'accepted': False}
    return {'chosen': chosen, 'trials': trials}


def _lock_antmaze(spec, episode, index: int) -> tuple[object, dict]:
    candidates = ('medium', 'large', 'giant', 'teleport', 'arena')
    trials = []
    best = None
    best_session = None
    for maze_type in candidates:
        session = open_session('antmaze', spec, maze_type=maze_type)
        try:
            session.restore(episode, index)
            mae, jpeg_mae, shift = _image_pair(session, episode, index)
        except Exception as exc:  # noqa: BLE001 - record and keep going
            mae, jpeg_mae, shift = float('inf'), float('nan'), str(exc)
        trial = {
            'maze_type': maze_type,
            'mae': mae,
            'jpeg_mae': jpeg_mae,
            'shift': shift[1:] if isinstance(shift, tuple) else shift,
        }
        trials.append(trial)
        if best is None or mae < best['mae']:
            if best_session is not None:
                best_session.close()
            best = trial
            best_session = session
        else:
            session.close()
    jpeg_ref = best.get('jpeg_mae') or 1.0
    best = {
        **best,
        'accepted': bool(
            np.isfinite(best['mae'])
            and best['mae'] <= max(8.0, 5.0 * float(jpeg_ref))
        ),
    }
    return best_session, {'chosen': best, 'trials': trials}


def _goal_index(start: int, length: int) -> int:
    return int(min(start + HOLDOUT['goal_offset'], length - 1))


def _check_episode(dataset, episode_id: int) -> list[str]:
    return check_episode_boundary(dataset, episode_id)


def _judge(result: dict) -> str:
    if result.get('error'):
        return 'FAIL'
    if result['boundary_problems']:
        return 'FAIL'
    if result['action_dim_dataset'] != result['action_dim_runtime']:
        return 'FAIL'
    one = result['one_step']['median']
    incoming = result['incoming_one_step']['median']
    five = result['five_step']['median']
    wrong = result['wrong_action_five_step']['median']
    travel = result['state_delta_one']['median']
    if one is None or five is None:
        return 'FAIL'
    if incoming is not None and one > incoming:
        return 'FAIL'
    # 1-step must track the recorded transition, not a full state jump.
    if travel is not None and one > 0.25 * travel + 1e-4:
        return 'FAIL'
    if wrong is not None and five > 0.5 * wrong:
        return 'FAIL'
    image = result['image_mae']['median']
    jpeg = result['jpeg_roundtrip_mae']['median']
    if image is None or not np.isfinite(image):
        return 'FAIL'
    if jpeg is not None and image > max(8.0, 5.0 * jpeg):
        return 'FAIL'
    image_n = result['image_mae']['n'] or 1
    if result['image_shift_counts'].get('nonzero', 0) > image_n / 2:
        return 'FAIL'
    if not result['goal_positive_all_true']:
        return 'FAIL'
    if result.get('button_mismatches', 0):
        return 'FAIL'
    if result.get('layout_error_max') not in (None, 0) and result[
        'layout_error_max'
    ] > 1e-3:
        return 'FAIL'
    return 'PASS'


def run_env(name: str, *, num_starts: int | None = None) -> dict:
    spec = ENVS[name]
    horizon = HOLDOUT['horizon']
    result = {
        'env': name,
        'lance_uri': spec['lance_uri'],
        'hf_revision': spec['hf_revision'],
        'error': None,
    }
    dataset = None
    session = None
    try:
        dataset = open_lance(spec['lance_uri'])
        schema = summarize_schema(dataset)
        result['schema'] = {
            'num_episodes': schema['num_episodes'],
            'length_min': schema['length_min'],
            'length_max': schema['length_max'],
            'image_hw': schema['image_hw'],
            'episode_columns': schema['episode_columns'],
            'terminal_action_all_nan': schema['columns']
            .get('action', {})
            .get('terminal_action_all_nan'),
            'column_names': sorted(schema['columns']),
        }
        action_info = schema['columns'].get('action', {})
        action_shape = action_info.get('shape') or []
        result['action_dim_dataset'] = (
            int(action_shape[-1]) if len(action_shape) >= 2 else None
        )
        starts = holdout_starts(
            dataset.lengths, num_starts=num_starts, horizon=horizon
        )
        first_episode, first_start = starts[0]
        first = dataset.load_episode(int(first_episode))
        frame = _normalize_frame(dataset_frame(first, 0))
        result['dataset_frame_hw'] = [int(frame.shape[0]), int(frame.shape[1])]

        view_info = None
        if name == 'pointmaze':
            height, width = frame.shape[:2]
            session = open_session(
                name, spec, width=int(width), height=int(height)
            )
            preferred = _finite_start(first, first_start, horizon)
            session.restore(first, 0 if preferred is None else preferred)
            view_info = _lock_pointmaze_view(
                session, first, 0 if preferred is None else preferred
            )
        elif name == 'scene':
            height, width = frame.shape[:2]
            scene_spec = dict(spec)
            scene_spec['env_kwargs'] = dict(spec['env_kwargs'])
            scene_spec['env_kwargs']['width'] = int(width)
            scene_spec['env_kwargs']['height'] = int(height)
            session = open_session(name, scene_spec)
        elif name == 'antmaze':
            preferred = _finite_start(first, first_start, horizon)
            session, view_info = _lock_antmaze(
                spec, first, 0 if preferred is None else preferred
            )
        else:
            session = open_session(name, spec)
        result['view'] = view_info
        result['action_dim_runtime'] = int(session.action_dim)
        result['action_low'] = session.action_low.tolist()
        result['action_high'] = session.action_high.tolist()

        restore_errs = []
        one_errs = []
        five_errs = []
        incoming_errs = []
        wrong_errs = []
        deltas = []
        travels = []
        image_maes = []
        jpeg_maes = []
        shift_nonzero = 0
        boundary = []
        goal_positive = []
        goal_negative_false = 0
        goal_negative_checked = 0
        episode_goal = []
        action_values = []
        used = []
        layout_errors = []

        cache = {int(first_episode): first}
        for episode_id, preferred in starts:
            episode = cache.get(int(episode_id))
            if episode is None:
                episode = dataset.load_episode(int(episode_id))
                cache[int(episode_id)] = episode
            problems = _check_episode(dataset, int(episode_id))
            boundary.extend(problems)
            start = _finite_start(episode, preferred, horizon)
            if start is None:
                boundary.append(
                    f'episode {episode_id} has no finite {horizon}-step window'
                )
                continue
            length = int(column(episode, 'action').shape[0])
            actions = [action_at(episode, start + offset) for offset in range(horizon)]
            action_values.append(np.stack(actions))
            restore_error, errors = _roll(session, episode, start, actions)
            if getattr(session, 'layout_error', None) is not None:
                layout_errors.append(float(session.layout_error))
            restore_errs.append(restore_error)
            one_errs.append(errors[0])
            five_errs.append(errors[-1])
            deltas.append(
                _l2(
                    _dataset_state(session, episode, start),
                    _dataset_state(session, episode, start + 1),
                )
            )
            travels.append(
                _l2(
                    _dataset_state(session, episode, start),
                    _dataset_state(session, episode, start + horizon),
                )
            )
            # Incoming hypothesis: the action stored at t+1 produced the
            # step, which would mean the table still needs a shift.
            # PointMaze must already be outgoing, so this stays diagnostic.
            if start + horizon < length and np.isfinite(
                action_at(episode, start + 1)
            ).all():
                _restore, incoming = _roll(
                    session, episode, start, [action_at(episode, start + 1)]
                )
                incoming_errs.append(incoming[0])
            _restore, wrong = _roll(
                session, episode, start, [-action for action in actions]
            )
            wrong_errs.append(wrong[-1])

            session.restore(episode, start)
            mae, jpeg_mae, shift = _image_pair(session, episode, start)
            image_maes.append(mae)
            jpeg_maes.append(jpeg_mae)
            if isinstance(shift, tuple) and (shift[1] or shift[2]):
                if shift[0] < 0.7 * mae:
                    shift_nonzero += 1

            goal_at = _goal_index(start, length)
            session.restore(episode, start)
            session.set_goal(episode, goal_at)
            goal_negative_checked += 1
            if not session.success():
                goal_negative_false += 1
            session.park_on_goal(episode, goal_at)
            goal_positive.append(bool(session.success()))
            if hasattr(session, 'episode_goal_matches'):
                episode_goal.append(
                    bool(session.episode_goal_matches(episode, start))
                )
            used.append({'episode': int(episode_id), 'start': int(start)})

        result['starts'] = used
        result['boundary_problems'] = boundary
        result['restore_error'] = _dist(restore_errs)
        result['one_step'] = _dist(one_errs)
        result['five_step'] = _dist(five_errs)
        result['incoming_one_step'] = _dist(incoming_errs)
        result['wrong_action_five_step'] = _dist(wrong_errs)
        result['state_delta_one'] = _dist(deltas)
        result['state_travel_five'] = _dist(travels)
        result['image_mae'] = _dist(image_maes)
        result['jpeg_roundtrip_mae'] = _dist(jpeg_maes)
        result['image_shift_counts'] = {
            'nonzero': shift_nonzero,
        }
        result['goal_positive'] = goal_positive
        result['goal_positive_all_true'] = bool(goal_positive) and all(
            goal_positive
        )
        result['goal_negative_checked'] = goal_negative_checked
        result['goal_negative_saw_false'] = goal_negative_false > 0
        result['episode_goal_positive'] = episode_goal
        result['button_mismatches'] = int(
            getattr(session, 'button_mismatch', 0)
        )
        result['layout_error_max'] = (
            None if not layout_errors else float(max(layout_errors))
        )
        if action_values:
            stacked = np.concatenate(action_values, axis=0)
            finite = stacked[np.isfinite(stacked).all(axis=1)]
            result['action_sample_min'] = finite.min(axis=0).tolist()
            result['action_sample_max'] = finite.max(axis=0).tolist()
        result['status'] = _judge(result)
    except Exception as exc:  # noqa: BLE001 - one env must not hide the rest
        result['error'] = f'{type(exc).__name__}: {exc}'
        result['status'] = 'FAIL'
        result.setdefault('boundary_problems', [])
    finally:
        if session is not None:
            session.close()
    return result


def run_gate(names: list[str] | None = None, *, num_starts: int | None = None):
    selected = list(ENVS) if names is None else names
    return [run_env(name, num_starts=num_starts) for name in selected]


def format_table(results: list[dict]) -> str:
    header = (
        'Env | Action dim | 1-step error | 5-step error | '
        'Image restore | Goal restore | Success check | Status'
    )
    lines = [header, '--- | --- | --- | --- | --- | --- | --- | ---']
    for result in results:
        if result.get('error') and 'one_step' not in result:
            lines.append(
                f"{result['env']} | - | - | - | - | - | - | FAIL"
            )
            continue
        dim = result.get('action_dim_runtime')
        goal = 'yes' if result.get('goal_positive_all_true') else 'no'
        success = goal
        if result.get('button_mismatches'):
            success = 'button mismatch'
        image = _fmt(result.get('image_mae') or {})
        lines.append(
            f"{result['env']} | {dim} | {_fmt(result.get('one_step') or {})} | "
            f"{_fmt(result.get('five_step') or {})} | {image} | {goal} | "
            f"{success} | {result.get('status')}"
        )
    return '\n'.join(lines)
