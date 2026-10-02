"""Remote Lance reads and deterministic holdout indices."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from rdwm.data.manifest import ACTION_ALIGNMENT, HOLDOUT


def open_lance(uri: str):
    """Open a Hub Lance table with the Stable-WM reader."""
    import stable_worldmodel as swm

    kwargs: dict[str, Any] = {}
    if uri.startswith('hf://') and os.environ.get('HF_TOKEN'):
        kwargs['connect_kwargs'] = {
            'storage_options': {
                'region': os.environ.get('AWS_DEFAULT_REGION', 'us-east-1'),
                'virtual_hosted_style_request': 'true',
                'token': os.environ['HF_TOKEN'],
            }
        }
    return swm.data.load_dataset(uri, num_steps=1, frameskip=1, **kwargs)


def as_np(value) -> np.ndarray:
    import torch

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def frame_hwc(pixels, index: int) -> np.ndarray:
    frame = as_np(pixels[index])
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
        if frame.shape[-1] not in (1, 3, 4):
            frame = np.transpose(frame, (1, 2, 0))
    return np.ascontiguousarray(frame)


def column(episode: dict, *names: str) -> np.ndarray:
    for name in names:
        if name in episode:
            return as_np(episode[name])
    raise KeyError(
        f'None of {names} are in the episode. '
        f'Columns: {sorted(episode)}'
    )


def summarize_schema(dataset) -> dict[str, Any]:
    lengths = np.asarray(dataset.lengths)
    sample = dataset.load_episode(0)
    columns = {}
    image_hw = None
    for name, value in sample.items():
        arr = as_np(value)
        info: dict[str, Any] = {
            'shape': list(arr.shape),
            'dtype': str(arr.dtype),
        }
        if name == 'pixels' or name.startswith('pixels'):
            frame = frame_hwc(value, 0)
            info['frame_hw'] = [int(frame.shape[0]), int(frame.shape[1])]
            if name == 'pixels':
                image_hw = info['frame_hw']
        elif arr.dtype.kind in 'fc':
            flat = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr
            finite = flat[np.isfinite(flat)]
            info['finite_min'] = (
                float(finite.min()) if finite.size else None
            )
            info['finite_max'] = (
                float(finite.max()) if finite.size else None
            )
            if name == 'action' and arr.ndim >= 2:
                last = arr[-1]
                info['terminal_action_finite'] = bool(
                    np.isfinite(last).all()
                )
                info['terminal_action_all_nan'] = bool(
                    np.isnan(last).all()
                )
        columns[name] = info
    schema_names = list(getattr(dataset, '_schema_names', sample.keys()))
    return {
        'num_episodes': int(len(lengths)),
        'length_min': int(lengths.min()),
        'length_max': int(lengths.max()),
        'length_mean': float(lengths.mean()),
        'episode_columns': list(dataset.episode_column_names),
        'schema_names': schema_names,
        'image_hw': image_hw,
        'columns': columns,
        'action_alignment_expected': ACTION_ALIGNMENT,
    }


def check_episode_boundary(dataset, episode_id: int) -> list[str]:
    """Read ``episode_idx`` / ``step_idx`` from the Lance table.

    The Stable-WM reader keeps those index columns out of ``load_episode``.
    """
    import lancedb

    names = set(getattr(dataset, '_schema_names', []))
    needed = ['episode_idx', 'step_idx']
    missing = [name for name in needed if name not in names]
    if missing:
        return [f'missing index columns: {missing}']
    start = int(dataset.offsets[episode_id])
    length = int(dataset.lengths[episode_id])
    table = lancedb.connect(
        dataset.uri, **dataset.connect_kwargs
    ).open_table(dataset.table_name)
    taken = table.to_lance().take(
        list(range(start, start + length)), columns=needed
    )
    episodes = np.asarray(taken.column('episode_idx').to_pylist())
    steps = np.rint(np.asarray(taken.column('step_idx').to_pylist())).astype(
        np.int64
    )
    problems = []
    if np.unique(episodes).size != 1:
        problems.append('episode_idx changes inside the episode')
    if not np.array_equal(steps, np.arange(length)):
        problems.append('step_idx is not 0..T-1')
    return problems


def holdout_starts(
    lengths: np.ndarray,
    *,
    num_starts: int | None = None,
    horizon: int | None = None,
    fraction: float | None = None,
) -> list[tuple[int, int]]:
    """Deterministic held-out (episode, start) pairs inside one episode.

    The last ``fraction`` of episodes are the holdout. Starts need
    ``horizon`` finite actions. When the stored terminal action is NaN,
    callers still pass a horizon that stops before that row because the
    state at ``start + horizon`` must exist and the actions
    ``start .. start+horizon-1`` must be finite. This function only
    enforces the index window; finite-action filtering happens per sample.
    """
    num_starts = HOLDOUT['num_starts'] if num_starts is None else num_starts
    horizon = HOLDOUT['horizon'] if horizon is None else horizon
    fraction = HOLDOUT['fraction'] if fraction is None else fraction
    lengths = np.asarray(lengths)
    eligible = [
        i for i, length in enumerate(lengths) if int(length) >= horizon + 2
    ]
    if len(eligible) < num_starts:
        raise RuntimeError(
            f'Only {len(eligible)} episodes are long enough for '
            f'a {horizon}-step replay; need {num_starts}.'
        )
    cut = int(len(eligible) * (1.0 - fraction))
    pool = eligible[cut:]
    if len(pool) < num_starts:
        pool = eligible
    pick_at = np.linspace(0, len(pool) - 1, num_starts, dtype=int)
    starts: list[tuple[int, int]] = []
    for index in pick_at:
        episode = int(pool[int(index)])
        length = int(lengths[episode])
        latest = length - horizon - 1
        start = min(max(length // 3, 0), latest)
        starts.append((episode, int(start)))
    return starts


def action_at(episode: dict, index: int) -> np.ndarray:
    action = column(episode, 'action')
    return np.asarray(action[index], dtype=np.float64).reshape(-1)


def window_actions_finite(episode: dict, start: int, horizon: int) -> bool:
    for offset in range(horizon):
        if not np.isfinite(action_at(episode, start + offset)).all():
            return False
    return True
