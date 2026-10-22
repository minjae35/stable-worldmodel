"""Label-free temporal windows for the Stage-3 G3 history experiment."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from stable_worldmodel.wm.spatial.data import pack_blocked_action
from stable_worldmodel.wm.spatial.g2 import (
    RAW_ACTION_DIMS,
    TRAINING_COLUMNS,
    _take,
    assert_label_free_columns,
    decode_rgb,
)


def history_clip_rows(
    clip: dict, frameskip: int, history: int
) -> tuple[list[int], int, list[list[int]]]:
    """Return observed rows, target row, and causal action rows per observation."""
    frameskip = int(frameskip)
    history = int(history)
    if frameskip < 1 or history < 1:
        raise ValueError('frameskip and history must be positive')
    anchor = int(clip['row'])
    start = int(clip['start'])
    required_past = (history - 1) * frameskip
    if start < required_past:
        raise ValueError(
            f'clip start {start} lacks {required_past} rows of episode history'
        )
    observation_rows = [
        anchor - offset * frameskip for offset in range(history - 1, -1, -1)
    ]
    target = anchor + frameskip
    action_rows = [list(range(row, row + frameskip)) for row in observation_rows]
    if observation_rows[-1] != anchor or action_rows[-1][-1] != target - 1:
        raise RuntimeError('history window is not aligned to the target anchor')
    flattened = [row for block in action_rows for row in block]
    if target in observation_rows or target in flattened or max(flattened) >= target:
        raise RuntimeError('history window contains future information')
    if start + frameskip >= int(clip['length']):
        raise ValueError('target is at or beyond the terminal row')
    return observation_rows, target, action_rows


def common_history_items(items: list[dict], frameskip: int, history: int) -> list[dict]:
    required_past = (int(history) - 1) * int(frameskip)
    selected = [dict(item) for item in items if int(item['start']) >= required_past]
    if not selected:
        raise ValueError('no common history-valid target anchors')
    for item in selected:
        history_clip_rows(item, frameskip, history)
    return selected


def items_sha256(items: list[dict]) -> str:
    payload = json.dumps(items, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def load_label_free_history_clips(
    dataset,
    clips: list[dict],
    env: str,
    frameskip: int,
    history: int,
) -> dict:
    """Load history observations/actions while requesting no state supervision."""
    columns = assert_label_free_columns(TRAINING_COLUMNS)
    rows: list[int] = []
    plans = []
    for clip in clips:
        observations, target, action_blocks = history_clip_rows(
            clip, frameskip, history
        )
        plans.append((clip, observations, target, action_blocks))
        rows.extend(observations)
        rows.append(target)
        rows.extend(row for block in action_blocks for row in block)
    _, by_row = _take(dataset, rows, columns)

    images_context = []
    images_target = []
    blocked_actions = []
    for clip, observations, target, action_blocks in plans:
        expected_episode = int(clip['episode'])
        used = [*observations, target, *(row for block in action_blocks for row in block)]
        observed_episodes = {int(by_row[row]['episode_idx']) for row in used}
        if observed_episodes != {expected_episode}:
            raise RuntimeError(
                f'{env} history clip crosses episode: expected {expected_episode}, '
                f'got {observed_episodes}'
            )
        action_history = []
        for rows_for_block in action_blocks:
            raw = np.stack([
                np.asarray(by_row[row]['action'], dtype=np.float32).reshape(-1)
                for row in rows_for_block
            ])
            expected = (frameskip, RAW_ACTION_DIMS[env])
            if raw.shape != expected:
                raise RuntimeError(f'{env} action shape {raw.shape}, expected {expected}')
            action_history.append(pack_blocked_action(raw, 1, frameskip)[0])
        images_context.append(
            np.stack([decode_rgb(by_row[row]['pixels']) for row in observations])
        )
        images_target.append(decode_rgb(by_row[target]['pixels']))
        blocked_actions.append(np.stack(action_history))
    return {
        'images_context': np.stack(images_context),
        'images_target': np.stack(images_target),
        'actions': np.stack(blocked_actions).astype(np.float32),
        'episode': np.asarray([int(clip['episode']) for clip in clips], dtype=np.int64),
        'start': np.asarray([int(clip['start']) for clip in clips], dtype=np.int64),
        'items': [dict(clip) for clip in clips],
    }


def normalize_action_history(actions: np.ndarray, normalizer: dict) -> torch.Tensor:
    actions = np.asarray(actions, dtype=np.float32)
    mean = normalizer['mean'].cpu().numpy().reshape(1, 1, -1)
    std = normalizer['std'].cpu().numpy().reshape(1, 1, -1)
    value = (actions - mean) / std
    if not np.isfinite(value).all():
        raise RuntimeError('non-finite normalized history action')
    return torch.from_numpy(np.ascontiguousarray(value))


def history_model_batch(model, bundle, indices, env_index: int, device):
    take = torch.as_tensor(indices, dtype=torch.long)
    context = bundle['context'].index_select(0, take).to(device)
    target = bundle['target'].index_select(0, take).to(device)[:, None]
    actions = bundle['actions'].index_select(0, take).to(device)
    env_id = torch.full((len(take),), env_index, device=device, dtype=torch.long)
    prediction = model(context, actions, env_id)[:, -1:]
    return context, actions, env_id, target, prediction


def select_history(bundle: dict, history: int) -> dict:
    """Select the most recent H observations/actions without changing targets."""
    history = int(history)
    if history < 1 or history > bundle['context'].shape[1]:
        raise ValueError('requested history is outside the cached history range')
    selected = dict(bundle)
    selected['context'] = bundle['context'][:, -history:]
    selected['actions'] = bundle['actions'][:, -history:]
    return selected
