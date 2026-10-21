"""Clip packing and episode splits for the spatial world model.

Action layout matches LanceDataset: image columns are strided by frameskip,
actions are not, then reshaped to ``(num_steps, frameskip * raw_dim)``.
``action[:, t]`` is the block executed at observation ``t`` that produces
observation ``t + 1``. The block at the target frame is not an input.
"""

from __future__ import annotations

import numpy as np

# Native per-step action widths in the Lance tables, before frameskip packing.
RAW_ACTION_DIM = {'pusht': 2, 'tworoom': 2, 'cube': 5}
DEFAULT_FRAMESKIP = 5


def blocked_action_dim(raw_dim: int, frameskip: int) -> int:
    if raw_dim < 1 or frameskip < 1:
        raise ValueError('raw_dim and frameskip must be positive')
    return int(raw_dim * frameskip)


def pack_blocked_action(raw, num_steps: int, frameskip: int):
    """Pack a contiguous raw action window the same way the dataset does.

    ``raw`` is ``(num_steps * frameskip, raw_dim)`` in row-major order.
    The result is ``(num_steps, frameskip * raw_dim)``.
    """
    raw = np.asarray(raw)
    if raw.ndim != 2:
        raise ValueError(f'raw action must be (span, dim), got {raw.shape}')
    span, raw_dim = raw.shape
    if span != num_steps * frameskip:
        raise ValueError(
            f'expected {num_steps * frameskip} raw steps, got {span}'
        )
    return np.ascontiguousarray(raw.reshape(num_steps, frameskip * raw_dim))


def one_step_inputs(tokens, actions):
    """Slice a two-frame window into a one-step prediction problem.

    ``tokens[:, 1]`` is the label. ``actions[:, 1]`` is the action that
    happens at the target frame and is dropped. Only ``actions[:, 0]`` is
    returned as the model input.
    """
    if tokens.shape[1] < 2 or actions.shape[1] < 2:
        raise ValueError('one-step slice needs two frames and two action blocks')
    if tokens.shape[0] != actions.shape[0]:
        raise ValueError('batch size mismatch between tokens and actions')
    context = tokens[:, :1]
    action_t = actions[:, :1]
    target = tokens[:, 1:2]
    if hasattr(target, 'detach'):
        target = target.detach()
    return context, action_t, target


def select_episode_ids(lengths, n_total: int, min_length: int, seed: int):
    """Spread ``n_total`` episodes across the dataset, longest-eligible only."""
    lengths = np.asarray(lengths)
    eligible = np.flatnonzero(lengths >= min_length)
    if len(eligible) < n_total:
        raise ValueError(
            f'need {n_total} episodes with length >= {min_length}, '
            f'found {len(eligible)}'
        )
    pick = np.linspace(0, len(eligible) - 1, n_total, dtype=int)
    chosen = eligible[pick]
    _, first = np.unique(chosen, return_index=True)
    chosen = chosen[np.sort(first)]
    if len(chosen) < n_total:
        rng = np.random.default_rng(seed)
        rest = np.setdiff1d(eligible, chosen, assume_unique=False)
        extra = rng.choice(rest, size=n_total - len(chosen), replace=False)
        chosen = np.concatenate([chosen, extra])
    return chosen.astype(np.int64)


def split_episodes(episodes, n_train: int, n_test: int, seed: int):
    """Episode-disjoint split. The caller must not pass overlapping ids."""
    episodes = np.asarray(episodes)
    if len(np.unique(episodes)) != len(episodes):
        raise ValueError('episode ids must be unique before the split')
    if len(episodes) < n_train + n_test:
        raise ValueError('not enough episodes for the requested split')
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(episodes))
    train = np.asarray(episodes[perm[:n_train]], dtype=np.int64)
    test = np.asarray(
        episodes[perm[n_train : n_train + n_test]], dtype=np.int64
    )
    if set(train.tolist()) & set(test.tolist()):
        raise RuntimeError('train and test episodes overlap')
    return train, test


def clip_start(length: int, span: int) -> int:
    """A legal dataset clip start. ``span`` is ``num_steps * frameskip``."""
    if length < span:
        raise ValueError(f'episode length {length} is shorter than span {span}')
    return int((length - span) // 3)


def candidate_starts(length: int, span: int, k: int = 16):
    last = int(length) - int(span)
    if last < 0:
        raise ValueError('episode is shorter than one clip')
    return np.unique(np.round(np.linspace(0, last, k)).astype(np.int64))


def fit_action_normalizer(actions):
    """Per-dimension z-score stats. ``actions`` is ``(N, D)``."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[0] < 2:
        raise ValueError('need at least two action rows to fit a normalizer')
    mean = actions.mean(axis=0)
    std = np.maximum(actions.std(axis=0), 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_action_normalizer(actions, mean, std):
    actions = np.asarray(actions, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    return (actions - mean) / std
