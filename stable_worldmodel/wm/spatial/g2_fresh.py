"""Label-free action normalization for the Stage-3 G2 fresh-init study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from stable_worldmodel.wm.spatial.data import fit_action_normalizer
from stable_worldmodel.wm.spatial.g2 import ENV_NAMES, RAW_ACTION_DIMS


def canonical_hash(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def train_episode_row_plan(manifest: dict, env: str) -> list[tuple[int, int, int]]:
    """Return ``(episode, first_row, length)`` using label-free clip metadata."""
    plans: dict[int, tuple[int, int]] = {}
    for clip in manifest[env]['s0']['train']:
        episode = int(clip['episode'])
        candidate = (int(clip['row']) - int(clip['start']), int(clip['length']))
        if episode in plans and plans[episode] != candidate:
            raise ValueError(f'inconsistent row plan for {env} episode {episode}')
        plans[episode] = candidate
    expected = {int(value) for value in manifest[env]['split']['train']}
    if set(plans) != expected:
        raise ValueError(f'{env} normalizer plan does not match train split')
    return [(episode, *plans[episode]) for episode in sorted(plans)]


def validate_sampling_episode_alignment(manifest: dict) -> None:
    """Require S0/S1 schedule indices to expose the same episode sequence."""
    for env in ENV_NAMES:
        for split in ('train', 'val', 'test'):
            uniform = [int(item['episode']) for item in manifest[env]['s0'][split]]
            temporal = [int(item['episode']) for item in manifest[env]['s1'][split]]
            if uniform != temporal:
                raise ValueError(f'{env}/{split} sampling episode order differs')


def packed_normalizer_from_raw_actions(
    actions: np.ndarray, frameskip: int
) -> dict[str, list[float] | int]:
    actions = np.asarray(actions, dtype=np.float32)
    raw_mean, raw_std = fit_action_normalizer(actions)
    return {
        'raw_mean': raw_mean.tolist(),
        'raw_std': raw_std.tolist(),
        'mean': np.tile(raw_mean, int(frameskip)).tolist(),
        'std': np.tile(raw_std, int(frameskip)).tolist(),
        'n_rows': int(len(actions)),
    }


def _read_action_rows(dataset, plan: list[tuple[int, int, int]], raw_dim: int):
    parts = []
    for episode, first_row, length in plan:
        for start in range(0, length, 4096):
            rows = list(range(first_row + start, first_row + min(start + 4096, length)))
            table = dataset.take(rows, columns=['episode_idx', 'action'])
            observed = np.asarray(table['episode_idx'].to_numpy(), dtype=np.int64)
            if not np.all(observed == episode):
                raise RuntimeError(f'normalizer row plan crosses episode {episode}')
            values = np.asarray(table['action'].to_pylist(), dtype=np.float32)
            if values.shape != (len(rows), raw_dim):
                raise ValueError(f'action shape {values.shape}, expected {(len(rows), raw_dim)}')
            parts.append(values)
    actions = np.concatenate(parts)
    finite = np.isfinite(actions).all(axis=1)
    excluded = int((~finite).sum())
    actions = actions[finite]
    if len(actions) < 2 or not np.isfinite(actions).all():
        raise ValueError('not enough finite actions to fit fresh normalizer')
    return actions, excluded


def _dataset_identity(dataset, path: Path) -> dict:
    return {
        'path': str(path.resolve()),
        'version': int(dataset.version),
        'rows': int(dataset.count_rows()),
        'schema': str(dataset.schema),
    }


def prepare_action_normalizers(
    path: Path,
    manifest: dict,
    manifest_sha256: str,
    datasets: dict,
    dataset_paths: dict,
    frameskip: int,
) -> dict:
    identity = {
        'schema': 1,
        'method': 'finite_raw_action_zscore_tiled_over_frameskip',
        'fit_split': 'manifest_train_episodes_all_valid_action_rows',
        'nonfinite_policy': 'exclude terminal placeholder rows before fitting',
        'label_free_columns': ['episode_idx', 'action'],
        'manifest_sha256': manifest_sha256,
        'frameskip': int(frameskip),
        'datasets': {
            env: _dataset_identity(datasets[env], dataset_paths[env])
            for env in ENV_NAMES
        },
    }
    key = canonical_hash(identity)
    if path.is_file():
        artifact = json.loads(path.read_text())
        if artifact.get('key') != key or artifact.get('identity') != identity:
            raise ValueError('existing fresh action normalizer key mismatch')
        return artifact
    if path.exists():
        raise FileExistsError(f'normalizer path is not a file: {path}')

    normalizers = {}
    for env in ENV_NAMES:
        plan = train_episode_row_plan(manifest, env)
        actions, excluded = _read_action_rows(
            datasets[env], plan, RAW_ACTION_DIMS[env]
        )
        normalizers[env] = packed_normalizer_from_raw_actions(actions, frameskip)
        normalizers[env]['n_nonfinite_rows_excluded'] = excluded
    artifact = {'key': key, 'identity': identity, 'normalizers': normalizers}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(artifact, handle, indent=2)
        handle.write('\n')
    return artifact


def torch_normalizers(artifact: dict) -> dict[str, dict[str, torch.Tensor]]:
    return {
        env: {
            'mean': torch.tensor(artifact['normalizers'][env]['mean'], dtype=torch.float32),
            'std': torch.tensor(artifact['normalizers'][env]['std'], dtype=torch.float32),
        }
        for env in ENV_NAMES
    }


def renormalize_actions(
    actions: torch.Tensor, source: dict, target: dict
) -> torch.Tensor:
    """Change z-score coordinates without requiring the raw action cache."""
    source_mean = torch.as_tensor(source['mean'], dtype=actions.dtype)
    source_std = torch.as_tensor(source['std'], dtype=actions.dtype)
    target_mean = torch.as_tensor(target['mean'], dtype=actions.dtype)
    target_std = torch.as_tensor(target['std'], dtype=actions.dtype)
    if actions.shape[-1] != len(source_mean) or len(source_mean) != len(target_mean):
        raise ValueError('action normalizer width mismatch')
    return (actions * source_std + source_mean - target_mean) / target_std
