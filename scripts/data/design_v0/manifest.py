"""Provenance and episode-level split manifests for Design v0 datasets."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .spec import (
    COLLECT_SEED,
    CONFIG_SEED,
    SCHEMA_VERSION,
    SPLIT_SEED,
    TRAIN_FRACTION,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: str | Path) -> str:
    """Hash a directory by sorted relative paths and per-file SHA-256."""
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(path)
    if root.is_file():
        return sha256_file(root)

    digest = hashlib.sha256()
    files = sorted(p for p in root.rglob('*') if p.is_file())
    for file_path in files:
        relative = file_path.relative_to(root).as_posix()
        digest.update(relative.encode('utf-8'))
        digest.update(b'\0')
        digest.update(bytes.fromhex(sha256_file(file_path)))
    return digest.hexdigest()


def sha256_artifact(path: str | Path) -> str:
    artifact = Path(path)
    if artifact.is_dir():
        return sha256_tree(artifact)
    return sha256_file(artifact)


def indices_sha256(indices: list[int]) -> str:
    encoded = ','.join(str(index) for index in indices).encode()
    return hashlib.sha256(encoded).hexdigest()


def git_commit(repo_root: str | Path | None = None) -> str | None:
    cwd = Path(repo_root) if repo_root is not None else _REPO_ROOT
    try:
        output = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return output.strip() or None


def episode_split(
    num_episodes: int,
    *,
    train_fraction: float = TRAIN_FRACTION,
    split_seed: int = SPLIT_SEED,
) -> tuple[list[int], list[int]]:
    """Deterministic episode-level permutation split.

    Uses ``numpy.random.Generator`` so the result does not depend on
    PyTorch or per-environment derived seeds.
    """
    if num_episodes <= 0:
        raise ValueError('num_episodes must be positive')
    if not 0.0 < train_fraction < 1.0:
        raise ValueError('train_fraction must be strictly between 0 and 1')

    train_length = int(num_episodes * train_fraction)
    val_length = num_episodes - train_length
    if train_length <= 0 or val_length <= 0:
        raise ValueError(
            'episode split needs non-empty train and validation sets, '
            f'got num_episodes={num_episodes}, train={train_length}, '
            f'val={val_length}'
        )

    rng = np.random.default_rng(int(split_seed))
    permutation = rng.permutation(num_episodes).tolist()
    train_indices = permutation[:train_length]
    val_indices = permutation[train_length:]
    return train_indices, val_indices


def dataset_stats(lengths) -> dict[str, Any]:
    values = np.asarray(lengths)
    if values.size == 0:
        raise ValueError('dataset has no episodes')
    return {
        'num_episodes': int(values.size),
        'num_transitions': int(values.sum()),
        'episode_length_min': int(values.min()),
        'episode_length_max': int(values.max()),
        'episode_length_mean': float(values.mean()),
    }


def build_split_manifest(
    *,
    environment: str,
    artifact_path: str | Path,
    num_episodes: int,
    artifact_sha256: str | None = None,
    train_fraction: float = TRAIN_FRACTION,
    split_seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    train_indices, val_indices = episode_split(
        num_episodes,
        train_fraction=train_fraction,
        split_seed=split_seed,
    )
    return {
        'schema_version': SCHEMA_VERSION,
        'environment': environment,
        'artifact_path': str(artifact_path),
        'artifact_sha256': artifact_sha256,
        'unit': 'episode',
        'split_seed': int(split_seed),
        'train_fraction': float(train_fraction),
        'num_episodes': int(num_episodes),
        'train_episode_indices': train_indices,
        'val_episode_indices': val_indices,
        'train_indices_sha256': indices_sha256(train_indices),
        'val_indices_sha256': indices_sha256(val_indices),
    }


def build_provenance(
    *,
    environment: str,
    source: dict[str, Any],
    artifacts: dict[str, Any],
    stats: dict[str, Any],
    split: dict[str, Any] | None = None,
    schema: list[str] | None = None,
    notes: str | None = None,
    git_sha: str | None = None,
    config_seed: int = CONFIG_SEED,
    collect_seed: int | None = COLLECT_SEED,
) -> dict[str, Any]:
    payload = {
        'schema_version': SCHEMA_VERSION,
        'kind': 'design_v0_canonical_dataset',
        'environment': environment,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'status': 'canonical',
        'git_commit': git_sha if git_sha is not None else git_commit(),
        'config_seed': int(config_seed),
        'collect_seed': (
            None if collect_seed is None else int(collect_seed)
        ),
        'source': source,
        'artifacts': artifacts,
        'stats': stats,
        'schema': schema,
        'split': split,
        'notes': notes,
    }
    return payload


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + '\n')
    return dest
