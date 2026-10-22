"""Strict continuation utilities for the G2 Fresh 500 -> 2,000 study."""

from __future__ import annotations

import random

import numpy as np
import torch

from stable_worldmodel.wm.spatial.g2 import config_sha256, schedule_sha256


def validate_schedule_prefix(full: np.ndarray, completed: int, expected_hash: str) -> str:
    if completed < 1 or completed > len(full):
        raise ValueError('completed step count is outside the full schedule')
    observed = schedule_sha256(full[:completed])
    if observed != expected_hash:
        raise ValueError('extended sampler schedule does not preserve the completed prefix')
    return observed


def optimizer_steps(optimizer_state: dict) -> set[int]:
    steps = set()
    for state in optimizer_state['state'].values():
        value = state.get('step')
        if value is None:
            continue
        steps.add(int(value.item()) if torch.is_tensor(value) else int(value))
    return steps


def validate_source_checkpoint(payload: dict, source_config: dict, completed: int) -> None:
    if payload.get('config_sha256') != config_sha256(source_config):
        raise ValueError('source checkpoint config hash mismatch')
    if int(payload.get('step', -1)) != int(completed):
        raise ValueError('source checkpoint is not at the requested completed step')
    steps = optimizer_steps(payload['optimizer'])
    if steps != {int(completed)}:
        raise ValueError(f'optimizer states are not all at step {completed}: {steps}')


def capture_rng_state() -> dict:
    legacy = np.random.get_state()
    return {
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all(),
        'numpy_legacy': {
            'name': legacy[0],
            'keys': torch.from_numpy(legacy[1].copy()),
            'position': int(legacy[2]),
            'has_gauss': int(legacy[3]),
            'cached_gaussian': float(legacy[4]),
        },
        'python_random': random.getstate(),
    }


def restore_rng_state(state: dict) -> None:
    torch.set_rng_state(state['torch_cpu'])
    torch.cuda.set_rng_state_all(state['torch_cuda'])
    legacy = state['numpy_legacy']
    np.random.set_state((
        legacy['name'],
        legacy['keys'].cpu().numpy(),
        int(legacy['position']),
        int(legacy['has_gauss']),
        float(legacy['cached_gaussian']),
    ))
    random.setstate(state['python_random'])
