"""Reproducibility helpers for the Stage-3 G5 controlled experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from stable_worldmodel.wm.spatial.g2 import config_sha256, tensor_state_sha256
from stable_worldmodel.wm.spatial.g2_extension import optimizer_steps


SOURCE_STEP = 2000
NEW_UPDATES = 500
FINAL_STEP = SOURCE_STEP + NEW_UPDATES


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_h1_source(payload: dict, source_config: dict) -> dict:
    """Require a complete H1 step-2000 model/optimizer/RNG continuation state."""
    if payload.get('config_sha256') != config_sha256(source_config):
        raise ValueError('H1 source config hash mismatch')
    if int(payload.get('step', -1)) != SOURCE_STEP:
        raise ValueError('G5 source must be the H1 step-2000 checkpoint')
    if optimizer_steps(payload['optimizer']) != {SOURCE_STEP}:
        raise ValueError('H1 optimizer state is not uniformly at step 2000')
    extra = payload.get('extra', {})
    if 'rng_state' not in extra:
        raise ValueError('H1 source lacks RNG state')
    if source_config.get('history') != 1 or source_config.get('loss') != 'mse':
        raise ValueError('G5 requires the primary H1 MSE baseline')
    if source_config.get('architecture_changed') is not False:
        raise ValueError('G5 requires the unchanged SpatialDynamics architecture')
    return {
        'step': SOURCE_STEP,
        'model_sha256': tensor_state_sha256(payload['model']),
        'optimizer_steps': [SOURCE_STEP],
        'rng_present': True,
    }


def run_config(plan: dict, arm: str, output_dir: Path) -> dict:
    from stable_worldmodel.wm.spatial.g5_objective import ARM_SPECS

    if arm not in ARM_SPECS:
        raise ValueError(f'unknown arm {arm!r}')
    spec = ARM_SPECS[arm]
    return {
        'experiment': 'g5_scalable_label_free_object_dynamics',
        'arm': arm,
        'source_step': SOURCE_STEP,
        'new_updates': NEW_UPDATES,
        'final_step': FINAL_STEP,
        'history': 1,
        'frameskip': 5,
        'seed': int(plan['seed']),
        'batch_per_environment': int(plan['batch_per_environment']),
        'optimizer': 'AdamW',
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'gradient_clip': 1.0,
        'sampling': 'common_uniform_anchor_schedule',
        'loss': 'mse_plus_label_free_g5_terms',
        'motion_balanced': spec.motion_balanced,
        'action_sensitive': spec.action_sensitive,
        'motion_blend': float(plan['objective']['motion_blend']),
        'relative_cap': float(plan['objective']['relative_cap']),
        'margin_fraction': float(plan['objective']['margin_fraction']),
        'ranking_lambda': float(plan['calibration'][arm]['ranking_lambda']),
        'objective_scale': float(plan['calibration'][arm]['objective_scale']),
        'source_checkpoint_sha256': plan['source']['checkpoint_sha256'],
        'source_model_sha256': plan['source']['model_sha256'],
        'train_cache_key': plan['train_cache_key'],
        'schedule_set_sha256': plan['schedule_set_sha256'],
        'derangement_set_sha256': plan['derangement_set_sha256'],
        'common_manifest_sha256': plan['common_manifest_sha256'],
        'normalizer_sha256': plan['normalizer_sha256'],
        'dino_sha256': plan['dino_sha256'],
        'training_columns': plan['training_columns'],
        'forbidden_training_inputs': plan['forbidden_training_inputs'],
        'output_dir': str(Path(output_dir).resolve()),
    }


def checkpoint_model_hash(path: Path) -> str:
    payload = torch.load(path, map_location='cpu', weights_only=True)
    return tensor_state_sha256(payload['model'])
