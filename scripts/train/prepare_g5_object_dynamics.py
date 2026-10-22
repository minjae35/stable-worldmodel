"""Prepare immutable label-free train cache and G5 gradient calibration."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault('HF_HUB_OFFLINE', '1')
ROOT = Path('/workspace/stable-worldmodel')
sys.path.insert(0, str(ROOT))

from scripts.train.g2_spatial import DATASETS, DINO_WEIGHTS  # noqa: E402
from scripts.train.g3_history_spatial import encode_bundle  # noqa: E402
from stable_worldmodel.wm.spatial.g2 import (  # noqa: E402
    ENV_NAMES, grad_norm, make_index_schedule, schedule_sha256, tensor_state_sha256,
)
from stable_worldmodel.wm.spatial.g2_fresh import torch_normalizers  # noqa: E402
from stable_worldmodel.wm.spatial.g3_history import load_label_free_history_clips  # noqa: E402
from stable_worldmodel.wm.spatial.g5 import (  # noqa: E402
    NEW_UPDATES, canonical_hash, sha256_file, validate_h1_source,
)
from stable_worldmodel.wm.spatial.g5_objective import (  # noqa: E402
    G5_ARMS, make_derangement_schedule, objective_components,
)
from stable_worldmodel.wm.spatial.model import FrozenDINOv2, SpatialDynamics  # noqa: E402

DEFAULT_ROOT = ROOT / 'results/stage3/g5_object_dynamics'
SOURCE_DIR = ROOT / 'results/stage3/g3_history/H1_uniform_mse'
SOURCE_CHECKPOINT = SOURCE_DIR / 'checkpoint_step2000.pt'
COMMON_MANIFEST = ROOT / 'results/stage3/g3_history/common_windows.json'
NORMALIZERS = ROOT / 'results/stage3/g2_fresh_ab/action_normalizers.json'


def _batch(model, bundle, indices, permutation, env_index, device):
    take = torch.as_tensor(indices, dtype=torch.long)
    context = bundle['context'].index_select(0, take).to(device).float()
    target = bundle['target'].index_select(0, take).to(device).float()[:, None]
    actions = bundle['actions'].index_select(0, take).to(device).float()
    env_id = torch.full((len(take),), env_index, device=device, dtype=torch.long)
    permutation = torch.as_tensor(permutation, device=device, dtype=torch.long)
    return context, target, actions, actions.index_select(0, permutation), env_id


def _gradient_norm_for(
    model, batches, arm, params, objective, rank_lambda=0.0, component='raw_objective'
):
    model.zero_grad(set_to_none=True)
    losses = []
    for context, target, actions, shuffled_actions, env_id in batches:
        prediction = model(context, actions, env_id)[:, -1:]
        shuffled = model(context, shuffled_actions, env_id)[:, -1:]
        parts = objective_components(
            arm, prediction, target, context[:, -1:], shuffled,
            motion_blend=objective['motion_blend'],
            relative_cap=objective['relative_cap'],
            margin_fraction=objective['margin_fraction'],
            ranking_lambda=rank_lambda, objective_scale=1.0,
        )
        losses.append(parts[component])
    torch.stack(losses).mean().backward()
    value = grad_norm(params)
    model.zero_grad(set_to_none=True)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--seed', type=int, default=3072)
    parser.add_argument('--batch-per-env', type=int, default=16)
    parser.add_argument('--encode-batch', type=int, default=32)
    args = parser.parse_args(argv)
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'STATUS').write_text('PREPARING\n')
    cache_dir = output / 'shared_train_cache'
    cache_dir.mkdir()

    source_config = json.loads((SOURCE_DIR / 'config.json').read_text())
    source = torch.load(SOURCE_CHECKPOINT, map_location='cpu', weights_only=True)
    source_validation = validate_h1_source(source, source_config)
    common = json.loads(COMMON_MANIFEST.read_text())
    normalizer_artifact = json.loads(NORMALIZERS.read_text())
    normalizers = torch_normalizers(normalizer_artifact)
    objective = {'motion_blend': 0.5, 'relative_cap': 4.0, 'margin_fraction': 0.05}

    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    encoder = FrozenDINOv2.from_pretrained(str(DINO_WEIGHTS)).to(device).eval()

    import lance
    schedules, derangements, bundles = {}, {}, {}
    cache_hashes, schedule_hashes, derangement_hashes = {}, {}, {}
    for env_index, env in enumerate(ENV_NAMES):
        items = list(common['environments'][env]['train'])
        raw = load_label_free_history_clips(
            lance.dataset(str(DATASETS[env])), items, env, 5, 1
        )
        bundle = encode_bundle(raw, normalizers[env], encoder, device, args.encode_batch)
        saved = {
            'context': bundle['context'].half(), 'target': bundle['target'].half(),
            'actions': bundle['actions'].float(), 'episode': torch.from_numpy(bundle['episode']),
            'start': torch.from_numpy(bundle['start']),
        }
        path = cache_dir / f'{env}_train.pt'
        torch.save(saved, path)
        cache_hashes[env] = sha256_file(path)
        bundles[env] = saved
        schedules[env] = make_index_schedule(
            len(items), args.batch_per_env, NEW_UPDATES, args.seed + 5101 + env_index
        )
        derangements[env] = make_derangement_schedule(
            NEW_UPDATES, args.batch_per_env, args.seed + 6101 + env_index
        )
        schedule_hashes[env] = schedule_sha256(schedules[env])
        derangement_hashes[env] = schedule_sha256(derangements[env])
    del encoder
    torch.cuda.empty_cache()
    schedule_path = cache_dir / 'schedules.pt'
    # Tensor-only serialization remains loadable with PyTorch's restricted
    # weights_only unpickler; do not require arbitrary pickle execution.
    torch.save({
        'indices': {env: torch.from_numpy(value) for env, value in schedules.items()},
        'derangements': {
            env: torch.from_numpy(value) for env, value in derangements.items()
        },
    }, schedule_path)

    model = SpatialDynamics().to(device)
    model.load_state_dict(source['model'], strict=True)
    if tensor_state_sha256(model.state_dict()) != source_validation['model_sha256']:
        raise RuntimeError('source model hash changed during load')
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    batches = [
        _batch(model, bundles[env], schedules[env][0], derangements[env][0], i, device)
        for i, env in enumerate(ENV_NAMES)
    ]
    base_norms = {
        arm: _gradient_norm_for(model, batches, arm, params, objective, 0.0)
        for arm in G5_ARMS
    }
    calibration = {}
    reference = base_norms['A_uniform_control']
    for arm in G5_ARMS:
        rank_lambda = 0.0
        if arm in ('C_action_sensitive', 'D_combined'):
            rank_norm = _gradient_norm_for(
                model, batches, arm, params, objective, 1.0, component='ranking_loss'
            )
            # At lambda=0 this arm is its base loss.  Use a conservative fixed
            # 25% reference-gradient budget for the factual ranking term.
            rank_lambda = min(1.0, 0.25 * reference / max(rank_norm, 1e-12))
        total_norm = _gradient_norm_for(model, batches, arm, params, objective, rank_lambda)
        if not np.isfinite(total_norm) or total_norm <= 0:
            raise RuntimeError(f'invalid calibration gradient for {arm}: {total_norm}')
        calibration[arm] = {
            'base_grad_norm': base_norms[arm], 'ranking_lambda': rank_lambda,
            'ranking_grad_norm_unit_lambda': rank_norm if arm in (
                'C_action_sensitive', 'D_combined'
            ) else 0.0,
            'combined_grad_norm': total_norm, 'objective_scale': reference / total_norm,
        }

    cache_key_payload = {
        'common_manifest_sha256': sha256_file(COMMON_MANIFEST),
        'normalizer_sha256': sha256_file(NORMALIZERS),
        'dino_sha256': sha256_file(DINO_WEIGHTS / 'model.safetensors'),
        'preprocessing': 'ImageNet mean/std; Frozen DINOv2-S/14 final postnorm patch tokens',
        'history': 1, 'frameskip': 5, 'cache_file_sha256': cache_hashes,
    }
    train_cache_key = canonical_hash(cache_key_payload)
    plan = {
        'schema': 1, 'seed': args.seed, 'batch_per_environment': args.batch_per_env,
        'new_updates_per_arm': NEW_UPDATES, 'total_optimizer_updates': NEW_UPDATES * 4,
        'smoke_updates_included_in_budget': 10, 'arms': list(G5_ARMS),
        'source': {
            **source_validation, 'checkpoint': str(SOURCE_CHECKPOINT),
            'checkpoint_sha256': sha256_file(SOURCE_CHECKPOINT),
            'config': str(SOURCE_DIR / 'config.json'),
        },
        'common_manifest': str(COMMON_MANIFEST),
        'common_manifest_sha256': sha256_file(COMMON_MANIFEST),
        'normalizer_file': str(NORMALIZERS), 'normalizer_sha256': sha256_file(NORMALIZERS),
        'dino_sha256': cache_key_payload['dino_sha256'],
        'train_cache_key': train_cache_key, 'cache_file_sha256': cache_hashes,
        'schedule_file': str(schedule_path), 'schedule_file_sha256': sha256_file(schedule_path),
        'schedule_sha256': schedule_hashes,
        'schedule_set_sha256': canonical_hash(schedule_hashes),
        'derangement_sha256': derangement_hashes,
        'derangement_set_sha256': canonical_hash(derangement_hashes),
        'objective': objective, 'calibration': calibration,
        'training_columns': ['episode_idx', 'pixels', 'action'],
        'forbidden_training_inputs': common['forbidden_training_inputs'],
        'architecture': 'unchanged Shared SpatialDynamics H1; no temporal encoding',
        'evaluation_only_labels': True,
    }
    (output / 'gradient_calibration.json').write_text(json.dumps(calibration, indent=2) + '\n')
    (output / 'experiment_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    (cache_dir / 'metadata.json').write_text(json.dumps(cache_key_payload, indent=2) + '\n')
    (cache_dir / 'STATUS').write_text('READY\n')
    (output / 'STATUS').write_text('PREPARED\n')


if __name__ == '__main__':
    main()
