#!/usr/bin/env python
"""TwoRoom Design v0 CEM planning smoke.

Loads ``design_v0_tworoom/last.ckpt``, wraps it with the planning adapter,
and runs a tiny dataset-driven TwoRoom evaluation through
``ShootingCostEvaluator`` + ``CEMSolver`` + ``WorldModelPolicy``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torchvision.transforms import v2 as transforms

from stable_worldmodel.planning import (
    CEMSolver,
    GoalMSE,
    ShootingCostEvaluator,
)
from stable_worldmodel.policy import PlanConfig, WorldModelPolicy
from stable_worldmodel.wm.design_v0 import load_planning_adapter


CKPT = Path(
    '/home/bgh980219/stablewm-data/checkpoints/'
    'design_v0_tworoom/last.ckpt'
)
SPLIT = Path(
    '/home/bgh980219/stablewm-data/datasets/design_v0/'
    'splits/tworoom.json'
)
DEVICE = 'cuda'
NUM_EPISODES = 2
GOAL_OFFSET = 25
EVAL_BUDGET = 25
MIN_EPISODE_LEN = GOAL_OFFSET + 1
CEM_SAMPLES = 16
CEM_STEPS = 3
CEM_TOPK = 4
ACTION_BLOCK = 5


def _img_transform():
    stats = spt.data.dataset_stats.ImageNet
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**stats),
            transforms.Resize(size=224),
        ]
    )


def _finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f'{name} contains NaN/Inf')


def adapter_sanity(adapter) -> dict[str, tuple]:
    adapter.eval()
    batch, samples, context, horizon = 1, 4, 2, 3
    env_action_dim = adapter.action_dim
    max_action_dim = adapter.core.action_encoder.max_action_dim
    latent_dim = adapter.core.latent_dim
    pixels = torch.rand(batch, context, 3, 224, 224, device=DEVICE)
    encoded = adapter.encode({'pixels': pixels})
    emb = encoded['emb']
    _finite('encode/emb', emb)
    if emb.shape != (batch, context, latent_dim):
        raise RuntimeError(f'unexpected encode shape {tuple(emb.shape)}')

    info = {
        'pixels': torch.rand(
            batch, samples, context, 3, 224, 224, device=DEVICE
        ),
        'action_history': torch.zeros(
            batch, samples, context - 1, env_action_dim, device=DEVICE
        ),
        'goal': torch.rand(
            batch, samples, 1, 3, 224, 224, device=DEVICE
        ),
        'action': torch.zeros(
            batch, samples, context, env_action_dim, device=DEVICE
        ),
    }
    candidates = torch.zeros(
        batch, samples, horizon, env_action_dim, device=DEVICE
    )
    seen_action = []
    seen_mask = []
    seen_env = []
    original = adapter.core.predict_next

    def wrapped(history, action, mask, env_id):
        seen_action.append(action.detach())
        seen_mask.append(mask.detach())
        seen_env.append(env_id.detach())
        return original(history, action, mask, env_id)

    adapter.core.predict_next = wrapped
    rolled = adapter.rollout(dict(info), candidates)
    adapter.core.predict_next = original
    predicted = rolled['predicted_emb']
    _finite('rollout/predicted_emb', predicted)
    expected = (batch, samples, context + horizon, latent_dim)
    if predicted.shape != expected:
        raise RuntimeError(
            f'unexpected rollout shape {tuple(predicted.shape)}'
        )
    padded = seen_action[0]
    mask = seen_mask[0]
    if padded.shape[-1] != max_action_dim:
        raise RuntimeError(
            f'padded action dim {padded.shape[-1]} != {max_action_dim}'
        )
    valid = int(mask[0].sum())
    if valid != env_action_dim:
        raise RuntimeError(
            f'mask has {valid} valid dims, expected {env_action_dim}'
        )
    if bool(mask[0, env_action_dim:].any()):
        raise RuntimeError('pad region of action_mask is not False')
    if int(seen_env[0][0]) != adapter.default_env_id:
        raise RuntimeError(
            f'env_id {int(seen_env[0][0])} != {adapter.default_env_id}'
        )
    cost = ShootingCostEvaluator(adapter, GoalMSE()).get_cost(
        dict(info), candidates
    )
    _finite('get_cost', cost)
    return {
        'encode': tuple(emb.shape),
        'rollout': tuple(predicted.shape),
        'cost': tuple(cost.shape),
        'padded_action': tuple(padded.shape),
        'valid_mask_dims': valid,
        'env_id': int(seen_env[0][0]),
    }


def _choose_val_episodes(dataset, val_indices: list[int]) -> list[int]:
    lengths = np.asarray(dataset.lengths)
    chosen = []
    for episode in val_indices:
        if int(lengths[episode]) >= MIN_EPISODE_LEN:
            chosen.append(int(episode))
        if len(chosen) == NUM_EPISODES:
            return chosen
    raise RuntimeError(
        'Not enough TwoRoom val episodes with length '
        f'>= {MIN_EPISODE_LEN}'
    )


def cem_env_smoke(adapter) -> dict:
    metadata = adapter.metadata or {}
    plan_horizon = int(metadata.get('horizon', 3))
    history_len = int(adapter.core.history_size)
    frameskip = ACTION_BLOCK
    datasets = metadata.get('datasets') or {}
    tworoom = datasets.get('TwoRoom') or {}
    frameskip = int(tworoom.get('frameskip', ACTION_BLOCK))

    split = json.loads(SPLIT.read_text())
    dataset = swm.data.load_dataset(
        'design_v0/tworoom_expert.lance',
        cache_dir=os.environ.get('STABLEWM_HOME'),
        keys_to_cache=['action'],
    )
    episodes = _choose_val_episodes(
        dataset, split['val_episode_indices']
    )
    start_steps = [0] * len(episodes)
    print('val episodes', episodes)

    cost = ShootingCostEvaluator(adapter, GoalMSE())
    solver = CEMSolver(
        cost=cost,
        num_samples=CEM_SAMPLES,
        n_steps=CEM_STEPS,
        topk=CEM_TOPK,
        device=DEVICE,
        seed=42,
    )
    config = PlanConfig(
        horizon=plan_horizon,
        receding_horizon=plan_horizon,
        history_len=history_len,
        action_block=frameskip,
        warm_start=False,
    )
    transform = {'pixels': _img_transform(), 'goal': _img_transform()}
    policy = WorldModelPolicy(
        solver=solver, config=config, transform=transform
    )
    world = swm.World(
        env_name='swm/TwoRoom-v1',
        num_envs=NUM_EPISODES,
        image_shape=(224, 224),
        max_episode_steps=2 * EVAL_BUDGET,
    )
    world.set_policy(policy)
    metrics = world.evaluate(
        dataset=dataset,
        episodes_idx=episodes,
        start_steps=start_steps,
        goal_offset=GOAL_OFFSET,
        eval_budget=EVAL_BUDGET,
        callables=[
            {
                'method': '_set_state',
                'args': {'state': {'value': 'state'}},
            },
            {
                'method': '_set_goal_state',
                'args': {
                    'goal_state': {'value': 'goal_state'}
                },
            },
        ],
    )
    return {
        'episodes': episodes,
        'metrics': metrics,
        'plan_horizon': plan_horizon,
        'history_len': history_len,
        'action_block': frameskip,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Design v0 TwoRoom CEM planning smoke'
    )
    parser.add_argument('--ckpt', type=Path, default=CKPT)
    parser.add_argument(
        '--environment',
        default=None,
        help='Checkpoint env name, required for joint checkpoints',
    )
    args = parser.parse_args()
    os.environ.setdefault('STABLEWM_HOME', '/home/bgh980219/stablewm-data')
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    if not args.ckpt.is_file():
        raise FileNotFoundError(f'missing checkpoint {args.ckpt}')

    print('=== adapter sanity ===')
    print('ckpt', args.ckpt)
    adapter = load_planning_adapter(
        args.ckpt,
        map_location=DEVICE,
        environment=args.environment,
    )
    adapter = adapter.to(DEVICE)
    meta = adapter.metadata or {}
    print('env_names', meta.get('env_names'))
    print('env_to_id', meta.get('env_to_id'))
    print('resolved env_id', adapter.default_env_id)
    print('history_size', adapter.core.history_size)
    print('env action_dim', adapter.action_dim)
    print('max_action_dim', adapter.core.action_encoder.max_action_dim)
    print(
        'env embed',
        tuple(adapter.core.environment_embedding.weight.shape),
    )
    shapes = adapter_sanity(adapter)
    print('encode shape', shapes['encode'])
    print('rollout shape', shapes['rollout'])
    print('padded action', shapes['padded_action'])
    print('valid mask dims', shapes['valid_mask_dims'])
    print('rollout env_id', shapes['env_id'])
    print('cost shape', shapes['cost'])
    print('SANITY PASS')

    print('=== TwoRoom CEM smoke ===')
    result = cem_env_smoke(adapter)
    print('plan_horizon', result['plan_horizon'])
    print('history_len', result['history_len'])
    print('action_block', result['action_block'])
    print('metrics', result['metrics'])
    successes = result['metrics']['episode_successes']
    if np.asarray(successes).shape != (NUM_EPISODES,):
        raise RuntimeError('unexpected episode_successes shape')
    print('CEM SMOKE PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
