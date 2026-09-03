#!/usr/bin/env python
"""Design v0 single vs joint planning comparison.

Uses the existing Stable-WM dataset-driven CEM protocol (eval_wm.py) with a
Design v0 Lightning checkpoint adapter. Does not modify baseline world models
or the planning core.

Single and joint runs of one environment share one frozen evaluation spec so
the 50 start/goal tasks are identical.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

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


os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('STABLEWM_HOME', '/home/bgh980219/stablewm-data')

CACHE_DIR = os.environ['STABLEWM_HOME']
CKPT_ROOT = Path(CACHE_DIR) / 'checkpoints'
OUT_ROOT = Path(CACHE_DIR) / 'runs' / 'design_v0_planning_compare'
DEVICE = 'cuda'

NUM_EVAL = 50
SEED = 42
GOAL_OFFSET = 25
EVAL_BUDGET = 50
IMG_SIZE = 224
HORIZON = 5
RECEDING_HORIZON = 5
ACTION_BLOCK = 5
HISTORY_LEN = 2
CEM = dict(num_samples=300, n_steps=30, topk=30, var_scale=1.0, batch_size=1)

ENVIRONMENTS: dict[str, dict[str, Any]] = {
    'TwoRoom': {
        'dataset_name': 'design_v0/tworoom_expert.lance',
        'env_name': 'swm/TwoRoom-v1',
        'world_kwargs': {},
        'single_ckpt': 'design_v0_tworoom/last.ckpt',
        'callables': [
            {
                'method': '_set_state',
                'args': {'state': {'value': 'state'}},
            },
            {
                'method': '_set_goal_state',
                'args': {'goal_state': {'value': 'goal_state'}},
            },
        ],
    },
    'PushT': {
        'dataset_name': 'design_v0/pusht_expert_train.h5',
        'env_name': 'swm/PushT-v1',
        'world_kwargs': {},
        'single_ckpt': 'design_v0_pusht/last.ckpt',
        'callables': [
            {
                'method': '_set_state',
                'args': {'state': {'value': 'state'}},
            },
            {
                'method': '_set_goal_state',
                'args': {'goal_state': {'value': 'goal_state'}},
            },
        ],
    },
    'OGBCube': {
        'dataset_name': 'ogbench/cube_single_expert.h5',
        'env_name': 'swm/OGBCube-v0',
        'world_kwargs': {
            'env_type': 'single',
            'ob_type': 'states',
            'multiview': False,
            'width': 224,
            'height': 224,
            'visualize_info': False,
            'terminate_at_goal': True,
        },
        'single_ckpt': 'design_v0_ogbcube/last.ckpt',
        'callables': [
            {
                'method': 'set_state',
                'args': {
                    'qpos': {'value': 'qpos'},
                    'qvel': {'value': 'qvel'},
                },
            },
            {
                'method': 'set_target_pos',
                'args': {
                    'cube_id': {'value': 0, 'in_dataset': False},
                    'target_pos': {'value': 'goal_privileged_block_0_pos'},
                    'target_quat': {
                        'value': 'goal_privileged_block_0_quat'
                    },
                },
            },
        ],
    },
}
JOINT_CKPT = 'design_v0_all/last.ckpt'


def img_transform():
    stats = spt.data.dataset_stats.ImageNet
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**stats),
            transforms.Resize(size=IMG_SIZE),
        ]
    )


def episode_col(dataset):
    names = set(dataset.column_names)
    names |= set(getattr(dataset, '_schema_names', ()))
    return 'episode_idx' if 'episode_idx' in names else 'ep_idx'


def get_episodes_length(dataset, episodes):
    col_name = episode_col(dataset)
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data('step_idx')
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def load_eval_dataset(dataset_name: str):
    return swm.data.load_dataset(
        dataset_name,
        cache_dir=CACHE_DIR,
        keys_to_cache=['action'],
    )


def sample_eval_tasks(dataset, num_eval: int, goal_offset: int, seed: int):
    """Copy of ``scripts/plan/eval_wm.py`` start-row sampling."""
    col_name = episode_col(dataset)
    ep_indices, _ = np.unique(
        dataset.get_col_data(col_name), return_index=True
    )
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - goal_offset - 1
    max_start_idx_dict = {
        ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)
    }
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data('step_idx') <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    if len(valid_indices) < num_eval:
        raise ValueError(
            f'Only {len(valid_indices)} valid starts, need {num_eval}'
        )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(valid_indices), size=num_eval, replace=False)
    chosen = np.sort(valid_indices[chosen])
    eval_episodes = dataset.get_col_data(col_name)[chosen]
    eval_start_idx = dataset.get_col_data('step_idx')[chosen]
    return [
        {
            'eval_index': int(i),
            'dataset_row': int(chosen[i]),
            'episode_idx': int(eval_episodes[i]),
            'start_step': int(eval_start_idx[i]),
        }
        for i in range(num_eval)
    ]


def spec_path(environment: str) -> Path:
    return OUT_ROOT / 'specs' / f'{environment.lower()}.json'


def result_path(environment: str, role: str) -> Path:
    return OUT_ROOT / 'results' / f'{environment.lower()}_{role}.json'


def protocol_dict() -> dict[str, Any]:
    return {
        'num_eval': NUM_EVAL,
        'seed': SEED,
        'goal_offset_steps': GOAL_OFFSET,
        'eval_budget': EVAL_BUDGET,
        'horizon': HORIZON,
        'receding_horizon': RECEDING_HORIZON,
        'history_len': HISTORY_LEN,
        'action_block': ACTION_BLOCK,
        'cem': CEM,
        'objective': 'GoalMSE(reduction=sum)',
        'action_scaler': None,
        'image_preprocessing': 'imagenet_normalize_resize_224',
        'checkpoint_rule': 'last.ckpt',
        'metric': 'success_rate',
        'warm_start': True,
    }


def make_spec(environment: str, *, force: bool = False) -> dict[str, Any]:
    cfg = ENVIRONMENTS[environment]
    path = spec_path(environment)
    if path.is_file() and not force:
        return json.loads(path.read_text())
    dataset_name = cfg['dataset_name']
    dataset_path = Path(CACHE_DIR) / 'datasets' / dataset_name
    if not dataset_path.exists():
        raise FileNotFoundError(
            f'{environment} evaluation dataset is missing: {dataset_path}'
        )
    dataset = load_eval_dataset(dataset_name)
    tasks = sample_eval_tasks(dataset, NUM_EVAL, GOAL_OFFSET, SEED)
    spec = {
        'environment': environment,
        'dataset_name': dataset_name,
        'protocol': protocol_dict(),
        'callables': cfg['callables'],
        'world_kwargs': cfg['world_kwargs'],
        'env_name': cfg['env_name'],
        'n_valid_starts_note': (
            'Sampled with eval_wm.py valid-start procedure, seed=42'
        ),
        'tasks': tasks,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2))
    return spec


def resolve_ckpt(relative: str) -> Path:
    path = CKPT_ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(f'missing checkpoint {path}')
    return path


def build_policy(adapter, environment: str):
    metadata = adapter.metadata or {}
    mapping = metadata.get('env_to_id') or {}
    if environment not in mapping:
        raise KeyError(
            f'{environment!r} not in checkpoint env_to_id {sorted(mapping)}'
        )
    dims = metadata.get('effective_action_dims') or {}
    if environment not in dims:
        raise KeyError(
            f'{environment!r} not in effective_action_dims {sorted(dims)}'
        )
    adapter.eval()
    adapter.requires_grad_(False)
    cost = ShootingCostEvaluator(adapter, GoalMSE(reduction='sum'))
    solver = CEMSolver(
        cost=cost,
        device=DEVICE,
        seed=SEED,
        **CEM,
    )
    config = PlanConfig(
        horizon=HORIZON,
        receding_horizon=RECEDING_HORIZON,
        history_len=HISTORY_LEN,
        action_block=ACTION_BLOCK,
        warm_start=True,
    )
    transform = {'pixels': img_transform(), 'goal': img_transform()}
    return WorldModelPolicy(
        solver=solver, config=config, transform=transform
    )


def _bool_successes(metrics: dict) -> list[bool]:
    raw = np.asarray(metrics['episode_successes']).reshape(-1)
    return [bool(x) for x in raw]


def run_eval(environment: str, role: str, spec: dict[str, Any]) -> dict[str, Any]:
    cfg = ENVIRONMENTS[environment]
    ckpt_rel = cfg['single_ckpt'] if role == 'single' else JOINT_CKPT
    ckpt = resolve_ckpt(ckpt_rel)
    tasks = spec['tasks']
    episodes = [int(t['episode_idx']) for t in tasks]
    starts = [int(t['start_step']) for t in tasks]
    n = len(tasks)

    adapter = load_planning_adapter(
        ckpt,
        map_location=DEVICE,
        environment=environment,
    )
    adapter = adapter.to(DEVICE)
    metadata = adapter.metadata or {}
    policy = build_policy(adapter, environment)

    world = swm.World(
        env_name=cfg['env_name'],
        num_envs=n,
        image_shape=(IMG_SIZE, IMG_SIZE),
        max_episode_steps=2 * EVAL_BUDGET,
        **cfg['world_kwargs'],
    )
    world.set_policy(policy)
    dataset = load_eval_dataset(cfg['dataset_name'])

    started = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        episodes_idx=episodes,
        start_steps=starts,
        goal_offset=GOAL_OFFSET,
        eval_budget=EVAL_BUDGET,
        callables=cfg['callables'],
    )
    elapsed = time.time() - started
    successes = _bool_successes(metrics)
    if len(successes) != n:
        raise RuntimeError(
            f'episode_successes length {len(successes)} != {n}'
        )
    success_rate = float(metrics['success_rate'])
    payload = {
        'environment': environment,
        'role': role,
        'ckpt': str(ckpt),
        'env_id': int(adapter.default_env_id),
        'action_dim': int(adapter.action_dim),
        'max_action_dim': int(adapter.core.action_encoder.max_action_dim),
        'env_to_id': metadata.get('env_to_id'),
        'effective_action_dims': metadata.get('effective_action_dims'),
        'success_rate': success_rate,
        'elapsed_sec': elapsed,
        'nan_or_exception': False,
        'tasks': [
            {
                **tasks[i],
                'success': successes[i],
            }
            for i in range(n)
        ],
        'metrics': {
            'success_rate': success_rate,
            'episode_successes': successes,
        },
    }
    out = result_path(environment, role)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    return payload


def paired_table(single: dict, joint: dict) -> dict[str, int]:
    s = [t['success'] for t in single['tasks']]
    j = [t['success'] for t in joint['tasks']]
    both = sum(a and b for a, b in zip(s, j))
    single_only = sum(a and not b for a, b in zip(s, j))
    joint_only = sum((not a) and b for a, b in zip(s, j))
    neither = sum((not a) and (not b) for a, b in zip(s, j))
    return {
        'both_success': both,
        'single_only': single_only,
        'joint_only': joint_only,
        'both_fail': neither,
    }


def write_trials_csv(environment: str, single: dict, joint: dict) -> Path:
    path = OUT_ROOT / 'results' / f'{environment.lower()}_trials.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'eval_index',
                'episode_idx',
                'start_step',
                'dataset_row',
                'single_success',
                'joint_success',
            ],
        )
        writer.writeheader()
        for s_task, j_task in zip(single['tasks'], joint['tasks']):
            if s_task['eval_index'] != j_task['eval_index']:
                raise RuntimeError('single/joint task order mismatch')
            writer.writerow(
                {
                    'eval_index': s_task['eval_index'],
                    'episode_idx': s_task['episode_idx'],
                    'start_step': s_task['start_step'],
                    'dataset_row': s_task['dataset_row'],
                    'single_success': int(s_task['success']),
                    'joint_success': int(j_task['success']),
                }
            )
    return path


def summarize(environments: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        'protocol': protocol_dict(),
        'environments': {},
    }
    for environment in environments:
        single_file = result_path(environment, 'single')
        joint_file = result_path(environment, 'joint')
        if not single_file.is_file() or not joint_file.is_file():
            summary['environments'][environment] = {
                'status': 'incomplete',
                'single': str(single_file),
                'joint': str(joint_file),
            }
            continue
        single = json.loads(single_file.read_text())
        joint = json.loads(joint_file.read_text())
        paired = paired_table(single, joint)
        csv_path = write_trials_csv(environment, single, joint)
        s_rate = float(single['success_rate'])
        j_rate = float(joint['success_rate'])
        summary['environments'][environment] = {
            'status': 'complete',
            'single_success_rate': s_rate,
            'joint_success_rate': j_rate,
            'delta_joint_minus_single': j_rate - s_rate,
            'paired': paired,
            'single_result': str(single_file),
            'joint_result': str(joint_file),
            'trials_csv': str(csv_path),
            'single_env_id': single.get('env_id'),
            'joint_env_id': joint.get('env_id'),
            'single_action_dim': single.get('action_dim'),
            'joint_action_dim': joint.get('action_dim'),
        }
    out = OUT_ROOT / 'summary.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Design v0 single vs joint planning comparison'
    )
    parser.add_argument(
        '--environment',
        choices=list(ENVIRONMENTS),
        action='append',
        dest='environments',
    )
    parser.add_argument(
        '--role',
        choices=('single', 'joint'),
        default=None,
        help='Run one role. Default: both, then summarize.',
    )
    parser.add_argument('--make-specs-only', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--force-spec', action='store_true')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    environments = args.environments or list(ENVIRONMENTS)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = OUT_ROOT / 'compare.log'

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open('a') as handle:
            handle.write(message + '\n')

    if args.summarize_only:
        summary = summarize(environments)
        log(json.dumps(summary, indent=2))
        return 0

    for environment in environments:
        log(f'=== spec {environment} ===')
        try:
            spec = make_spec(environment, force=args.force_spec)
        except FileNotFoundError as exc:
            log(f'SPEC FAILED {environment}: {exc}')
            if environment == 'OGBCube':
                log(
                    'OGBCube stopped: existing protocol dataset '
                    'ogbench/cube_single_expert.h5 is not on disk. '
                    'Not substituting canonical front-only lance.'
                )
            continue
        log(f'wrote {spec_path(environment)} n={len(spec["tasks"])}')
        if args.make_specs_only:
            continue
        roles = (args.role,) if args.role else ('single', 'joint')
        for role in roles:
            log(f'=== eval {environment} {role} ===')
            started = time.time()
            try:
                payload = run_eval(environment, role, spec)
            except Exception as exc:
                log(f'EVAL FAILED {environment} {role}: {type(exc).__name__}: {exc}')
                fail = {
                    'environment': environment,
                    'role': role,
                    'nan_or_exception': True,
                    'error': f'{type(exc).__name__}: {exc}',
                }
                out = result_path(environment, role)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(fail, indent=2))
                if args.role:
                    raise
                log(f'{environment} {role} stopped; continuing other runs')
                break
            log(
                f'{environment} {role} success_rate='
                f'{payload["success_rate"]:.2f} '
                f'elapsed={payload["elapsed_sec"]:.1f}s '
                f'env_id={payload["env_id"]} '
                f'action_dim={payload["action_dim"]}'
            )
            log(f'finished in {time.time() - started:.1f}s')
        if not args.role:
            summarize([environment])

    if not args.make_specs_only and not args.role:
        summary = summarize(environments)
        log('=== summary ===')
        log(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
