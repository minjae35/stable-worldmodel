"""Episode-paired H3-minus-H1 bootstrap on the shared G3 cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/workspace/stable-worldmodel')
sys.path.insert(0, str(ROOT))

from scripts.train.g2_evaluate_shared import probe_from_state  # noqa: E402
from scripts.train.g2_spatial import sha256_file  # noqa: E402
from scripts.train.g3_evaluate import (  # noqa: E402
    BASE_CACHE,
    load_model,
    predict_history_bundle,
)
from stable_worldmodel.wm.spatial.g2 import ENV_NAMES  # noqa: E402
from stable_worldmodel.wm.spatial.g2_eval import (  # noqa: E402
    challenge_mask,
    pool4,
    state_distance,
)


def episode_means(values: np.ndarray, episodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(episodes)
    return unique, np.asarray([values[episodes == episode].mean() for episode in unique])


def paired_bootstrap(a, b, rng, draws: int) -> dict:
    delta = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    indices = rng.integers(0, len(delta), size=(draws, len(delta)))
    sampled = delta[indices].mean(axis=1)
    return {
        'effect': 'H3_minus_H1; negative favors history 3',
        'n_paired_episodes': int(len(delta)),
        'observed_mean_delta': float(delta.mean()),
        'relative_delta_percent': float(100.0 * delta.mean() / np.asarray(a).mean()),
        'ci95': [float(x) for x in np.quantile(sampled, [0.025, 0.975])],
        'bootstrap_probability_H3_better': float((sampled < 0).mean()),
        'draws': int(draws),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-h1', type=Path, required=True)
    parser.add_argument('--checkpoint-h3', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=53072)
    parser.add_argument('--draws', type=int, default=10000)
    parser.add_argument('--eval-batch', type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'refusing to overwrite {args.output}')
    model_h1, checkpoint_h1 = load_model(args.checkpoint_h1)
    model_h3, checkpoint_h3 = load_model(args.checkpoint_h3)
    probe_states = torch.load(BASE_CACHE / 'probes.pt', map_location='cpu', weights_only=True)
    rng = np.random.default_rng(args.seed)
    report = {
        'effect': 'H3_minus_H1; negative favors history 3',
        'checkpoint_h1_sha256': sha256_file(args.checkpoint_h1),
        'checkpoint_h3_sha256': sha256_file(args.checkpoint_h3),
        'step_h1': int(checkpoint_h1['step']),
        'step_h3': int(checkpoint_h3['step']),
        'bootstrap_seed': args.seed,
        'environments': {},
    }
    for env_index, env in enumerate(ENV_NAMES):
        bundle = torch.load(args.cache_dir / f'{env}_test.pt', map_location='cpu', weights_only=True)
        h1 = predict_history_bundle(
            model_h1, bundle, env_index, 1, args.eval_batch, 3089
        )['correct']
        h3 = predict_history_bundle(
            model_h3, bundle, env_index, 3, args.eval_batch, 3089
        )['correct']
        target = bundle['target'].float()
        episodes = bundle['episode'].numpy()
        env_report = {'feature': {}, 'state': {}}
        for name, error_h1, error_h3 in (
            ('mse', (h1-target).square().mean((1, 2)), (h3-target).square().mean((1, 2))),
            ('mae', (h1-target).abs().mean((1, 2)), (h3-target).abs().mean((1, 2))),
        ):
            unique_h1, values_h1 = episode_means(error_h1.numpy(), episodes)
            unique_h3, values_h3 = episode_means(error_h3.numpy(), episodes)
            if not np.array_equal(unique_h1, unique_h3):
                raise RuntimeError('episode pairing mismatch')
            env_report['feature'][name] = paired_bootstrap(
                values_h1, values_h3, rng, args.draws
            )
        h1_pool = pool4(h1)
        h3_pool = pool4(h3)
        for target_name, labels in bundle['labels'].items():
            probe = probe_from_state(probe_states[env][target_name])
            truth = labels['future'].numpy()
            prediction_h1 = probe.predict(h1_pool)
            prediction_h3 = probe.predict(h3_pool)
            error_h1 = state_distance(prediction_h1, truth, probe.kind)
            error_h3 = state_distance(prediction_h3, truth, probe.kind)
            current = labels['current'].numpy()
            challenge = challenge_mask(env, target_name, current, truth)
            env_report['state'][target_name] = {}
            for subset_name, mask in (
                ('natural_non_challenge', ~challenge),
                ('object_motion_challenge', challenge),
            ):
                if not bool(mask.any()):
                    continue
                common_episodes = episodes[mask]
                unique_h1, values_h1 = episode_means(error_h1[mask], common_episodes)
                unique_h3, values_h3 = episode_means(error_h3[mask], common_episodes)
                if not np.array_equal(unique_h1, unique_h3):
                    raise RuntimeError('state episode pairing mismatch')
                env_report['state'][target_name][subset_name] = paired_bootstrap(
                    values_h1, values_h3, rng, args.draws
                )
        report['environments'][env] = env_report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    del model_h1, model_h3
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
