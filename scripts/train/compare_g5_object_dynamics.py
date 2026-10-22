"""Episode-paired Development analysis for H1 and the four G5 arms."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/workspace/stable-worldmodel')
sys.path.insert(0, str(ROOT))

from scripts.train.g2_evaluate_shared import probe_from_state  # noqa: E402
from scripts.train.g3_evaluate import load_model, predict_history_bundle  # noqa: E402
from scripts.train.g4_representation_diagnostic import object_crop4  # noqa: E402
from stable_worldmodel.wm.spatial.g2 import ENV_NAMES  # noqa: E402
from stable_worldmodel.wm.spatial.g2_eval import (  # noqa: E402
    challenge_mask, state_delta, state_distance,
)

G5 = ROOT / 'results/stage3/g5_object_dynamics'
CACHE = ROOT / 'results/stage3/g3_history/shared_eval_cache'
OBJECT_CACHE = ROOT / 'results/stage3/g4_diagnostics/object_readout_cache_postnorm'
MODELS = {
    'H1_source': ROOT / 'results/stage3/g3_history/H1_uniform_mse/checkpoint_step2000.pt',
    'A_uniform_control': G5 / 'A_uniform_control/checkpoint_step2500.pt',
    'B_motion_balanced': G5 / 'B_motion_balanced/checkpoint_step2500.pt',
    'C_action_sensitive': G5 / 'C_action_sensitive/checkpoint_step2500.pt',
    'D_combined': G5 / 'D_combined/checkpoint_step2500.pt',
}


def per_episode(values, episodes):
    values, episodes = np.asarray(values), np.asarray(episodes)
    ids = np.unique(episodes)
    return ids, np.asarray([values[episodes == episode].mean() for episode in ids])


def paired_bootstrap(a, b, seed=5501, draws=10000):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError('paired arrays differ')
    delta = b - a
    rng = np.random.default_rng(seed)
    samples = delta[rng.integers(0, len(delta), size=(draws, len(delta)))].mean(1)
    return {
        'n_episodes': len(delta), 'mean_B_minus_A': float(delta.mean()),
        'ci95': [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))],
        'probability_B_better_for_lower_metric': float((samples < 0).mean()),
    }


def main(argv=None):
    global G5, CACHE, OBJECT_CACHE, MODELS
    parser = argparse.ArgumentParser()
    parser.add_argument('--g5-root', type=Path, default=G5)
    parser.add_argument('--g3-cache', type=Path, default=CACHE)
    parser.add_argument('--object-cache', type=Path, default=OBJECT_CACHE)
    parser.add_argument(
        '--h1-checkpoint', type=Path,
        default=ROOT / 'results/stage3/g3_history/H1_uniform_mse/checkpoint_step2000.pt',
    )
    args = parser.parse_args(argv)
    G5 = args.g5_root.resolve()
    CACHE = args.g3_cache.resolve()
    OBJECT_CACHE = args.object_cache.resolve()
    MODELS = {
        'H1_source': args.h1_checkpoint.resolve(),
        'A_uniform_control': G5 / 'A_uniform_control/checkpoint_step2500.pt',
        'B_motion_balanced': G5 / 'B_motion_balanced/checkpoint_step2500.pt',
        'C_action_sensitive': G5 / 'C_action_sensitive/checkpoint_step2500.pt',
        'D_combined': G5 / 'D_combined/checkpoint_step2500.pt',
    }
    output = G5 / 'analysis'
    output.mkdir(exist_ok=False)
    object_data = torch.load(
        OBJECT_CACHE / 'pusht_object_readout.pt', map_location='cpu', weights_only=True
    )
    values = {name: {} for name in MODELS}
    episodes_by_env = {}
    for name, checkpoint in MODELS.items():
        model, _ = load_model(checkpoint)
        for env_index, env in enumerate(ENV_NAMES):
            bundle = torch.load(CACHE / f'{env}_test.pt', map_location='cpu', weights_only=True)
            episodes = bundle['episode'].numpy()
            episodes_by_env[env] = episodes
            prediction = predict_history_bundle(model, bundle, env_index, 1, 32, 3089)
            target, current = bundle['target'].float(), bundle['context'][:, -1].float()
            correct = prediction['correct'].float()
            shuffled = prediction['future_action_shuffle'].float()
            mse = (correct - target).square().mean((1, 2)).numpy()
            mae = (correct - target).abs().mean((1, 2)).numpy()
            persistence = (current - target).square().mean((1, 2)).numpy()
            shuffle_mse = (shuffled - target).square().mean((1, 2)).numpy()
            motion = (target - current).square().mean(-1)
            static = motion <= torch.quantile(motion, .25, dim=-1, keepdim=True)
            patch_error = (correct - target).square().mean(-1)
            static_mse = np.asarray([
                float(patch_error[i][static[i]].mean()) for i in range(len(target))
            ])
            values[name][env] = {
                'feature_mse': mse, 'feature_mae': mae,
                'persistence_advantage_mse': persistence - mse,
                'action_shuffle_gap_mse': shuffle_mse - mse,
                'static_quartile_mse': static_mse,
            }
            if env == 'pusht':
                centroid = object_data['centroids_xy'].numpy()
                current_feature = object_crop4(bundle['context'][:, -1], centroid[:, 0]).numpy()
                pred_feature = object_crop4(correct, centroid[:, 1]).numpy()
                for target_name, state_name in (
                    ('block_xy', 'block_xy_probe'), ('block_angle', 'block_angle_probe')
                ):
                    probe = probe_from_state(object_data[state_name])
                    labels = bundle['labels'][target_name]
                    state_now = labels['current'].numpy()
                    state_future = labels['future'].numpy()
                    decoded_now = probe.predict(current_feature)
                    decoded_pred = probe.predict(pred_feature)
                    kind = 'angle' if target_name == 'block_angle' else 'vector'
                    absolute = state_distance(decoded_pred, state_future, kind)
                    persistence_error = state_distance(decoded_now, state_future, kind)
                    predicted_delta = state_delta(decoded_now, decoded_pred, kind)
                    true_delta = state_delta(state_now, state_future, kind)
                    delta_error = state_distance(predicted_delta, true_delta, kind)
                    challenge = challenge_mask('pusht', target_name, state_now, state_future)
                    for subset_name, mask in (
                        ('natural', ~challenge), ('challenge', challenge)
                    ):
                        prefix = f'postnorm_{target_name}_{subset_name}'
                        values[name][env][f'{prefix}_absolute_error'] = absolute[mask]
                        values[name][env][f'{prefix}_persistence_advantage'] = (
                            persistence_error[mask] - absolute[mask]
                        )
                        values[name][env][f'{prefix}_delta_error'] = delta_error[mask]
                        for suffix in (
                            'absolute_error', 'persistence_advantage', 'delta_error'
                        ):
                            values[name][env][f'{prefix}_{suffix}_episodes'] = episodes[mask]
        del model
        torch.cuda.empty_cache()

    summary_rows, bootstraps = [], {}
    for name in MODELS:
        for env in ENV_NAMES:
            episodes = episodes_by_env[env]
            for metric, array in values[name][env].items():
                if metric.endswith('_episodes'):
                    continue
                metric_episodes = values[name][env].get(f'{metric}_episodes', episodes)
                ids, means = per_episode(array, metric_episodes)
                summary_rows.append({
                    'model': name, 'environment': env, 'metric': metric,
                    'n_episodes': len(ids), 'episode_mean': float(means.mean()),
                    'episode_std': float(means.std(ddof=1)) if len(means) > 1 else 0.0,
                })
    for challenger in ('B_motion_balanced', 'C_action_sensitive', 'D_combined'):
        bootstraps[challenger] = {}
        for env in ENV_NAMES:
            bootstraps[challenger][env] = {}
            common_metrics = set(values['A_uniform_control'][env]) & set(values[challenger][env])
            for metric in sorted(common_metrics):
                if metric.endswith('_episodes'):
                    continue
                a_episode = values['A_uniform_control'][env].get(
                    f'{metric}_episodes', episodes_by_env[env]
                )
                b_episode = values[challenger][env].get(
                    f'{metric}_episodes', episodes_by_env[env]
                )
                a_ids, a = per_episode(values['A_uniform_control'][env][metric], a_episode)
                b_ids, b = per_episode(values[challenger][env][metric], b_episode)
                if not np.array_equal(a_ids, b_ids):
                    raise RuntimeError(f'episode pairing differs for {challenger}/{env}/{metric}')
                bootstraps[challenger][env][metric] = paired_bootstrap(a, b)

    with (output / 'episode_metrics.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=summary_rows[0].keys(),
            lineterminator="\n",
        )
        writer.writeheader(); writer.writerows(summary_rows)
    gates = {}
    for name in MODELS:
        common = json.loads((G5 / 'evaluation' / f'{name}_common.json').read_text())
        obj = json.loads((G5 / 'evaluation' / f'{name}_object_postnorm.json').read_text())
        gates[name] = {
            'common': {
                env: {
                    target: {
                        subset: metric['eligible_for_model_claim']
                        for subset, metric in subsets.items()
                    } for target, subsets in report['state_metrics'].items()
                } for env, report in common['environments'].items()
            },
            'postnorm_pusht': {
                target: {
                    subset: metric['eligible_for_model_claim']
                    for subset, metric in subsets.items()
                } for target, subsets in obj['targets'].items()
            },
        }
    report = {
        'cohort': 'G3 Development only; no final holdout use',
        'bootstrap_unit': 'episode (per-episode means, 10000 paired resamples)',
        'comparison_reference': 'A_uniform_control',
        'metric_direction': {
            'errors': 'lower_is_better', 'persistence_advantage': 'higher_is_better',
            'action_shuffle_gap': 'higher_is_more_action_sensitive',
        },
        'paired_bootstrap': bootstraps, 'probe_gates': gates,
    }
    (output / 'paired_bootstrap.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'STATUS').write_text('READY\n')
    (G5 / 'STATUS').write_text('ANALYSIS_READY_REPORT_PENDING\n')


if __name__ == '__main__':
    main()
