"""Aggregate G3 H1/H3 stepwise evaluations into durable tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ENVIRONMENTS = ('pusht', 'tworoom', 'cube')
ARMS = {'H1': 'H1_uniform_mse', 'H3': 'H3_uniform_mse'}


def write_csv(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f'refusing to overwrite {path}')
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--steps', type=int, nargs='+', default=(500, 1000, 1500, 2000))
    args = parser.parse_args()
    root = args.root.resolve()
    feature_rows = []
    state_rows = []
    ablation_rows = []
    reports = {}
    for step in args.steps:
        for arm, directory in ARMS.items():
            path = root / directory / f'evaluation_development_step{step}.json'
            report = json.loads(path.read_text())
            reports[(step, arm)] = report
            mse_values = []
            mae_values = []
            for env in ENVIRONMENTS:
                env_report = report['environments'][env]
                token = env_report['token_metrics']
                row = {
                    'step': step, 'arm': arm, 'environment': env,
                    'feature_mse': token['correct_action']['mse'],
                    'feature_mae': token['correct_action']['l1'],
                    'persistence_mse': token['persistence']['mse'],
                    'persistence_mae': token['persistence']['l1'],
                    'action_shuffle_mse': token['action_shuffle']['mse'],
                    'action_shuffle_mae': token['action_shuffle']['l1'],
                    'moving_quartile_mse': token['correct_action']['moving_quartile_mse'],
                    'motion_weighted_mse': token['correct_action']['motion_weighted_mse'],
                }
                feature_rows.append(row)
                mse_values.append(float(row['feature_mse']))
                mae_values.append(float(row['feature_mae']))
                for name, metric in env_report['history_ablations'].items():
                    ablation_rows.append({
                        'step': step, 'arm': arm, 'environment': env,
                        'ablation': name, 'mse': metric['mse'], 'mae': metric['l1'],
                        'moving_quartile_mse': metric['moving_quartile_mse'],
                        'motion_weighted_mse': metric['motion_weighted_mse'],
                    })
                for target, subsets in env_report['state_metrics'].items():
                    for subset, metric in subsets.items():
                        rotation = metric.get('rotation_diagnostics', {})
                        state_rows.append({
                            'step': step, 'arm': arm, 'environment': env,
                            'target': target, 'subset': subset, 'n': metric['n'],
                            'eligible_for_model_claim': metric['eligible_for_model_claim'],
                            'legacy_probe_gate_pass': metric.get(
                                'legacy_probe_gate_pass', metric['eligible_for_model_claim']
                            ),
                            'probe_motion_margin': metric.get('probe_motion_margin', ''),
                            'probe_true_dino_mae': metric['probe_true_dino_mae'],
                            'probe_true_dino_delta_mae': metric['probe_true_dino_delta_mae'],
                            'mean_true_motion': metric['mean_true_motion'],
                            'correct_action_mae': metric['correct_action_mae'],
                            'persistence_mae': metric['persistence_mae'],
                            'action_shuffle_mae': metric['action_shuffle_mae'],
                            'delta_correct_mae': metric['delta_motion']['correct_action_mae'],
                            'delta_persistence_mae': metric['delta_motion']['persistence_mae'],
                            'delta_action_shuffle_mae': metric['delta_motion']['action_shuffle_mae'],
                            'rotation_direction_accuracy': rotation.get('direction_accuracy', ''),
                            'rotation_delta_mae_deg': rotation.get('delta_rotation_mae_deg', ''),
                            'rotation_magnitude_mae_deg': rotation.get(
                                'motion_magnitude_mae_deg', ''
                            ),
                            'claim_status': (
                                'eligible' if metric['eligible_for_model_claim']
                                else 'diagnostic_only'
                            ),
                        })
            feature_rows.append({
                'step': step, 'arm': arm, 'environment': 'macro',
                'feature_mse': float(np.mean(mse_values)),
                'feature_mae': float(np.mean(mae_values)),
                'persistence_mse': '', 'persistence_mae': '',
                'action_shuffle_mse': '', 'action_shuffle_mae': '',
                'moving_quartile_mse': '', 'motion_weighted_mse': '',
            })

    paired_rows = []
    for step in args.steps:
        paired = json.loads((root / f'development_paired_bootstrap_step{step}.json').read_text())
        for env in ENVIRONMENTS:
            for metric, result in paired['environments'][env]['feature'].items():
                paired_rows.append({
                    'step': step, 'environment': env, 'target': 'feature',
                    'subset': 'all', 'metric': metric,
                    'H3_minus_H1': result['observed_mean_delta'],
                    'relative_delta_percent': result['relative_delta_percent'],
                    'ci95_low': result['ci95'][0], 'ci95_high': result['ci95'][1],
                    'probability_H3_better': result['bootstrap_probability_H3_better'],
                    'n_paired_episodes': result['n_paired_episodes'],
                })
            for target, subsets in paired['environments'][env]['state'].items():
                for subset, result in subsets.items():
                    paired_rows.append({
                        'step': step, 'environment': env, 'target': target,
                        'subset': subset, 'metric': 'state_mae',
                        'H3_minus_H1': result['observed_mean_delta'],
                        'relative_delta_percent': result['relative_delta_percent'],
                        'ci95_low': result['ci95'][0], 'ci95_high': result['ci95'][1],
                        'probability_H3_better': result['bootstrap_probability_H3_better'],
                        'n_paired_episodes': result['n_paired_episodes'],
                    })

    feature_index = {
        (int(row['step']), row['arm'], row['environment']): row for row in feature_rows
    }
    step_summary = {}
    for step in args.steps:
        step_summary[str(step)] = {}
        for env in (*ENVIRONMENTS, 'macro'):
            h1 = feature_index[(step, 'H1', env)]
            h3 = feature_index[(step, 'H3', env)]
            step_summary[str(step)][env] = {
                'h1_mse': h1['feature_mse'], 'h3_mse': h3['feature_mse'],
                'mse_relative_delta_percent': 100.0 * (
                    float(h3['feature_mse']) - float(h1['feature_mse'])
                ) / float(h1['feature_mse']),
                'h1_mae': h1['feature_mae'], 'h3_mae': h3['feature_mae'],
                'mae_relative_delta_percent': 100.0 * (
                    float(h3['feature_mae']) - float(h1['feature_mae'])
                ) / float(h1['feature_mae']),
            }
    analysis = {
        'scope': {
            'seed': 3072, 'single_seed': True,
            'steps': list(args.steps), 'effect': 'H3 minus H1; negative favors H3',
            'final_holdout_used': False,
        },
        'stepwise_feature_summary': step_summary,
        'rotation_claim_policy': json.loads(
            (root / 'rotation_reliability_pretraining.json').read_text()
        )['claim_policy'],
        'warning': 'Single-seed controlled experiment; initialization variance is unknown.',
    }
    outputs = {
        root / 'g3_feature_results.csv': feature_rows,
        root / 'g3_state_results.csv': state_rows,
        root / 'g3_history_ablation_results.csv': ablation_rows,
        root / 'g3_paired_bootstrap_results.csv': paired_rows,
    }
    for path in [*outputs, root / 'g3_analysis.json']:
        if path.exists():
            raise FileExistsError(f'refusing to overwrite {path}')
    for path, rows in outputs.items():
        write_csv(path, rows)
    (root / 'g3_analysis.json').write_text(json.dumps(analysis, indent=2) + '\n')


if __name__ == '__main__':
    main()
