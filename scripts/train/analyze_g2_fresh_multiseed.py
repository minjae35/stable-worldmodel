"""Aggregate four-seed G2 Fresh A/B development evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path('/workspace/stable-worldmodel')
ENVIRONMENTS = ('pusht', 'tworoom', 'cube')
ARMS = {
    'A': 'A_fresh_uniform_mse',
    'B': 'B_fresh_temporal_mse',
}
T95_DF3 = 3.182446305284263


def evaluation_path(multiseed_root: Path, seed: int, arm: str, step: int) -> Path:
    if seed == 3072:
        if step == 500:
            return (
                ROOT / 'results/stage3/g2_fresh_ab' / ARMS[arm]
                / 'evaluation_optimized.json'
            )
        return (
            ROOT / 'results/stage3/g2_fresh_ab_extension_2000'
            / f'{ARMS[arm]}_extension'
            / f'evaluation_development_step{step}.json'
        )
    return (
        multiseed_root / f'seed_{seed}' / ARMS[arm]
        / f'evaluation_development_step{step}.json'
    )


def bootstrap_path(multiseed_root: Path, seed: int, step: int) -> Path:
    if seed == 3072:
        if step == 500:
            return ROOT / 'results/stage3/g2_fresh_ab/development_paired_bootstrap.json'
        return (
            ROOT / 'results/stage3/g2_fresh_ab_extension_2000'
            / f'development_paired_bootstrap_step{step}.json'
        )
    return multiseed_root / f'seed_{seed}' / f'development_paired_bootstrap_step{step}.json'


def across_seed(values: list[float], a_values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    std = float(array.std(ddof=1))
    se = std / np.sqrt(len(array))
    mean = float(array.mean())
    return {
        'effect': 'B_minus_A; negative favors temporal coverage',
        'n_seeds': len(values),
        'mean_delta': mean,
        'mean_relative_delta_percent': float(100.0 * mean / np.mean(a_values)),
        'sample_variance': float(array.var(ddof=1)),
        'sample_std': std,
        'standard_error': float(se),
        'exploratory_t95_ci_df3': [mean - T95_DF3 * se, mean + T95_DF3 * se],
        'min_delta': float(array.min()),
        'max_delta': float(array.max()),
        'n_B_better': int((array < 0).sum()),
        'n_B_worse': int((array > 0).sum()),
        'deltas_by_seed': [float(value) for value in array],
        'warning': 'Only four seeds; interval and direction counts are exploratory.',
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f'refusing to overwrite {path}')
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--multiseed-root', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=(3072, 3073, 3074, 3075))
    parser.add_argument('--steps', type=int, nargs='+', default=(500, 1000, 1500, 2000))
    args = parser.parse_args()
    root = args.multiseed_root.resolve()
    seeds = tuple(args.seeds)
    steps = tuple(args.steps)

    feature_rows: list[dict] = []
    state_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    evaluations: dict[tuple[int, str, int], dict] = {}
    for seed in seeds:
        for step in steps:
            for arm in ARMS:
                path = evaluation_path(root, seed, arm, step)
                report = json.loads(path.read_text())
                evaluations[(seed, arm, step)] = report
                env_mse = []
                env_mae = []
                for env in ENVIRONMENTS:
                    env_report = report['environments'][env]
                    tokens = env_report['token_metrics']
                    mse = float(tokens['correct_action']['mse'])
                    mae = float(tokens['correct_action']['l1'])
                    env_mse.append(mse)
                    env_mae.append(mae)
                    feature_rows.append({
                        'seed': seed, 'step': step, 'arm': arm, 'environment': env,
                        'feature_mse': mse, 'feature_mae': mae,
                        'persistence_mse': tokens['persistence']['mse'],
                        'persistence_mae': tokens['persistence']['l1'],
                        'action_shuffle_mse': tokens['action_shuffle']['mse'],
                        'action_shuffle_mae': tokens['action_shuffle']['l1'],
                    })
                    for target, target_report in env_report['state_metrics'].items():
                        for cohort in ('natural_non_challenge', 'object_motion_challenge'):
                            metric = target_report[cohort]
                            delta = metric['delta_motion']
                            state_rows.append({
                                'seed': seed, 'step': step, 'arm': arm,
                                'environment': env, 'target': target, 'cohort': cohort,
                                'n': metric['n'],
                                'probe_gate_pass': metric['eligible_for_model_claim'],
                                'probe_true_dino_mae': metric['probe_true_dino_mae'],
                                'probe_true_dino_delta_mae': metric['probe_true_dino_delta_mae'],
                                'mean_true_motion': metric['mean_true_motion'],
                                'state_correct_mae': metric['correct_action_mae'],
                                'state_persistence_mae': metric['persistence_mae'],
                                'state_action_shuffle_mae': metric['action_shuffle_mae'],
                                'delta_correct_mae': delta['correct_action_mae'],
                                'delta_persistence_mae': delta['persistence_mae'],
                                'delta_action_shuffle_mae': delta['action_shuffle_mae'],
                                'claim_status': (
                                    'eligible' if metric['eligible_for_model_claim']
                                    else 'diagnostic_only'
                                ),
                            })
                feature_rows.append({
                    'seed': seed, 'step': step, 'arm': arm, 'environment': 'macro',
                    'feature_mse': float(np.mean(env_mse)),
                    'feature_mae': float(np.mean(env_mae)),
                    'persistence_mse': '', 'persistence_mae': '',
                    'action_shuffle_mse': '', 'action_shuffle_mae': '',
                })

            bootstrap = json.loads(bootstrap_path(root, seed, step).read_text())
            for env in ENVIRONMENTS:
                for metric in ('mse', 'mae'):
                    effect = bootstrap['environments'][env][metric]
                    bootstrap_rows.append({
                        'seed': seed, 'step': step, 'environment': env,
                        'metric': metric,
                        'n_paired_episodes': effect['n_paired_episodes'],
                        'observed_B_minus_A': effect['observed_mean_delta'],
                        'relative_delta_percent': effect['relative_delta_percent'],
                        'episode_bootstrap_ci95_low': effect['ci95'][0],
                        'episode_bootstrap_ci95_high': effect['ci95'][1],
                        'episode_bootstrap_probability_B_better': effect[
                            'bootstrap_probability_B_better'
                        ],
                    })

    feature_index = {
        (int(row['seed']), int(row['step']), row['arm'], row['environment']): row
        for row in feature_rows
    }
    seed_summary_rows: list[dict] = []
    across_seed_report: dict[str, dict] = {}
    for step in steps:
        across_seed_report[str(step)] = {}
        for environment in (*ENVIRONMENTS, 'macro'):
            across_seed_report[str(step)][environment] = {}
            for metric in ('feature_mse', 'feature_mae'):
                deltas = []
                a_values = []
                for seed in seeds:
                    a = float(feature_index[(seed, step, 'A', environment)][metric])
                    b = float(feature_index[(seed, step, 'B', environment)][metric])
                    delta = b - a
                    a_values.append(a)
                    deltas.append(delta)
                    seed_summary_rows.append({
                        'seed': seed, 'step': step, 'environment': environment,
                        'metric': metric, 'A': a, 'B': b, 'B_minus_A': delta,
                        'relative_delta_percent': 100.0 * delta / a,
                    })
                across_seed_report[str(step)][environment][metric] = across_seed(
                    deltas, a_values
                )

    persistence_checks = []
    for row in state_rows:
        if (
            row['environment'] == 'pusht'
            and row['target'] in ('block_xy', 'block_angle')
            and row['cohort'] == 'object_motion_challenge'
        ):
            persistence_checks.append({
                'seed': row['seed'], 'step': row['step'], 'arm': row['arm'],
                'target': row['target'], 'probe_gate_pass': row['probe_gate_pass'],
                'correct_mae': row['state_correct_mae'],
                'persistence_mae': row['state_persistence_mae'],
                'correct_beats_persistence': (
                    bool(row['probe_gate_pass'])
                    and float(row['state_correct_mae']) < float(row['state_persistence_mae'])
                ),
                'delta_correct_mae': row['delta_correct_mae'],
                'delta_persistence_mae': row['delta_persistence_mae'],
                'delta_correct_beats_persistence': (
                    bool(row['probe_gate_pass'])
                    and float(row['delta_correct_mae']) < float(row['delta_persistence_mae'])
                ),
            })

    report = {
        'scope': {
            'seeds': list(seeds),
            'steps': list(steps),
            'cohort': 'development only',
            'effect': 'B temporal minus A uniform; negative favors temporal coverage',
            'final_holdout_used': False,
        },
        'across_seed_feature_effects': across_seed_report,
        'episode_bootstrap': {
            'description': (
                'Within each seed/environment, episodes are resampled as paired units. '
                'These intervals do not estimate between-seed variation.'
            ),
            'rows': bootstrap_rows,
        },
        'push_block_vs_persistence': persistence_checks,
        'probe_gate_rule': (
            'State metrics with probe_gate_pass=false are diagnostic_only and excluded '
            'from model-improvement claims.'
        ),
        'uncertainty_separation': (
            'across_seed_feature_effects uses four seed-level paired deltas; '
            'episode_bootstrap uses episodes within one seed. They answer different questions.'
        ),
    }

    outputs = {
        root / 'multiseed_feature_results.csv': feature_rows,
        root / 'multiseed_state_results.csv': state_rows,
        root / 'multiseed_episode_bootstrap.csv': bootstrap_rows,
        root / 'multiseed_seed_differences.csv': seed_summary_rows,
    }
    for path in [*outputs, root / 'multiseed_analysis.json']:
        if path.exists():
            raise FileExistsError(f'refusing to overwrite {path}')
    for path, rows in outputs.items():
        write_csv(path, rows)
    (root / 'multiseed_analysis.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
