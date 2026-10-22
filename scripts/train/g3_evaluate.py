"""Common-cohort evaluation for G3 H1/H3 checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/workspace/stable-worldmodel')
sys.path.insert(0, str(ROOT))

from scripts.train.g2_evaluate_shared import (  # noqa: E402
    probe_from_state,
    token_subset,
)
from scripts.train.g2_spatial import sha256_file  # noqa: E402
from stable_worldmodel.wm.spatial.g2 import ENV_NAMES  # noqa: E402
from stable_worldmodel.wm.spatial.g2_eval import (  # noqa: E402
    challenge_mask,
    finite_report,
    pool4,
    probe_metric_report,
    state_delta,
    state_distance,
    token_error_metrics,
)
from stable_worldmodel.wm.spatial.g3_history import select_history  # noqa: E402
from stable_worldmodel.wm.spatial.model import SpatialDynamics  # noqa: E402

BASE_CACHE = ROOT / 'results/stage3/g2_2x2/shared_eval_cache'
ROTATION_POLICY = ROOT / 'results/stage3/g3_history/rotation_reliability_pretraining.json'


def load_model(
    checkpoint_path: Path,
    temporal_encoding: str = 'none',
    temporal_encoding_scale: float = 1.0,
) -> tuple[SpatialDynamics, dict]:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = SpatialDynamics(
        temporal_encoding=temporal_encoding,
        temporal_encoding_scale=temporal_encoding_scale,
    ).cuda()
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def predict_history_bundle(
    model,
    bundle: dict,
    env_index: int,
    history: int,
    batch_size: int,
    shuffle_seed: int,
) -> dict[str, torch.Tensor]:
    selected = select_history(bundle, history)
    n = len(selected['context'])
    permutation = np.random.default_rng(shuffle_seed).permutation(n)
    outputs: dict[str, list[torch.Tensor]] = {'correct': [], 'future_action_shuffle': []}
    if history == 3:
        outputs.update({
            'history_pair_order_swap': [],
            'historical_actions_zero': [],
            'historical_actions_shuffle': [],
            'current_only': [],
        })
    device = torch.device('cuda:0')
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        take = torch.arange(start, stop)
        shuffled_take = torch.as_tensor(permutation[start:stop], dtype=torch.long)
        context = selected['context'][take].to(device)
        actions = selected['actions'][take].to(device)
        env_id = torch.full((len(take),), env_index, device=device, dtype=torch.long)
        outputs['correct'].append(model(context, actions, env_id)[:, -1].float().cpu())

        shuffled = actions.clone()
        shuffled[:, -1] = selected['actions'][shuffled_take, -1].to(device)
        outputs['future_action_shuffle'].append(
            model(context, shuffled, env_id)[:, -1].float().cpu()
        )
        if history == 3:
            pair_context = context[:, [1, 0, 2]]
            pair_actions = actions[:, [1, 0, 2]]
            outputs['history_pair_order_swap'].append(
                model(pair_context, pair_actions, env_id)[:, -1].float().cpu()
            )
            zero_actions = actions.clone()
            zero_actions[:, :-1] = 0
            outputs['historical_actions_zero'].append(
                model(context, zero_actions, env_id)[:, -1].float().cpu()
            )
            history_shuffle = actions.clone()
            history_shuffle[:, :-1] = selected['actions'][shuffled_take, :-1].to(device)
            outputs['historical_actions_shuffle'].append(
                model(context, history_shuffle, env_id)[:, -1].float().cpu()
            )
            outputs['current_only'].append(
                model(
                    context[:, -1:], actions[:, -1:], env_id,
                    time_indices=torch.tensor([2], device=device)
                    if getattr(model, 'temporal_encoding', 'none') != 'none'
                    else None,
                )[:, -1].float().cpu()
            )
    return {name: torch.cat(parts) for name, parts in outputs.items()}


def rotation_diagnostics(probe, predicted, current_features, labels, mask) -> dict:
    decoded_current = probe.predict(current_features)
    decoded_prediction = probe.predict(pool4(predicted))
    state_current = labels['current'].numpy()
    state_future = labels['future'].numpy()
    true_delta = state_delta(state_current, state_future, 'angle')
    predicted_delta = state_delta(decoded_current, decoded_prediction, 'angle')
    selected_true = true_delta[mask]
    selected_prediction = predicted_delta[mask]
    return {
        'n': int(mask.sum()),
        'direction_accuracy': float(
            (np.sign(selected_prediction) == np.sign(selected_true)).mean()
        ),
        'delta_rotation_mae_deg': float(
            state_distance(selected_prediction, selected_true, 'angle').mean()
        ),
        'motion_magnitude_mae_deg': float(
            np.abs(
                np.abs(np.degrees(selected_prediction))
                - np.abs(np.degrees(selected_true))
            ).mean()
        ),
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    cache_dir: Path,
    output: Path,
    history: int,
    eval_batch: int = 32,
    seed: int = 3072,
    temporal_encoding: str = 'none',
    temporal_encoding_scale: float = 1.0,
) -> dict:
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    metadata = json.loads((cache_dir / 'metadata.json').read_text())
    probe_metadata = json.loads((BASE_CACHE / 'probes.json').read_text())
    probe_states = torch.load(BASE_CACHE / 'probes.pt', map_location='cpu', weights_only=True)
    rotation_policy = json.loads(ROTATION_POLICY.read_text())['claim_policy']
    model, checkpoint = load_model(
        checkpoint_path, temporal_encoding, temporal_encoding_scale
    )
    report = {
        'checkpoint': str(checkpoint_path.resolve()),
        'checkpoint_sha256': sha256_file(checkpoint_path),
        'step': int(checkpoint['step']),
        'history': int(history),
        'temporal_encoding': temporal_encoding,
        'temporal_encoding_scale': temporal_encoding_scale,
        'cache_key': metadata['cache_key'],
        'probe_key': probe_metadata['probe_key'],
        'cohort': 'G3 common history-valid Development target anchors',
        'final_holdout_used': False,
        'environments': {},
        'timing': {},
    }
    for env_index, env in enumerate(ENV_NAMES):
        bundle = torch.load(cache_dir / f'{env}_test.pt', map_location='cpu', weights_only=True)
        torch.cuda.synchronize()
        began = time.perf_counter()
        predictions = predict_history_bundle(
            model, bundle, env_index, history, eval_batch, seed + 17
        )
        torch.cuda.synchronize()
        report['timing'][f'{env}/dynamics_and_ablations_s'] = time.perf_counter() - began
        current = bundle['context'][:, -1]
        target = bundle['target']
        token_metrics = token_error_metrics(
            predictions['correct'], target, current, predictions['future_action_shuffle']
        )
        ablations = {
            name: token_error_metrics(value, target, current, predictions['future_action_shuffle'])[
                'correct_action'
            ]
            for name, value in predictions.items()
            if name not in ('correct', 'future_action_shuffle')
        }
        env_report = {
            'n_test': len(target),
            'token_metrics': token_metrics,
            'history_ablations': ablations,
            'state_metrics': {},
        }
        predicted_pool4 = pool4(predictions['correct'])
        shuffled_pool4 = pool4(predictions['future_action_shuffle'])
        for target_name, labels in bundle['labels'].items():
            probe = probe_from_state(probe_states[env][target_name])
            current_state = labels['current'].numpy()
            future_state = labels['future'].numpy()
            challenge = challenge_mask(env, target_name, current_state, future_state)
            subsets = {}
            for subset_name, mask in (
                ('natural_non_challenge', ~challenge),
                ('object_motion_challenge', challenge),
            ):
                metric = probe_metric_report(
                    probe,
                    true_current_features=bundle['context_pool4'].numpy(),
                    true_future_features=bundle['target_pool4'].numpy(),
                    predicted_features=predicted_pool4,
                    shuffled_features=shuffled_pool4,
                    state_current=current_state,
                    state_future=future_state,
                    mask=mask,
                )
                metric['token_metrics'] = token_subset(
                    predictions['correct'], target, current,
                    predictions['future_action_shuffle'], mask
                )
                if env == 'pusht' and target_name == 'block_angle' and metric['n']:
                    margin = metric['mean_true_motion'] / max(
                        metric['probe_true_dino_mae'], metric['probe_true_dino_delta_mae']
                    )
                    metric['probe_motion_margin'] = margin
                    metric['legacy_probe_gate_pass'] = metric['eligible_for_model_claim']
                    metric['eligible_for_model_claim'] = bool(
                        metric['legacy_probe_gate_pass']
                        and margin >= rotation_policy['minimum_probe_motion_margin_for_claim']
                    )
                    metric['g3_rotation_claim_gate'] = (
                        'legacy gate and motion/probe-error >= '
                        f'{rotation_policy["minimum_probe_motion_margin_for_claim"]}'
                    )
                    if not metric['eligible_for_model_claim']:
                        metric['interpretation'] = 'diagnostic_only: marginal rotation readout'
                    metric['rotation_diagnostics'] = rotation_diagnostics(
                        probe, predictions['correct'],
                        bundle['context_pool4'].numpy(), labels, mask
                    )
                subsets[subset_name] = metric
            env_report['state_metrics'][target_name] = subsets
        report['environments'][env] = env_report
    report['timing']['total_seconds'] = sum(report['timing'].values())
    if not finite_report(report):
        raise RuntimeError('G3 evaluation produced non-finite values')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    del model
    torch.cuda.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--history', type=int, choices=(1, 3), required=True)
    parser.add_argument('--eval-batch', type=int, default=32)
    parser.add_argument(
        '--temporal-encoding', choices=('none', 'sinusoidal'), default='none'
    )
    parser.add_argument('--temporal-encoding-scale', type=float, default=1.0)
    args = parser.parse_args()
    evaluate_checkpoint(
        args.checkpoint.resolve(), args.cache_dir.resolve(), args.output.resolve(),
        args.history, args.eval_batch,
        temporal_encoding=args.temporal_encoding,
        temporal_encoding_scale=args.temporal_encoding_scale,
    )


if __name__ == '__main__':
    main()
