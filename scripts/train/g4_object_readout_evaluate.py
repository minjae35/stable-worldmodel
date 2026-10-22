"""Apply the validation-selected object-local final DINO probe to any G3/G4 model."""

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
    load_model,
    predict_history_bundle,
    rotation_diagnostics,
)
from scripts.train.g4_representation_diagnostic import object_crop4  # noqa: E402
from stable_worldmodel.wm.spatial.g2_eval import (  # noqa: E402
    challenge_mask,
    finite_report,
    probe_metric_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--g3-cache', type=Path, required=True)
    parser.add_argument('--object-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--temporal-encoding', choices=('none', 'sinusoidal'), default='none')
    parser.add_argument('--history', type=int, choices=(1, 3), required=True)
    parser.add_argument('--eval-batch', type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'refusing to overwrite {args.output}')
    if (args.object_cache / 'STATUS').read_text().strip() != 'READY':
        raise RuntimeError('object readout cache is not READY')
    metadata = json.loads((args.object_cache / 'metadata.json').read_text())
    object_data = torch.load(
        args.object_cache / 'pusht_object_readout.pt', map_location='cpu', weights_only=True
    )
    bundle = torch.load(
        args.g3_cache / 'pusht_test.pt', map_location='cpu', weights_only=True
    )
    if len(bundle['target']) != len(object_data['centroids_xy']):
        raise RuntimeError('object readout cohort mismatch')
    model, checkpoint = load_model(args.checkpoint, args.temporal_encoding, 1.0)
    predictions = predict_history_bundle(
        model, bundle, 0, args.history, args.eval_batch, 3089
    )
    centroid = object_data['centroids_xy'].numpy()
    current_feature = object_crop4(bundle['context'][:, -1], centroid[:, 0]).numpy()
    target_feature = object_crop4(bundle['target'], centroid[:, 1]).numpy()
    predicted_feature = object_crop4(predictions['correct'], centroid[:, 1]).numpy()
    shuffled_feature = object_crop4(
        predictions['future_action_shuffle'], centroid[:, 1]
    ).numpy()
    report = {
        'checkpoint': str(args.checkpoint.resolve()),
        'checkpoint_sha256': sha256_file(args.checkpoint),
        'step': int(checkpoint['step']),
        'history': args.history,
        'temporal_encoding': args.temporal_encoding,
        'cache_key': metadata['cache_key'],
        'readout': 'validation-selected L12 object-local crop4; oracle RGB centroid',
        'evaluation_only': True,
        'final_holdout_used': False,
        'targets': {},
    }
    for name, state_name in (
        ('block_xy', 'block_xy_probe'), ('block_angle', 'block_angle_probe')
    ):
        labels = bundle['labels'][name]
        probe = probe_from_state(object_data[state_name])
        current = labels['current'].numpy()
        future = labels['future'].numpy()
        challenge = challenge_mask('pusht', name, current, future)
        subsets = {}
        for subset_name, mask in (
            ('natural_non_challenge', ~challenge),
            ('object_motion_challenge', challenge),
        ):
            metric = probe_metric_report(
                probe,
                true_current_features=current_feature,
                true_future_features=target_feature,
                predicted_features=predicted_feature,
                shuffled_features=shuffled_feature,
                state_current=current,
                state_future=future,
                mask=mask,
            )
            if name == 'block_angle' and metric['n']:
                margin = metric['mean_true_motion'] / max(
                    metric['probe_true_dino_mae'], metric['probe_true_dino_delta_mae']
                )
                metric['probe_motion_margin'] = margin
                metric['eligible_for_model_claim'] = bool(
                    metric['eligible_for_model_claim'] and margin >= 1.5
                )
                metric['claim_gate'] = 'legacy gate and motion/probe-error >= 1.5'
                if not metric['eligible_for_model_claim']:
                    metric['interpretation'] = 'diagnostic_only'
                decoded_current = probe.predict(current_feature)
                decoded_prediction = probe.predict(predicted_feature)
                from stable_worldmodel.wm.spatial.g2_eval import state_delta, state_distance
                true_delta = state_delta(current, future, 'angle')
                pred_delta = state_delta(decoded_current, decoded_prediction, 'angle')
                moving = mask & (np.abs(np.degrees(true_delta)) >= 1.0)
                metric['rotation_diagnostics'] = {
                    'direction_accuracy_ge1deg': float(
                        (np.sign(pred_delta[moving]) == np.sign(true_delta[moving])).mean()
                    ) if bool(moving.any()) else float('nan'),
                    'delta_rotation_mae_deg': float(
                        state_distance(pred_delta[mask], true_delta[mask], 'angle').mean()
                    ),
                    'magnitude_mae_deg': float(np.abs(
                        np.abs(np.degrees(pred_delta[mask]))
                        - np.abs(np.degrees(true_delta[mask]))
                    ).mean()),
                }
            subsets[subset_name] = metric
        report['targets'][name] = subsets
    if not finite_report(report):
        # Direction accuracy can be NaN only for an empty moving subset; common
        # challenge subsets are non-empty, so any NaN here is unexpected.
        raise RuntimeError('non-finite object-local evaluation')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
