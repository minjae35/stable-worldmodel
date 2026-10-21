"""Benchmark the G2 evaluation stages and ridge solver equivalence."""

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

from scripts.train.g2_spatial import (
    DATASETS,
    DINO_WEIGHTS,
    G1_CHECKPOINT,
    MANIFEST,
    encode_images,
    normalize_actions,
    predict_bundle,
)
from stable_worldmodel.wm.spatial.g2 import load_label_free_clips
from stable_worldmodel.wm.spatial.g2_eval import (
    fit_probe,
    fit_probe_cuda_dual,
    load_state_labels,
    pool4,
    state_distance,
    token_error_metrics,
)
from stable_worldmodel.wm.spatial.model import FrozenDINOv2, SpatialDynamics


def elapsed(callable_):
    began = time.perf_counter()
    value = callable_()
    return value, time.perf_counter() - began


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--train-items', type=int, default=128)
    parser.add_argument('--val-items', type=int, default=32)
    parser.add_argument('--test-items', type=int, default=32)
    parser.add_argument('--encode-batch', type=int, default=32)
    parser.add_argument('--skip-lsqr', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    import lance

    device = torch.device('cuda:0')
    manifest = json.loads(MANIFEST.read_text())
    base = torch.load(G1_CHECKPOINT, map_location='cpu', weights_only=True)
    encoder = FrozenDINOv2.from_pretrained(str(DINO_WEIGHTS)).to(device)
    dataset = lance.dataset(str(DATASETS['pusht']))
    limits = {
        'train': args.train_items,
        'val': args.val_items,
        'test': args.test_items,
    }
    bundles = {}
    labels = {}
    timings = {}
    for split, limit in limits.items():
        items = manifest['pusht']['s0'][split][:limit]
        raw, timings[f'{split}_lance_decode_s'] = elapsed(
            lambda items=items: load_label_free_clips(
                dataset, items, 'pusht', int(manifest['frameskip'])
            )
        )

        def encode_pair(raw=raw):
            return (
                encode_images(encoder, raw['images_context'], device, args.encode_batch),
                encode_images(encoder, raw['images_target'], device, args.encode_batch),
            )

        (context, target), timings[f'{split}_dino_encode_s'] = elapsed(encode_pair)
        bundles[split] = {
            'context': context,
            'target': target,
            'actions': normalize_actions(raw['actions'], base['normalizer']['pusht']),
        }
        labels[split] = load_state_labels(
            dataset, items, 'pusht', int(manifest['frameskip'])
        )

    train_features = np.concatenate(
        [pool4(bundles['train']['context']), pool4(bundles['train']['target'])]
    )
    val_features = np.concatenate(
        [pool4(bundles['val']['context']), pool4(bundles['val']['target'])]
    )
    train_now, train_future = labels['train']['block_xy']
    val_now, val_future = labels['val']['block_xy']
    train_target = np.concatenate([train_now, train_future])
    val_target = np.concatenate([val_now, val_future])
    solvers = {}
    probes = {}
    solver_jobs = [
        ('cpu_cholesky', lambda: fit_probe(
            train_features, train_target, val_features, val_target, 'vector', 'cholesky'
        )),
        ('cuda_dual_float64', lambda: fit_probe_cuda_dual(
            train_features, train_target, val_features, val_target, 'vector'
        )),
    ]
    if not args.skip_lsqr:
        solver_jobs.insert(0, ('cpu_lsqr', lambda: fit_probe(
            train_features, train_target, val_features, val_target, 'vector', 'lsqr'
        )))
    for name, fit in solver_jobs:
        (probe, fit_report), seconds = elapsed(fit)
        probes[name] = probe
        prediction = probe.predict(val_features)
        solvers[name] = {
            'seconds': seconds,
            'chosen_alpha': fit_report['chosen_alpha'],
            'val_mae': float(state_distance(prediction, val_target, 'vector').mean()),
        }
    reference_name = 'cpu_lsqr' if 'cpu_lsqr' in probes else 'cpu_cholesky'
    reference = probes[reference_name].predict(val_features)
    for name, probe in probes.items():
        delta = np.abs(probe.predict(val_features) - reference)
        solvers[name][f'vs_{reference_name}_max_abs'] = float(delta.max())
        solvers[name][f'vs_{reference_name}_mean_abs'] = float(delta.mean())

    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = SpatialDynamics().to(device)
    model.load_state_dict(payload['model'], strict=True)
    model.eval()
    torch.cuda.synchronize()
    (correct, shuffled, _), timings['dynamics_correct_and_shuffle_s'] = elapsed(
        lambda: predict_bundle(model, bundles['test'], 0, device, 32, 3089)
    )
    torch.cuda.synchronize()
    _, timings['token_metrics_s'] = elapsed(
        lambda: token_error_metrics(
            correct, bundles['test']['target'], bundles['test']['context'], shuffled
        )
    )
    report = {
        'cohort': limits,
        'feature_shape': list(train_features.shape),
        'timings': timings,
        'ridge_solvers': solvers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
