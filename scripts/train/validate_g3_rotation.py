"""Pre-register G3 PushT rotation readout reliability before training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lance
import numpy as np
import torch

ROOT = Path('/workspace/stable-worldmodel')
sys.path.insert(0, str(ROOT))

from scripts.train.g2_evaluate_shared import probe_from_state  # noqa: E402
from scripts.train.g2_spatial import DATASETS, MANIFEST, sha256_file  # noqa: E402
from stable_worldmodel.wm.spatial.g2 import _take, decode_rgb  # noqa: E402
from stable_worldmodel.wm.spatial.g2_eval import state_delta, state_distance, state_motion  # noqa: E402

BASE_CACHE = ROOT / 'results/stage3/g2_2x2/shared_eval_cache'
BLOCK_COLOR = np.asarray([144.0, 162.0, 184.0])


def rgb_orientation(image: np.ndarray) -> float:
    """Estimate directed T-block orientation from its fixed renderer color."""
    mask = np.linalg.norm(image.astype(np.float64) - BLOCK_COLOR, axis=2) < 50.0
    y, x = np.nonzero(mask)
    if len(x) < 30:
        raise ValueError(f'only {len(x)} block-colored pixels')
    points = np.stack([x, y], axis=1).astype(np.float64)
    centered = points - points.mean(axis=0)
    values, vectors = np.linalg.eigh(centered.T @ centered)
    direction = vectors[:, int(np.argmax(values))]
    projection = centered @ direction
    # The crossbar gives the directed T shape a non-zero third moment.
    if np.mean(projection**3) < 0:
        direction = -direction
    return float(np.arctan2(direction[1], direction[0]))


def circular_offset(prediction: np.ndarray, target: np.ndarray) -> float:
    delta = target - prediction
    return float(np.arctan2(np.mean(np.sin(delta)), np.mean(np.cos(delta))))


def angle_error(prediction, target) -> np.ndarray:
    return state_distance(prediction, target, 'angle')


def load_rgb_angles(items: list[dict]) -> np.ndarray:
    rows = [row for item in items for row in (int(item['row']), int(item['row']) + 5)]
    _, by_row = _take(lance.dataset(str(DATASETS['pusht'])), rows, ('pixels',))
    return np.asarray([
        [
            rgb_orientation(decode_rgb(by_row[int(item['row'])]['pixels'])),
            rgb_orientation(decode_rgb(by_row[int(item['row']) + 5]['pixels'])),
        ]
        for item in items
    ])


def split_labels(split: str) -> np.ndarray:
    shard = torch.load(
        BASE_CACHE / f'pusht_{split}.pt', map_location='cpu', weights_only=True
    )
    label = shard['labels']['block_angle']
    return np.stack([label['current'].numpy(), label['future'].numpy()], axis=1)


def rgb_report(items_by_split: dict[str, list[dict]], common_test: list[dict]) -> dict:
    angles = {split: load_rgb_angles(items) for split, items in items_by_split.items()}
    labels = {split: split_labels(split) for split in items_by_split}
    offset = circular_offset(angles['train'].reshape(-1), labels['train'].reshape(-1))
    report = {'fit_split': 'train only', 'fitted_sign': 1, 'fitted_offset_rad': offset}
    for split in ('train', 'val', 'test'):
        prediction = angles[split] + offset
        error = angle_error(prediction, labels[split])
        report[split] = {
            'n_frames': int(error.size),
            'absolute_angle_mae_deg': float(error.mean()),
            'absolute_angle_p95_deg': float(np.quantile(error, 0.95)),
        }
    source_test = items_by_split['test']
    index = {(int(item['episode']), int(item['start'])): i for i, item in enumerate(source_test)}
    take = np.asarray([index[(int(item['episode']), int(item['start']))] for item in common_test])
    prediction = angles['test'][take] + offset
    target = labels['test'][take]
    true_delta = state_delta(target[:, 0], target[:, 1], 'angle')
    predicted_delta = state_delta(prediction[:, 0], prediction[:, 1], 'angle')
    challenge = state_motion(target[:, 0], target[:, 1], 'angle') >= 10.0
    report['g3_common_test'] = {
        'n_windows': int(len(take)),
        'n_rotation_challenge': int(challenge.sum()),
        'absolute_frame_mae_deg': float(angle_error(prediction, target).mean()),
        'delta_mae_deg_all': float(angle_error(predicted_delta, true_delta).mean()),
        'delta_mae_deg_challenge': float(
            angle_error(predicted_delta[challenge], true_delta[challenge]).mean()
        ),
        'direction_accuracy_challenge': float(
            (np.sign(predicted_delta[challenge]) == np.sign(true_delta[challenge])).mean()
        ),
        'magnitude_mae_deg_challenge': float(
            np.abs(
                np.abs(np.degrees(predicted_delta[challenge]))
                - np.abs(np.degrees(true_delta[challenge]))
            ).mean()
        ),
    }
    return report


def nonlinear_probe_report(seed: int = 3100) -> dict:
    def data(split):
        shard = torch.load(
            BASE_CACHE / f'pusht_{split}.pt', map_location='cpu', weights_only=True
        )
        x = torch.cat([shard['context_pool4'], shard['target_pool4']]).float()
        angle = torch.cat([
            shard['labels']['block_angle']['current'],
            shard['labels']['block_angle']['future'],
        ]).float()
        y = torch.stack([torch.sin(angle), torch.cos(angle)], dim=1)
        return x, y, angle

    train_x, train_y, _ = data('train')
    val_x, _, val_angle = data('val')
    test_x, _, test_angle = data('test')
    mean = train_x.mean(0)
    std = train_x.std(0).clamp_min(1e-4)
    device = torch.device('cuda:0')
    train_x = ((train_x - mean) / std).to(device)
    val_x = ((val_x - mean) / std).to(device)
    test_x = ((test_x - mean) / std).to(device)
    train_y = train_y.to(device)
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(6144, 64), torch.nn.GELU(), torch.nn.Linear(64, 2)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    best = None

    def mae(features, target):
        raw = model(features)
        prediction = torch.atan2(raw[:, 0], raw[:, 1])
        delta = torch.atan2(torch.sin(prediction - target), torch.cos(prediction - target))
        return float(torch.rad2deg(delta.abs()).mean())

    for epoch in range(121):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_mae = mae(val_x, val_angle.to(device))
        if best is None or val_mae < best['val_mae_deg']:
            best = {
                'epoch': epoch,
                'val_mae_deg': val_mae,
                'state': {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
            }
    model.load_state_dict(best.pop('state'))
    model.eval()
    with torch.no_grad():
        best['test_mae_deg'] = mae(test_x, test_angle.to(device))
    return {
        'architecture': 'pool4_6144 -> GELU hidden64 -> sin/cos2',
        'selection_split': 'validation only',
        'seed': seed,
        'epochs': 121,
        'optimizer': {'name': 'AdamW', 'lr': 3e-3, 'weight_decay': 1e-3},
        **best,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--common-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'refusing to overwrite {args.output}')
    source = json.loads(MANIFEST.read_text())['pusht']['s0']
    common = json.loads(args.common_manifest.read_text())
    rgb = rgb_report(source, common['environments']['pusht']['test'])
    probe_metadata = json.loads((BASE_CACHE / 'probes.json').read_text())
    ridge_val = probe_metadata['fits']['pusht']['block_angle']['val_mae']
    nonlinear = nonlinear_probe_report()
    selection = 'ridge_sincos' if ridge_val <= nonlinear['val_mae_deg'] else 'nonlinear_mlp_sincos'
    report = {
        'protocol': {
            'state_and_rgb_used_for_evaluation_only': True,
            'episode_disjoint_train_val_test': True,
            'test_used_for_selection': False,
            'circular_error': True,
        },
        'source_manifest_sha256': sha256_file(MANIFEST),
        'common_manifest_sha256': sha256_file(args.common_manifest),
        'rgb_orientation': rgb,
        'probe_comparison': {
            'ridge_sincos_val_mae_deg': ridge_val,
            'nonlinear_mlp_sincos': nonlinear,
            'selected_before_g3_training': selection,
            'selection_rule': 'lower validation circular MAE',
        },
        'claim_policy': {
            'rotation_primary_probe': selection,
            'minimum_probe_motion_margin_for_claim': 1.5,
            'reason': (
                'The legacy gate only barely passes; G3 rotation superiority requires '
                'mean motion / max(true-DINO absolute, delta probe error) >= 1.5.'
            ),
            'otherwise': 'diagnostic_only',
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
