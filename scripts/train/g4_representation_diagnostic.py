"""G4-A: episode-disjoint DINOv2 rotation representation diagnostics.

State and RGB object localization are evaluation-only.  This script never
constructs a world-model optimizer or exposes labels to dynamics training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import lance
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/workspace/stable-worldmodel')
sys.path.insert(0, str(ROOT))

from scripts.train.g2_spatial import DATASETS, DINO_WEIGHTS, MANIFEST, sha256_file  # noqa: E402
from scripts.train.validate_g3_rotation import (  # noqa: E402
    BLOCK_COLOR,
    circular_offset,
    rgb_orientation,
)
from stable_worldmodel.wm.spatial.g2 import _take, decode_rgb  # noqa: E402
from stable_worldmodel.wm.spatial.g2_eval import (  # noqa: E402
    ArrayLinearModel,
    RidgeProbe,
    _angle_features,
    _angle_from_features,
    fit_probe,
    state_delta,
    state_distance,
)

AGENT_COLOR = np.asarray([80.0, 123.0, 250.0])
LAYERS = (3, 6, 9, 12)
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def pool(tokens: torch.Tensor, size: int) -> torch.Tensor:
    grid = tokens.float().reshape(len(tokens), 16, 16, 384).permute(0, 3, 1, 2)
    value = F.adaptive_avg_pool2d(grid, (size, size))
    return value.permute(0, 2, 3, 1).reshape(len(tokens), -1)


def block_centroid(image: np.ndarray) -> tuple[float, float]:
    mask = np.linalg.norm(image.astype(np.float64) - BLOCK_COLOR, axis=2) < 50.0
    y, x = np.nonzero(mask)
    if len(x) < 30:
        raise ValueError(f'block mask has only {len(x)} pixels')
    return float(x.mean()), float(y.mean())


def object_crop4(tokens: torch.Tensor, centroids: np.ndarray) -> torch.Tensor:
    """Evaluation-only 6x6-token crop centered from the RGB T-block mask."""
    grid = tokens.float().reshape(len(tokens), 16, 16, 384).permute(0, 3, 1, 2)
    center = torch.as_tensor(centroids, dtype=torch.float32, device=tokens.device)
    center = center / 224.0 * 16.0 - 0.5
    offsets = torch.linspace(-2.5, 2.5, 4, device=tokens.device)
    yy, xx = torch.meshgrid(offsets, offsets, indexing='ij')
    sample_x = center[:, 0, None, None] + xx
    sample_y = center[:, 1, None, None] + yy
    sample_grid = torch.stack([
        2.0 * sample_x / 15.0 - 1.0,
        2.0 * sample_y / 15.0 - 1.0,
    ], dim=-1)
    cropped = F.grid_sample(
        grid, sample_grid, mode='bilinear', padding_mode='border', align_corners=True
    )
    return cropped.permute(0, 2, 3, 1).reshape(len(tokens), -1)


def contact_gap(image: np.ndarray) -> float:
    from scipy.ndimage import distance_transform_edt

    # A 50-level radius makes the antialiased block and agent color spheres
    # touch, falsely reporting contact everywhere.  Radius 30 keeps both masks
    # non-empty on every diagnostic frame and separates their boundaries.
    block = np.linalg.norm(image.astype(np.float64) - BLOCK_COLOR, axis=2) < 30.0
    agent = np.linalg.norm(image.astype(np.float64) - AGENT_COLOR, axis=2) < 30.0
    if int(block.sum()) < 30 or int(agent.sum()) < 10:
        return float('inf')
    distance = distance_transform_edt(~block)
    return float(distance[agent].min())


def load_split(items: list[dict]) -> dict:
    rows = [row for item in items for row in (int(item['row']), int(item['row']) + 5)]
    _, by_row = _take(
        lance.dataset(str(DATASETS['pusht'])), rows, ('pixels', 'state')
    )
    images = []
    state = []
    episodes = []
    for item in items:
        for row in (int(item['row']), int(item['row']) + 5):
            images.append(decode_rgb(by_row[row]['pixels']))
            state.append(np.asarray(by_row[row]['state'], dtype=np.float32))
            episodes.append(int(item['episode']))
    images = np.stack(images)
    state = np.stack(state)
    centroids = np.asarray([block_centroid(image) for image in images], dtype=np.float32)
    gaps = np.asarray([contact_gap(image) for image in images], dtype=np.float32)
    return {
        'images': images,
        'angle': state[:, 4],
        'block_xy': state[:, 2:4],
        'centroids': centroids,
        'contact_gap_px': gaps,
        'episodes': np.asarray(episodes),
    }


@torch.inference_mode()
def encode_split(backbone, data: dict, batch_size: int) -> dict[str, torch.Tensor]:
    outputs: dict[str, list[torch.Tensor]] = {}
    images = data['images']
    for start in range(0, len(images), batch_size):
        pixels = torch.from_numpy(images[start:start + batch_size]).permute(0, 3, 1, 2)
        pixels = pixels.float().div(255.0)
        pixels = ((pixels - MEAN) / STD).cuda()
        result = backbone(
            pixel_values=pixels, output_hidden_states=True, return_dict=True
        )
        hidden_states = result.hidden_states
        centroids = data['centroids'][start:start + batch_size]
        for layer in LAYERS:
            # Hugging Face exposes the last block output in hidden_states[12],
            # before Dinov2Model's final LayerNorm.  SpatialDynamics is trained
            # against last_hidden_state, so the final-layer diagnostic must use
            # that exact post-norm representation.  Intermediate layers remain
            # the raw block outputs because no corresponding terminal norm is
            # applied to them by the frozen encoder.
            hidden = (
                result.last_hidden_state if layer == 12 else hidden_states[layer]
            ).float()
            cls = hidden[:, 0]
            tokens = hidden[:, 1:]
            values = {
                f'L{layer:02d}_cls': cls,
                f'L{layer:02d}_patch_mean': tokens.mean(1),
                f'L{layer:02d}_pool4': pool(tokens, 4),
                f'L{layer:02d}_object_crop4': object_crop4(tokens, centroids),
            }
            if layer == 12:
                values.update({
                    'L12_pool2': pool(tokens, 2),
                    'L12_pool8': pool(tokens, 8),
                    'L12_pool16': tokens.reshape(len(tokens), -1),
                })
            for name, value in values.items():
                outputs.setdefault(name, []).append(value.half().cpu())
    return {name: torch.cat(parts) for name, parts in outputs.items()}


def fit_ridge(train_x, train_y, val_x, val_y, kind):
    """Fast float32 CUDA dual ridge; alpha is selected on validation only."""
    began = time.perf_counter()
    train_x = np.asarray(train_x, dtype=np.float32)
    val_x = np.asarray(val_x, dtype=np.float32)
    train_y = np.asarray(train_y)
    val_y = np.asarray(val_y)
    x_mean = train_x.mean(0, keepdims=True)
    x_std = np.maximum(train_x.std(0, keepdims=True), 1e-6)
    x_train = torch.from_numpy((train_x - x_mean) / x_std).cuda()
    x_val = torch.from_numpy((val_x - x_mean) / x_std).cuda()
    y_np = _angle_features(train_y) if kind == 'angle' else train_y
    if y_np.ndim == 1:
        y_np = y_np[:, None]
    y_train = torch.as_tensor(y_np, dtype=torch.float32, device='cuda')
    x_offset = x_train.mean(0, keepdim=True)
    y_offset = y_train.mean(0, keepdim=True)
    xc = x_train - x_offset
    yc = y_train - y_offset
    kernel = xc @ xc.T
    eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
    eigenvalues = eigenvalues.clamp_min(0.0)
    projected_y = eigenvectors.T @ yc
    rows = []
    best = None
    for alpha in ALPHAS:
        dual = eigenvectors @ (projected_y / (eigenvalues[:, None] + alpha))
        weight = xc.T @ dual
        intercept = y_offset - x_offset @ weight
        raw = (x_val @ weight + intercept).cpu().numpy()
        prediction = _angle_from_features(raw) if kind == 'angle' else raw
        mae = float(state_distance(prediction, val_y, kind).mean())
        rows.append({'alpha': alpha, 'val_mae': mae})
        if best is None or mae < best[0]:
            best = (mae, alpha, weight.detach().cpu().numpy(), intercept.cpu().numpy().reshape(-1))
    del x_train, x_val, y_train, xc, yc, kernel, eigenvectors, projected_y
    torch.cuda.empty_cache()
    model = ArrayLinearModel(best[2], best[3])
    probe = RidgeProbe(model, x_mean, x_std, kind, best[1])
    return probe, {
        'solver': 'torch_cuda_dual_eigh_float32',
        'alpha_search': rows,
        'chosen_alpha': best[1],
        'val_mae': best[0],
        'seconds': time.perf_counter() - began,
    }


def serialize_probe(probe: RidgeProbe) -> dict:
    """Serialize either sklearn Ridge or the local ArrayLinearModel."""
    model = probe.model
    if isinstance(model, ArrayLinearModel):
        weight = np.asarray(model.weight)
        intercept = np.asarray(model.intercept).reshape(-1)
    else:
        coef = np.asarray(model.coef_)
        if coef.ndim == 1:
            coef = coef[None, :]
        weight = np.ascontiguousarray(coef.T)
        intercept = np.asarray(model.intercept_).reshape(-1)
    return {
        'weight': torch.from_numpy(np.ascontiguousarray(weight)),
        'intercept': torch.from_numpy(np.ascontiguousarray(intercept)),
        'x_mean': torch.from_numpy(np.ascontiguousarray(probe.x_mean)),
        'x_std': torch.from_numpy(np.ascontiguousarray(probe.x_std)),
        'kind': probe.kind,
        'alpha': probe.alpha,
    }


def angle_summary(prediction, target) -> dict:
    error = state_distance(prediction, target, 'angle')
    return {
        'n': int(len(error)),
        'mae_deg': float(error.mean()),
        'median_deg': float(np.median(error)),
        'p95_deg': float(np.quantile(error, 0.95)),
    }


def delta_summary(prediction_frames, target_frames, data: dict) -> dict:
    prediction = np.asarray(prediction_frames).reshape(-1, 2)
    target = np.asarray(target_frames).reshape(-1, 2)
    pred_delta = state_delta(prediction[:, 0], prediction[:, 1], 'angle')
    true_delta = state_delta(target[:, 0], target[:, 1], 'angle')
    magnitude = np.abs(np.degrees(true_delta))
    gaps = data['contact_gap_px'].reshape(-1, 2)
    masks = {
        'all': np.ones(len(true_delta), dtype=bool),
        'small_lt5deg': magnitude < 5.0,
        'medium_5to15deg': (magnitude >= 5.0) & (magnitude < 15.0),
        'large_ge15deg': magnitude >= 15.0,
        'no_contact': (gaps[:, 0] > 8.0) & (gaps[:, 1] > 8.0),
        'approach_into_contact': (gaps[:, 0] > 8.0) & (gaps[:, 1] <= 8.0),
        'contact_at_current': gaps[:, 0] <= 8.0,
    }
    report = {}
    for name, mask in masks.items():
        if not bool(mask.any()):
            report[name] = {'n': 0}
            continue
        error = state_distance(pred_delta[mask], true_delta[mask], 'angle')
        moving = magnitude[mask] >= 1.0
        direction = float('nan')
        if bool(moving.any()):
            direction = float(
                (np.sign(pred_delta[mask][moving]) == np.sign(true_delta[mask][moving])).mean()
            )
        report[name] = {
            'n': int(mask.sum()),
            'delta_mae_deg': float(error.mean()),
            'direction_accuracy_ge1deg': direction,
            'magnitude_mae_deg': float(
                np.abs(np.abs(np.degrees(pred_delta[mask])) - magnitude[mask]).mean()
            ),
            'mean_true_motion_deg': float(magnitude[mask].mean()),
        }
    return report


def rgb_baseline(train: dict, test: dict) -> dict:
    train_rgb = np.asarray([rgb_orientation(image) for image in train['images']])
    test_rgb = np.asarray([rgb_orientation(image) for image in test['images']])
    offset = circular_offset(train_rgb, train['angle'])
    prediction = test_rgb + offset
    return {
        'calibration': 'offset fit on train only',
        'absolute': angle_summary(prediction, test['angle']),
        'delta': delta_summary(prediction, test['angle'], test),
    }


def train_mlp(train_x, train_angle, val_x, val_angle, test_x, test_angle) -> dict:
    mean = train_x.mean(0, keepdim=True)
    std = train_x.std(0, keepdim=True).clamp_min(1e-4)
    train_x = ((train_x - mean) / std).float().cuda()
    val_x = ((val_x - mean) / std).float().cuda()
    test_x = ((test_x - mean) / std).float().cuda()
    train_y = torch.from_numpy(
        np.stack([np.sin(train_angle), np.cos(train_angle)], 1)
    ).float().cuda()
    torch.manual_seed(3101)
    model = torch.nn.Sequential(
        torch.nn.Linear(train_x.shape[1], 64), torch.nn.GELU(), torch.nn.Linear(64, 2)
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best = None
    for epoch in range(81):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            raw = model(val_x).float().cpu().numpy()
        prediction = np.arctan2(raw[:, 0], raw[:, 1])
        mae = float(state_distance(prediction, val_angle, 'angle').mean())
        if best is None or mae < best['val_mae_deg']:
            best = {
                'epoch': epoch,
                'val_mae_deg': mae,
                'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
    model.load_state_dict(best.pop('state'))
    model.eval()
    with torch.no_grad():
        raw = model(test_x).float().cpu().numpy()
    prediction = np.arctan2(raw[:, 0], raw[:, 1])
    return {
        'architecture': f'{train_x.shape[1]}-64-2 GELU',
        'selection': 'epoch selected on validation only',
        'seed': 3101,
        **best,
        'test': angle_summary(prediction, test_angle),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--feature-cache', type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'STATUS').write_text('RUNNING\n')

    manifest = json.loads(MANIFEST.read_text())
    items = {split: manifest['pusht']['s0'][split] for split in ('train', 'val', 'test')}
    episodes = {split: set(int(x['episode']) for x in values) for split, values in items.items()}
    if episodes['train'] & episodes['val'] or episodes['train'] & episodes['test'] or episodes['val'] & episodes['test']:
        raise RuntimeError('probe splits are not episode-disjoint')
    data = {split: load_split(values) for split, values in items.items()}

    feature_paths = {}
    if args.feature_cache is not None:
        cache = args.feature_cache.resolve()
        for split in ('train', 'val', 'test'):
            path = cache / f'{split}_features.pt'
            if not path.is_file():
                raise FileNotFoundError(path)
            feature_paths[split] = path
    else:
        from transformers import Dinov2Model
        backbone = Dinov2Model.from_pretrained(DINO_WEIGHTS, local_files_only=True).cuda().eval()
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        for split in ('train', 'val', 'test'):
            split_features = encode_split(backbone, data[split], args.batch_size)
            path = output / f'{split}_features.pt'
            torch.save(split_features, path)
            feature_paths[split] = path
            del split_features
            torch.cuda.empty_cache()
        del backbone
        torch.cuda.empty_cache()
    features = {
        split: torch.load(path, map_location='cpu', weights_only=True)
        for split, path in feature_paths.items()
    }

    candidates = {}
    probes = {}
    for name in sorted(features['train']):
        probe, fit = fit_ridge(
            features['train'][name].float().numpy(), data['train']['angle'],
            features['val'][name].float().numpy(), data['val']['angle'], 'angle',
        )
        candidates[name] = fit
        probes[name] = probe
        print(name, 'val rotation MAE', fit['val_mae'], flush=True)

    best_by_layer = {}
    for layer in LAYERS:
        prefix = f'L{layer:02d}_'
        pool = {name: report for name, report in candidates.items() if name.startswith(prefix)}
        best_by_layer[str(layer)] = min(pool, key=lambda name: pool[name]['val_mae'])
    best_final = min(
        (name for name in candidates if name.startswith('L12_')),
        key=lambda name: candidates[name]['val_mae'],
    )
    best_overall = min(candidates, key=lambda name: candidates[name]['val_mae'])
    selected = list(dict.fromkeys([
        'L12_cls', 'L12_pool4', 'L12_object_crop4', best_final, best_overall,
        *best_by_layer.values(),
    ]))

    test_reports = {}
    selected_probe_states = {}
    for name in selected:
        angle_probe = probes[name]
        test_x = features['test'][name].float().numpy()
        angle_prediction = angle_probe.predict(test_x)
        xy_probe, xy_fit = fit_ridge(
            features['train'][name].float().numpy(), data['train']['block_xy'],
            features['val'][name].float().numpy(), data['val']['block_xy'], 'vector',
        )
        xy_prediction = xy_probe.predict(test_x)
        test_reports[name] = {
            'selected_without_test': True,
            'angle': {
                'fit': candidates[name],
                'absolute_test': angle_summary(angle_prediction, data['test']['angle']),
                'delta_test': delta_summary(angle_prediction, data['test']['angle'], data['test']),
            },
            'block_xy': {
                'fit': xy_fit,
                'test_l2_mae': float(
                    np.linalg.norm(xy_prediction - data['test']['block_xy'], axis=1).mean()
                ),
            },
        }
        selected_probe_states[name] = {
            'angle': serialize_probe(angle_probe), 'block_xy': serialize_probe(xy_probe)
        }

    # A separate paired-frame ridge asks whether change is linearly available even
    # when absolute orientation readout is imperfect.
    pair_name = best_overall
    pair_features = {
        split: features[split][pair_name].float().reshape(-1, 2, features[split][pair_name].shape[-1])
        for split in ('train', 'val', 'test')
    }
    pair_delta = {
        split: state_delta(
            data[split]['angle'].reshape(-1, 2)[:, 0],
            data[split]['angle'].reshape(-1, 2)[:, 1], 'angle'
        ) for split in ('train', 'val', 'test')
    }
    delta_probe, delta_fit = fit_ridge(
        (pair_features['train'][:, 1] - pair_features['train'][:, 0]).numpy(), pair_delta['train'],
        (pair_features['val'][:, 1] - pair_features['val'][:, 0]).numpy(), pair_delta['val'], 'angle',
    )
    delta_prediction = delta_probe.predict(
        (pair_features['test'][:, 1] - pair_features['test'][:, 0]).numpy()
    )
    pair_delta_report = {
        'representation': pair_name,
        'fit': delta_fit,
        'test': angle_summary(delta_prediction, pair_delta['test']),
        'direction_accuracy_ge1deg': float((
            np.sign(delta_prediction[np.abs(np.degrees(pair_delta['test'])) >= 1.0])
            == np.sign(pair_delta['test'][np.abs(np.degrees(pair_delta['test'])) >= 1.0])
        ).mean()),
    }

    mlp = train_mlp(
        features['train'][best_overall].float(), data['train']['angle'],
        features['val'][best_overall].float(), data['val']['angle'],
        features['test'][best_overall].float(), data['test']['angle'],
    )
    report = {
        'status': 'PASS',
        'protocol': {
            'world_model_training_performed': False,
            'state_and_rgb_masks': 'evaluation_only',
            'episode_disjoint': True,
            'final_layer_representation': (
                'Dinov2Model.last_hidden_state (post-final-LayerNorm), exactly '
                'matching the SpatialDynamics prediction target'
            ),
            'intermediate_layer_representation': (
                'raw transformer block hidden state; the frozen encoder applies '
                'no terminal LayerNorm at these depths'
            ),
            'alpha_and_readout_selection': 'validation only',
            'test_use': 'selected candidates and preregistered final-layer baselines only',
            'angle_periodicity': 'directed asymmetric T-block; sin(theta), cos(theta), 2pi circular error',
            'contact_proxy': (
                'RGB agent [80,123,250] and block [144,162,184] masks, '
                'color radius 30, minimum boundary distance <= 8 pixels'
            ),
            'motion_bins_deg': {'small': '<5', 'medium': '[5,15)', 'large': '>=15'},
        },
        'identity': {
            'manifest_sha256': sha256_file(MANIFEST),
            'dino_weights_sha256': sha256_file(DINO_WEIGHTS / 'model.safetensors'),
            'layers': list(LAYERS),
            'feature_cache_source': str(args.feature_cache.resolve()) if args.feature_cache else str(output),
            'split_windows': {name: len(values) for name, values in items.items()},
            'split_episodes': {name: len(values) for name, values in episodes.items()},
        },
        'candidate_validation_only': candidates,
        'selection': {
            'best_by_layer': best_by_layer,
            'best_final_layer': best_final,
            'best_overall': best_overall,
            'test_reported': selected,
        },
        'selected_test_results': test_reports,
        'paired_delta_probe': pair_delta_report,
        'nonlinear_capacity_check': {'representation': best_overall, **mlp},
        'rgb_orientation_test': rgb_baseline(data['train'], data['test']),
        'limitations': [
            'Object-local crop uses an RGB-derived oracle centroid and is evaluation-only.',
            'Probe failure does not prove information absence.',
            'Test is reported once after validation-only selection.',
        ],
    }
    torch.save(
        {
            'readout_selection': report['selection'],
            'probes': selected_probe_states,
            'paired_delta_probe': serialize_probe(delta_probe),
        },
        output / 'selected_probes.pt',
    )
    atomic_json(report, output / 'representation_diagnostic.json')
    (output / 'STATUS').write_text('DONE\n')
    print(json.dumps(report['selection'], indent=2))


if __name__ == '__main__':
    main()
