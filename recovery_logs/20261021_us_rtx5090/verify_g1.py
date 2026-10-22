#!/usr/bin/env python3
"""Read-only G1 reproduction check on transferred data and checkpoint."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import platform
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.pop('HF_ENDPOINT', None)

ROOT = Path('/workspace/stable-worldmodel')
OUT = ROOT / 'recovery_logs/20261021_us_rtx5090/verification_report.json'
RUN = ROOT / 'results/stage3/g1_spatial/overfit500'
WEIGHTS = ROOT / 'results/stage3/g0_dinov2/weights'
CHECKPOINT = RUN / 'checkpoint_step500.pt'
DATASETS = {
    'pusht': Path('/workspace/datasets/lewm-pusht/train.lance'),
    'tworoom': Path('/workspace/datasets/lewm-tworooms/train.lance'),
    'cube': Path('/workspace/datasets/lewm-cube/train.lance'),
}


def load_training_module():
    path = ROOT / 'scripts/train/g1_spatial.py'
    spec = importlib.util.spec_from_file_location('g1_spatial_verify', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def compare_numbers(expected, actual, prefix=''):
    rows = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            path = f'{prefix}.{key}' if prefix else str(key)
            if key not in expected or key not in actual:
                rows.append({'path': path, 'kind': 'missing'})
            else:
                rows.extend(compare_numbers(expected[key], actual[key], path))
    elif isinstance(expected, list) and isinstance(actual, list):
        for index, (left, right) in enumerate(zip(expected, actual)):
            rows.extend(compare_numbers(left, right, f'{prefix}[{index}]'))
        if len(expected) != len(actual):
            rows.append({'path': prefix, 'kind': 'length'})
    elif (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        left = float(expected)
        right = float(actual)
        abs_diff = abs(left - right)
        scale = max(abs(left), abs(right), 1e-12)
        rows.append(
            {
                'path': prefix,
                'expected': left,
                'actual': right,
                'abs_diff': abs_diff,
                'rel_diff': abs_diff / scale,
            }
        )
    return rows


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite {OUT}')

    import lance
    import transformers

    g1 = load_training_module()
    from stable_worldmodel.wm.spatial.model import FrozenDINOv2, SpatialDynamics

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')

    report = {
        'environment': {
            'python': platform.python_version(),
            'torch': torch.__version__,
            'torch_cuda': torch.version.cuda,
            'transformers': transformers.__version__,
            'lance': getattr(lance, '__version__', 'unknown'),
            'cuda_available': torch.cuda.is_available(),
            'device': torch.cuda.get_device_name(0),
            'capability': list(torch.cuda.get_device_capability(0)),
            'arch_list': torch.cuda.get_arch_list(),
        },
        'artifacts': {
            'checkpoint': str(CHECKPOINT),
            'checkpoint_sha256': sha256_file(CHECKPOINT),
            'dino_weights': str(WEIGHTS / 'model.safetensors'),
            'dino_weights_sha256': sha256_file(WEIGHTS / 'model.safetensors'),
        },
    }
    print(json.dumps(report['environment'], indent=2), flush=True)

    dino_config = json.loads((WEIGHTS / 'config.json').read_text())
    preprocessor = json.loads((WEIGHTS / 'preprocessor_config.json').read_text())
    encoder = FrozenDINOv2.from_pretrained(str(WEIGHTS)).to(device)
    encoder.train(True)
    report['dino'] = {
        'model_type': dino_config['model_type'],
        'hidden_size': dino_config['hidden_size'],
        'patch_size': dino_config['patch_size'],
        'layers': dino_config['num_hidden_layers'],
        'heads': dino_config['num_attention_heads'],
        'image_size_used_by_g1': 224,
        'mean_used_by_g1': list(g1.MEAN),
        'std_used_by_g1': list(g1.STD),
        'preprocessor_crop_size': preprocessor['crop_size'],
        'preprocessor_mean_matches': list(g1.MEAN) == preprocessor['image_mean'],
        'preprocessor_std_matches': list(g1.STD) == preprocessor['image_std'],
        'training': encoder.training,
        'backbone_training': encoder.backbone.training,
        'parameter_count': sum(p.numel() for p in encoder.parameters()),
        'trainable_parameters': sum(p.numel() for p in encoder.parameters() if p.requires_grad),
    }
    print(json.dumps(report['dino'], indent=2), flush=True)

    blob = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    config = json.loads((RUN / 'config.json').read_text())
    model = SpatialDynamics(
        token_dim=config['token_dim'],
        hidden=config['hidden'],
        depth=config['depth'],
        heads=config['heads'],
        mlp_ratio=config['mlp_ratio'],
        grid=config['grid'],
        dropout=config['dropout'],
    )
    incompatible = model.load_state_dict(blob['model'], strict=True)
    model.to(device).eval()
    report['checkpoint'] = {
        'step': int(blob['step']),
        'keys': sorted(blob),
        'state_tensors': len(blob['model']),
        'missing_keys': list(incompatible.missing_keys),
        'unexpected_keys': list(incompatible.unexpected_keys),
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'expected_parameter_count': config['parameters'],
        'architecture': {
            key: config[key]
            for key in ('token_dim', 'hidden', 'depth', 'heads', 'mlp_ratio', 'grid', 'dropout')
        },
    }
    print(json.dumps(report['checkpoint'], indent=2), flush=True)

    saved_episodes = json.loads((RUN / 'episodes.json').read_text())
    saved_norm = json.loads((RUN / 'action_norm.json').read_text())
    data = {}
    report['datasets'] = {}
    for env in g1.ENVS:
        print(f'[data] opening {env}: {DATASETS[env]}', flush=True)
        dataset = lance.dataset(str(DATASETS[env]))
        starts, ends = g1.episode_bounds(dataset)
        schema_names = list(dataset.schema.names)
        report['datasets'][env] = {
            'path': str(DATASETS[env]),
            'rows': int(dataset.count_rows()),
            'episodes': int(len(starts)),
            'schema': schema_names,
            'required_columns_present': all(
                name in schema_names
                for name in ['pixels', 'action', 'episode_idx', *g1.LABELS[env]]
            ),
        }
        data[env] = {}
        for split in ('train', 'test'):
            saved = saved_episodes[env][split]
            clips = []
            for episode, start, motion in zip(
                saved['episode'], saved['start'], saved['motion']
            ):
                ctx_row = int(starts[episode]) + int(start)
                clips.append(
                    {
                        'episode': int(episode),
                        'start': int(start),
                        'motion': float(motion),
                        'ctx_row': ctx_row,
                        'tgt_row': ctx_row + g1.FRAMESKIP,
                    }
                )
            part = g1.load_split(dataset, clips, env)
            mean = np.asarray(saved_norm[env]['mean'], dtype=np.float32)
            std = np.asarray(saved_norm[env]['std'], dtype=np.float32)
            part['action'] = g1.apply_action_normalizer(part['action'], mean, std)
            report['datasets'][env][split] = {
                'samples': int(len(part['episode'])),
                'image_context_shape': list(part['images_ctx'].shape),
                'image_target_shape': list(part['images_tgt'].shape),
                'image_dtype': str(part['images_ctx'].dtype),
                'action_shape': list(part['action'].shape),
                'action_dtype': str(part['action'].dtype),
                'finite_actions': bool(np.isfinite(part['action']).all()),
            }
            part['ctx'] = g1.encode_images(encoder, part['images_ctx'], device)
            part['tgt'] = g1.encode_images(encoder, part['images_tgt'], device)
            part['act'] = torch.from_numpy(np.ascontiguousarray(part['action']))
            del part['images_ctx'], part['images_tgt']
            data[env][split] = part
        del dataset

    report['forward'] = {}
    with torch.inference_mode():
        for env_index, env in enumerate(g1.ENVS):
            part = data[env]['test']
            tokens = part['ctx'][:1, None].to(device)
            actions = part['act'][:1, None].to(device)
            env_id = torch.tensor([env_index], device=device, dtype=torch.long)
            output = model(tokens, actions, env_id)
            report['forward'][env] = {
                'input_tokens_shape': list(tokens.shape),
                'input_action_shape': list(actions.shape),
                'output_shape': list(output.shape),
                'device': str(output.device),
                'dtype': str(output.dtype),
                'nan_count': int(torch.isnan(output).sum()),
                'inf_count': int(torch.isinf(output).sum()),
                'finite': bool(torch.isfinite(output).all()),
                'mean': float(output.mean()),
                'std': float(output.std()),
            }
            print(f'[forward] {env}: {report["forward"][env]}', flush=True)

    probes = {}
    for env in g1.ENVS:
        train = data[env]['train']
        features = np.concatenate(
            [g1.pool4(train['ctx']), g1.pool4(train['tgt'])], axis=0
        )
        labels = np.concatenate(
            [
                g1.label_matrix(env, train['labels_ctx']),
                g1.label_matrix(env, train['labels_tgt']),
            ],
            axis=0,
        )
        probes[env] = g1.fit_probe(features, labels)

    shuffle_perm = {
        env: {
            split: np.random.default_rng(g1.SEED + 17).permutation(
                len(data[env][split]['ctx'])
            )
            for split in ('train', 'test')
        }
        for env in g1.ENVS
    }
    reproduced = {'step': 500}
    for env_index, env in enumerate(g1.ENVS):
        reproduced[env] = {}
        for split in ('train', 'test'):
            reproduced[env][split] = g1.evaluate_split(
                model,
                data[env][split],
                env,
                env_index,
                probes[env],
                shuffle_perm[env][split],
            )
            print(f'[eval] {env}/{split} complete', flush=True)

    expected = json.loads((RUN / 'eval.json').read_text())[-1]
    rows = compare_numbers(expected, reproduced)
    numeric = [row for row in rows if 'abs_diff' in row]
    numeric.sort(key=lambda row: row['abs_diff'], reverse=True)
    report['evaluation'] = {
        'numeric_values_compared': len(numeric),
        'max_abs_diff': max((row['abs_diff'] for row in numeric), default=0.0),
        'max_rel_diff': max((row['rel_diff'] for row in numeric), default=0.0),
        'values_abs_diff_gt_1e-5': sum(row['abs_diff'] > 1e-5 for row in numeric),
        'values_abs_diff_gt_1e-4': sum(row['abs_diff'] > 1e-4 for row in numeric),
        'largest_differences': numeric[:20],
        'reproduced_step500': reproduced,
    }

    normalizer_match = True
    for env in g1.ENVS:
        for key in ('mean', 'std'):
            old = np.asarray(saved_norm[env][key], dtype=np.float32)
            new = blob['normalizer'][env][key].cpu().numpy()
            normalizer_match &= bool(np.array_equal(old, new))
    report['checkpoint']['normalizer_matches_action_norm_json'] = normalizer_match

    if report['artifacts']['dino_weights_sha256'] != g1.WEIGHT_SHA:
        raise RuntimeError('DINOv2 SHA-256 does not match G1 source constant')
    if not all(item['finite'] for item in report['forward'].values()):
        raise RuntimeError('non-finite G1 forward output')
    if report['checkpoint']['parameter_count'] != config['parameters']:
        raise RuntimeError('checkpoint architecture parameter count mismatch')
    if any(not item['required_columns_present'] for item in report['datasets'].values()):
        raise RuntimeError('required dataset column missing')
    if not normalizer_match:
        raise RuntimeError('checkpoint normalizer mismatch')
    if not math.isfinite(report['evaluation']['max_abs_diff']):
        raise RuntimeError('non-finite evaluation comparison')

    OUT.write_text(json.dumps(g1.jsonable(report), indent=2) + '\n')
    print(
        '[done] evaluation max_abs_diff='
        f'{report["evaluation"]["max_abs_diff"]:.9g}',
        flush=True,
    )


if __name__ == '__main__':
    main()
