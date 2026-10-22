"""Cached, shared evaluation for Stage-3 G2 checkpoints.

The immutable true-DINO cohort and evaluation-only ridge probes are prepared
once. Each arm then runs only its own dynamics forward passes and metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/workspace/stable-worldmodel')
sys.path.insert(0, str(ROOT))

from scripts.train.g2_spatial import (  # noqa: E402
    DATASETS,
    DINO_WEIGHTS,
    G1_CHECKPOINT,
    MANIFEST,
    encode_images,
    normalize_actions,
    predict_bundle,
    sha256_file,
)
from stable_worldmodel.wm.spatial.g2 import ENV_NAMES, load_label_free_clips  # noqa: E402
from stable_worldmodel.wm.spatial.g2 import load_label_free_actions  # noqa: E402
from stable_worldmodel.wm.spatial.g2_eval import (  # noqa: E402
    ArrayLinearModel,
    RidgeProbe,
    challenge_mask,
    finite_report,
    fit_probe,
    load_state_labels,
    pool4,
    probe_metric_report,
    token_error_metrics,
)
from stable_worldmodel.wm.spatial.model import FrozenDINOv2, SpatialDynamics  # noqa: E402
from stable_worldmodel.wm.spatial.g2_fresh import torch_normalizers  # noqa: E402

CACHE_SCHEMA = 1
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def canonical_hash(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_torch_save(value, path: Path):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json(value: dict, path: Path):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def dataset_identity(path: Path) -> dict:
    import lance

    dataset = lance.dataset(str(path))
    return {
        'path': str(path.resolve()),
        'version': int(dataset.version),
        'rows': int(dataset.count_rows()),
        'schema': str(dataset.schema),
    }


def configure_determinism(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def expected_cache_metadata(limits: dict[str, int], seed: int) -> dict:
    identity = {
        'schema': CACHE_SCHEMA,
        'manifest_sha256': sha256_file(MANIFEST),
        'dino_weights_sha256': sha256_file(DINO_WEIGHTS / 'model.safetensors'),
        'g1_checkpoint_sha256': sha256_file(G1_CHECKPOINT),
        'sampling': 's0',
        'limits': limits,
        'seed': seed,
        'cuda_math': {
            'matmul_allow_tf32': False,
            'cudnn_allow_tf32': False,
            'cudnn_benchmark': False,
        },
        'preprocessing': {
            'image_size': [224, 224],
            'rgb': True,
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'token_shape': [256, 384],
            'pooling': 'adaptive_avg_pool2d_4x4',
        },
        'packages': {
            'torch': torch.__version__,
            'transformers': importlib.metadata.version('transformers'),
            'lance': importlib.metadata.version('pylance'),
        },
        'datasets': {env: dataset_identity(DATASETS[env]) for env in ENV_NAMES},
    }
    return {'cache_key': canonical_hash(identity), 'identity': identity}


def cache_items(manifest: dict, env: str, split: str, limit: int) -> list[dict]:
    items = list(manifest[env]['s0'][split])
    return items if limit <= 0 else items[:limit]


def prepare_feature_cache(
    cache_dir: Path, limits: dict[str, int], encode_batch: int, seed: int
):
    expected = expected_cache_metadata(limits, seed)
    metadata_path = cache_dir / 'metadata.json'
    if metadata_path.is_file():
        observed = json.loads(metadata_path.read_text())
        if observed['cache_key'] != expected['cache_key']:
            raise ValueError('existing evaluation cache key does not match requested cohort')
        if (cache_dir / 'STATUS').read_text().strip() != 'READY':
            raise RuntimeError('existing evaluation cache is incomplete')
        return observed
    if cache_dir.exists():
        raise FileExistsError(f'refusing incomplete/unidentified cache {cache_dir}')

    cache_dir.mkdir(parents=True, exist_ok=False)
    (cache_dir / 'STATUS').write_text('BUILDING\n')
    manifest = json.loads(MANIFEST.read_text())
    checkpoint = torch.load(G1_CHECKPOINT, map_location='cpu', weights_only=True)
    device = torch.device('cuda:0')
    encoder = FrozenDINOv2.from_pretrained(str(DINO_WEIGHTS)).to(device)
    timing = {}
    import lance

    for env in ENV_NAMES:
        dataset = lance.dataset(str(DATASETS[env]))
        for split in ('train', 'val', 'test'):
            items = cache_items(manifest, env, split, limits[split])
            began = time.perf_counter()
            raw = load_label_free_clips(
                dataset, items, env, int(manifest['frameskip'])
            )
            timing[f'{env}/{split}/lance_decode_s'] = time.perf_counter() - began
            began = time.perf_counter()
            context = encode_images(encoder, raw['images_context'], device, encode_batch)
            target = encode_images(encoder, raw['images_target'], device, encode_batch)
            timing[f'{env}/{split}/dino_encode_s'] = time.perf_counter() - began
            labels = load_state_labels(
                dataset, items, env, int(manifest['frameskip'])
            )
            shard = {
                'context': context,
                'target': target,
                'context_pool4': torch.from_numpy(pool4(context)),
                'target_pool4': torch.from_numpy(pool4(target)),
                'actions': normalize_actions(raw['actions'], checkpoint['normalizer'][env]),
                'episode': torch.as_tensor(raw['episode'], dtype=torch.long),
                'start': torch.as_tensor(raw['start'], dtype=torch.long),
                'labels': {
                    name: {
                        'current': torch.from_numpy(np.asarray(current)),
                        'future': torch.from_numpy(np.asarray(future)),
                    }
                    for name, (current, future) in labels.items()
                },
            }
            atomic_torch_save(shard, cache_dir / f'{env}_{split}.pt')
    metadata = {**expected, 'timing': timing}
    atomic_json(metadata, metadata_path)
    (cache_dir / 'STATUS').write_text('READY\n')
    return metadata


def load_shard(cache_dir: Path, env: str, split: str) -> dict:
    return torch.load(
        cache_dir / f'{env}_{split}.pt', map_location='cpu', weights_only=True
    )


def probe_state(probe: RidgeProbe) -> dict:
    model = probe.model
    coef = np.asarray(model.coef_)
    if coef.ndim == 1:
        coef = coef[None, :]
    return {
        'weight': torch.from_numpy(np.ascontiguousarray(coef.T)),
        'intercept': torch.from_numpy(
            np.ascontiguousarray(np.asarray(model.intercept_).reshape(-1))
        ),
        'x_mean': torch.from_numpy(np.ascontiguousarray(probe.x_mean)),
        'x_std': torch.from_numpy(np.ascontiguousarray(probe.x_std)),
        'kind': probe.kind,
        'alpha': probe.alpha,
    }


def probe_from_state(state: dict) -> RidgeProbe:
    model = ArrayLinearModel(
        state['weight'].numpy(), state['intercept'].numpy()
    )
    return RidgeProbe(
        model,
        state['x_mean'].numpy(),
        state['x_std'].numpy(),
        state['kind'],
        float(state['alpha']),
    )


def prepare_probe_cache(cache_dir: Path, feature_metadata: dict):
    probe_path = cache_dir / 'probes.pt'
    report_path = cache_dir / 'probes.json'
    identity = {
        'feature_cache_key': feature_metadata['cache_key'],
        'solver': 'sklearn_cholesky',
        'alphas': list(ALPHAS),
        'alpha_selection': 'validation_only',
        'targets': 'environment_state_targets',
    }
    probe_key = canonical_hash(identity)
    if probe_path.is_file() and report_path.is_file():
        report = json.loads(report_path.read_text())
        if report['probe_key'] != probe_key:
            raise ValueError('existing probe cache key mismatch')
        return report
    if probe_path.exists() or report_path.exists():
        raise FileExistsError('partial probe cache exists; refusing overwrite')

    states = {}
    fits = {}
    total_began = time.perf_counter()
    for env in ENV_NAMES:
        train = load_shard(cache_dir, env, 'train')
        val = load_shard(cache_dir, env, 'val')
        train_features = np.concatenate(
            [train['context_pool4'].numpy(), train['target_pool4'].numpy()]
        )
        val_features = np.concatenate(
            [val['context_pool4'].numpy(), val['target_pool4'].numpy()]
        )
        states[env] = {}
        fits[env] = {}
        for name, train_labels in train['labels'].items():
            val_labels = val['labels'][name]
            train_target = np.concatenate(
                [train_labels['current'].numpy(), train_labels['future'].numpy()]
            )
            val_target = np.concatenate(
                [val_labels['current'].numpy(), val_labels['future'].numpy()]
            )
            kind = 'angle' if name == 'block_angle' else 'vector'
            began = time.perf_counter()
            probe, fit = fit_probe(
                train_features,
                train_target,
                val_features,
                val_target,
                kind,
                solver='cholesky',
            )
            fit['seconds'] = time.perf_counter() - began
            states[env][name] = probe_state(probe)
            fits[env][name] = fit
    atomic_torch_save(states, probe_path)
    report = {
        'probe_key': probe_key,
        'identity': identity,
        'fits': fits,
        'total_seconds': time.perf_counter() - total_began,
    }
    atomic_json(report, report_path)
    return report


def token_subset(correct, target, context, shuffled, mask: np.ndarray) -> dict:
    take = torch.as_tensor(mask, dtype=torch.bool)
    if not bool(take.any()):
        return {'n': 0, 'reason': 'empty subset'}
    return {
        'n': int(take.sum()),
        **token_error_metrics(
            correct[take], target[take], context[take], shuffled[take]
        ),
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    cache_dir: Path,
    output: Path,
    seed: int,
    eval_batch: int,
    action_normalizer_path: Path | None = None,
):
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    feature_metadata = json.loads((cache_dir / 'metadata.json').read_text())
    probe_metadata = json.loads((cache_dir / 'probes.json').read_text())
    probe_states = torch.load(
        cache_dir / 'probes.pt', map_location='cpu', weights_only=True
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = SpatialDynamics().cuda()
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    report = {
        'checkpoint': str(checkpoint_path.resolve()),
        'checkpoint_sha256': sha256_file(checkpoint_path),
        'step': int(checkpoint['step']),
        'feature_cache_key': feature_metadata['cache_key'],
        'probe_key': probe_metadata['probe_key'],
        'probe_fit_reused_across_arms': True,
        'natural_sampling': 'manifest s0 test for every arm',
        'environments': {},
        'timing': {},
    }
    action_normalizers = None
    if action_normalizer_path is not None:
        normalizer_artifact = json.loads(action_normalizer_path.read_text())
        action_normalizers = torch_normalizers(normalizer_artifact)
        report['action_normalizer'] = {
            'path': str(action_normalizer_path.resolve()),
            'sha256': sha256_file(action_normalizer_path),
            'key': normalizer_artifact['key'],
            'source': 'direct label-free Lance actions from cached cohort rows',
        }
        manifest = json.loads(MANIFEST.read_text())
        import lance
    device = torch.device('cuda:0')
    for env_index, env in enumerate(ENV_NAMES):
        test = load_shard(cache_dir, env, 'test')
        if action_normalizers is not None:
            limit = int(feature_metadata['identity']['limits']['test'])
            items = cache_items(manifest, env, 'test', limit)
            raw_actions = load_label_free_actions(
                lance.dataset(str(DATASETS[env])), items, env,
                int(manifest['frameskip']),
            )
            if not np.array_equal(raw_actions['episode'], test['episode'].numpy()):
                raise RuntimeError(f'{env} cached episode order mismatch')
            if not np.array_equal(raw_actions['start'], test['start'].numpy()):
                raise RuntimeError(f'{env} cached start order mismatch')
            test = dict(test)
            test['actions'] = normalize_actions(
                raw_actions['actions'], action_normalizers[env]
            )
        torch.cuda.synchronize()
        began = time.perf_counter()
        correct, shuffled, _ = predict_bundle(
            model, test, env_index, device, eval_batch, seed + 17
        )
        torch.cuda.synchronize()
        report['timing'][f'{env}/dynamics_forward_s'] = time.perf_counter() - began
        began = time.perf_counter()
        common_tokens = token_error_metrics(
            correct, test['target'], test['context'], shuffled
        )
        predicted_pool4 = pool4(correct)
        shuffled_pool4 = pool4(shuffled)
        report['timing'][f'{env}/pool_and_token_metrics_s'] = (
            time.perf_counter() - began
        )
        env_report = {
            'n_test': len(test['context']),
            'token_metrics': common_tokens,
            'state_metrics': {},
        }
        for target_name, label in test['labels'].items():
            probe = probe_from_state(probe_states[env][target_name])
            current = label['current'].numpy()
            future = label['future'].numpy()
            features = {
                'true_current_features': test['context_pool4'].numpy(),
                'true_future_features': test['target_pool4'].numpy(),
                'predicted_features': predicted_pool4,
                'shuffled_features': shuffled_pool4,
                'state_current': current,
                'state_future': future,
            }
            challenge = challenge_mask(env, target_name, current, future)
            natural = ~challenge
            env_report['state_metrics'][target_name] = {
                'probe_fit': probe_metadata['fits'][env][target_name],
                'natural_non_challenge': probe_metric_report(
                    probe, mask=natural, **features
                ),
                'object_motion_challenge': probe_metric_report(
                    probe, mask=challenge, **features
                ),
                'token_subsets': {
                    'natural_non_challenge': token_subset(
                        correct,
                        test['target'],
                        test['context'],
                        shuffled,
                        natural,
                    ),
                    'object_motion_challenge': token_subset(
                        correct,
                        test['target'],
                        test['context'],
                        shuffled,
                        challenge,
                    ),
                },
            }
        report['environments'][env] = env_report
    if not finite_report(report):
        raise RuntimeError('cached evaluation produced non-finite values')
    report['timing']['total_arm_seconds'] = sum(report['timing'].values())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--encode-batch', type=int, default=32)
    parser.add_argument('--eval-batch', type=int, default=32)
    parser.add_argument('--seed', type=int, default=3072)
    parser.add_argument('--train-items', type=int, default=0)
    parser.add_argument('--val-items', type=int, default=0)
    parser.add_argument('--test-items', type=int, default=0)
    parser.add_argument('--action-normalizers', type=Path)
    args = parser.parse_args()
    cache_dir = args.cache_dir.resolve()
    limits = {
        'train': args.train_items,
        'val': args.val_items,
        'test': args.test_items,
    }
    configure_determinism(args.seed)
    feature_metadata = prepare_feature_cache(
        cache_dir, limits, args.encode_batch, args.seed
    )
    probe_metadata = prepare_probe_cache(cache_dir, feature_metadata)
    print(json.dumps({
        'feature_cache_key': feature_metadata['cache_key'],
        'probe_key': probe_metadata['probe_key'],
        'feature_timing': feature_metadata['timing'],
        'probe_total_seconds': probe_metadata['total_seconds'],
    }, indent=2), flush=True)
    if args.prepare_only:
        return
    if args.checkpoint is None or args.output is None:
        raise ValueError('--checkpoint and --output are required unless --prepare-only')
    report = evaluate_checkpoint(
        args.checkpoint.resolve(), cache_dir, args.output.resolve(), args.seed,
        args.eval_batch,
        None if args.action_normalizers is None else args.action_normalizers.resolve(),
    )
    print(json.dumps({'output': str(args.output), 'timing': report['timing']}, indent=2))


if __name__ == '__main__':
    main()
