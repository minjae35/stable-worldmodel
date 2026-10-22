"""Episode-paired feature-error bootstrap for two G2 Fresh checkpoints."""

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

from scripts.train.g2_evaluate_shared import (  # noqa: E402
    evaluation_items,
    load_shard,
)
from scripts.train.g2_spatial import (  # noqa: E402
    DATASETS,
    normalize_actions,
    predict_bundle,
    sha256_file,
)
from stable_worldmodel.wm.spatial.g2 import ENV_NAMES, load_label_free_actions  # noqa: E402
from stable_worldmodel.wm.spatial.g2_fresh import torch_normalizers  # noqa: E402
from stable_worldmodel.wm.spatial.model import SpatialDynamics  # noqa: E402


def episode_means(values: np.ndarray, episodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(episodes)
    return unique, np.asarray([values[episodes == episode].mean() for episode in unique])


def paired_bootstrap(a, b, rng, draws: int) -> dict:
    if len(a) != len(b):
        raise ValueError('paired samples differ in length')
    delta = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    indices = rng.integers(0, len(delta), size=(draws, len(delta)))
    sampled = delta[indices].mean(axis=1)
    return {
        'effect': 'B_minus_A; negative favors temporal coverage',
        'n_paired_episodes': int(len(delta)),
        'observed_mean_delta': float(delta.mean()),
        'relative_delta_percent': float(100.0 * delta.mean() / np.asarray(a).mean()),
        'ci95': [float(x) for x in np.quantile(sampled, [0.025, 0.975])],
        'bootstrap_probability_B_better': float((sampled < 0).mean()),
        'draws': draws,
    }


def load_model(path: Path) -> SpatialDynamics:
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = SpatialDynamics().cuda()
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    return model


def compare_checkpoints(
    cache_dir: Path,
    cohort_manifest: Path,
    action_normalizers: Path,
    checkpoint_a: Path,
    checkpoint_b: Path,
    output: Path,
    seed: int = 43072,
    draws: int = 10000,
    eval_batch: int = 32,
) -> dict:
    """Compare two checkpoints with episode-paired bootstrap resampling."""
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')

    cache = cache_dir.resolve()
    manifest_path = cohort_manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    metadata = json.loads((cache / 'metadata.json').read_text())
    normalizer_artifact = json.loads(action_normalizers.read_text())
    normalizers = torch_normalizers(normalizer_artifact)
    model_a = load_model(checkpoint_a.resolve())
    model_b = load_model(checkpoint_b.resolve())
    rng = np.random.default_rng(seed)
    report = {
        'cohort_manifest': str(manifest_path),
        'cohort_manifest_sha256': sha256_file(manifest_path),
        'cache_key': metadata['cache_key'],
        'checkpoint_a_sha256': sha256_file(checkpoint_a),
        'checkpoint_b_sha256': sha256_file(checkpoint_b),
        'bootstrap_seed': seed,
        'environments': {},
    }
    for env_index, env in enumerate(ENV_NAMES):
        shard = load_shard(cache, env, 'test')
        items = evaluation_items(manifest, env, 'test', int(metadata['identity']['limits']['test']))
        raw = load_label_free_actions(
            lance.dataset(str(DATASETS[env])), items, env, int(manifest['frameskip'])
        )
        if not np.array_equal(raw['episode'], shard['episode'].numpy()):
            raise RuntimeError(f'{env} episode alignment mismatch')
        shard = dict(shard)
        shard['actions'] = normalize_actions(raw['actions'], normalizers[env])
        pred_a, _, _ = predict_bundle(model_a, shard, env_index, torch.device('cuda:0'), eval_batch, 3089)
        pred_b, _, _ = predict_bundle(model_b, shard, env_index, torch.device('cuda:0'), eval_batch, 3089)
        target = shard['target'].float()
        per_sample_a_mse = (pred_a - target).square().mean(dim=(1, 2)).numpy()
        per_sample_b_mse = (pred_b - target).square().mean(dim=(1, 2)).numpy()
        per_sample_a_mae = (pred_a - target).abs().mean(dim=(1, 2)).numpy()
        per_sample_b_mae = (pred_b - target).abs().mean(dim=(1, 2)).numpy()
        episodes, a_mse = episode_means(per_sample_a_mse, raw['episode'])
        episodes_b, b_mse = episode_means(per_sample_b_mse, raw['episode'])
        _, a_mae = episode_means(per_sample_a_mae, raw['episode'])
        _, b_mae = episode_means(per_sample_b_mae, raw['episode'])
        if not np.array_equal(episodes, episodes_b):
            raise RuntimeError(f'{env} paired episode mismatch')
        report['environments'][env] = {
            'n_clips': len(raw['episode']),
            'mse': paired_bootstrap(a_mse, b_mse, rng, draws),
            'mae': paired_bootstrap(a_mae, b_mae, rng, draws),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    del model_a, model_b
    torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--cohort-manifest', type=Path, required=True)
    parser.add_argument('--action-normalizers', type=Path, required=True)
    parser.add_argument('--checkpoint-a', type=Path, required=True)
    parser.add_argument('--checkpoint-b', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=43072)
    parser.add_argument('--draws', type=int, default=10000)
    parser.add_argument('--eval-batch', type=int, default=32)
    args = parser.parse_args()
    report = compare_checkpoints(
        args.cache_dir,
        args.cohort_manifest,
        args.action_normalizers,
        args.checkpoint_a,
        args.checkpoint_b,
        args.output,
        args.seed,
        args.draws,
        args.eval_batch,
    )
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
