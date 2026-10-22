"""Episode-paired G4 temporal-encoding minus G3 H3 bootstrap."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/workspace/stable-worldmodel')
sys.path.insert(0, str(ROOT))

from scripts.train.g3_evaluate import load_model, predict_history_bundle  # noqa: E402
from scripts.train.compare_g3_history import episode_means, paired_bootstrap  # noqa: E402
from scripts.train.g2_spatial import sha256_file  # noqa: E402
from stable_worldmodel.wm.spatial.g2 import ENV_NAMES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--experimental', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'refusing to overwrite {args.output}')
    baseline, base_ckpt = load_model(args.baseline, 'none')
    experimental, exp_ckpt = load_model(args.experimental, 'sinusoidal')
    rng = np.random.default_rng(64072 + int(exp_ckpt['step']))
    report = {
        'effect': 'G4_sinusoidal_minus_G3_H3; negative favors temporal encoding',
        'step': int(exp_ckpt['step']),
        'baseline_sha256': sha256_file(args.baseline),
        'experimental_sha256': sha256_file(args.experimental),
        'environments': {},
    }
    if int(base_ckpt['step']) != int(exp_ckpt['step']):
        raise ValueError('step mismatch')
    for index, env in enumerate(ENV_NAMES):
        bundle = torch.load(
            args.cache_dir / f'{env}_test.pt', map_location='cpu', weights_only=True
        )
        base = predict_history_bundle(baseline, bundle, index, 3, 32, 3089)['correct']
        exp = predict_history_bundle(experimental, bundle, index, 3, 32, 3089)['correct']
        target = bundle['target'].float()
        episodes = bundle['episode'].numpy()
        metrics = {}
        for name, a, b in (
            ('mse', (base-target).square().mean((1,2)), (exp-target).square().mean((1,2))),
            ('mae', (base-target).abs().mean((1,2)), (exp-target).abs().mean((1,2))),
        ):
            ea, va = episode_means(a.numpy(), episodes)
            eb, vb = episode_means(b.numpy(), episodes)
            if not np.array_equal(ea, eb):
                raise RuntimeError('episode mismatch')
            result = paired_bootstrap(va, vb, rng, args.draws)
            result['effect'] = report['effect']
            metrics[name] = result
        report['environments'][env] = metrics
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2); handle.write('\n')
    del baseline, experimental
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
