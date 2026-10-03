"""Action-conditioning diagnostics for Experiment 1 checkpoints.

    CUDA_VISIBLE_DEVICES=2 /venv/main/bin/python scripts/rdwm/exp1_action_diag.py \
        --env pusht --dataset-checks --out /tmp/diag.json init:B:pusht,cube CKPT [CKPT ...]

``init:<variant>:<envs>`` is the seed-0 shared initialization with
train-split action stats loaded (update 0).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from rdwm.exp1.action_diag import (  # noqa: E402
    checkpoint_report,
    dataset_alignment,
    visual_change,
)
from rdwm.exp1.config import DEFAULT_CONFIG, env_path, load_config  # noqa: E402
from rdwm.exp1.data import EnvData  # noqa: E402
from rdwm.exp1.evaluate import load_model  # noqa: E402
from rdwm.exp1.init import build_model  # noqa: E402
from rdwm.exp1.util import write_json  # noqa: E402


def _init_model(cfg, spec, device):
    _, variant, envs = spec.split(':')
    envs = envs.split(',')
    model, _ = build_model(cfg, variant, envs, 0)
    for env in envs:
        data = EnvData(cfg, env, env_path(cfg, env))
        model.action_norm[env].load(data.action_mean, data.action_std)
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ckpts', nargs='*')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--env', default='pusht')
    parser.add_argument('--dataset-checks', action='store_true')
    parser.add_argument('--compare-env', default='cube')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device('cuda')
    report = {}
    if args.dataset_checks:
        report['alignment'] = dataset_alignment(cfg, args.env)
        report['visual_change'] = {
            args.env: visual_change(cfg, args.env),
            args.compare_env: visual_change(cfg, args.compare_env),
        }
        print(json.dumps({k: report[k] for k in ('alignment', 'visual_change')}, indent=1))
    report['checkpoints'] = {}
    for spec in args.ckpts:
        if spec.startswith('init:'):
            model = _init_model(cfg, spec, device)
        else:
            model = load_model(cfg, Path(spec), device)
        report['checkpoints'][spec] = checkpoint_report(model, cfg, args.env, device=device)
        sens = report['checkpoints'][spec]['sensitivity']
        mags = report['checkpoints'][spec]['magnitudes']
        print(spec, 'rel_action_effect_t2', round(sens['rel_action_effect_t2'], 4),
              'mse_true', [round(v, 5) for v in sens['mse_true']],
              'persist', [round(v, 5) for v in sens['persistence_mse']],
              'varying/env', round(mags['varying_over_env_ratio'], 3), flush=True)
    write_json(Path(args.out), report)


if __name__ == '__main__':
    main()
