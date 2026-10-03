"""Train one budget, then rank and plan it, before the next budget.

Each budget is a fresh run. The cosine schedule covers that budget only.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
if os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'void':
    del os.environ['NVIDIA_VISIBLE_DEVICES']
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from rdwm.exp1.config import DEFAULT_CONFIG, load_config, run_dir  # noqa: E402
from rdwm.exp1.exposure_eval import evaluate_lewm, evaluate_rdwm, notify_result  # noqa: E402
from rdwm.exp1.lewm_train import train as train_lewm  # noqa: E402
from rdwm.exp1.train import train as train_rdwm  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--model', required=True, choices=['rdwm', 'lewm'])
    parser.add_argument('--budgets', required=True, help='comma-separated update counts')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--eval-workers', type=int, default=16)
    parser.add_argument('--allow-above-10k', action='store_true',
                        help='permit budgets above 10000; withheld until 5k/10k are reviewed')
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device('cuda')
    log_root = Path('/workspace/rdwm_runs/exp1/logs/exposure')
    log_root.mkdir(parents=True, exist_ok=True)
    for updates in [int(x) for x in args.budgets.split(',')]:
        if updates > 10000 and not args.allow_above_10k:
            print(f'[skip] {updates} updates: 25k and above stay off until 5k/10k are reviewed', flush=True)
            continue
        stage = f'exposure_{args.model}_{updates}'
        try:
            if args.model == 'rdwm':
                cfg['train']['ckpt_every'] = updates + 1
                train_rdwm(
                    cfg,
                    variant='A',
                    seed=args.seed,
                    envs=['pusht'],
                    updates=updates,
                    stage=stage,
                    schedule_envs=['pusht', 'cube'],
                    num_workers=args.num_workers,
                )
                ckpt = run_dir(cfg, stage, 'A', args.seed, ['pusht']) / f'ckpt_{updates:06d}.pt'
                gc.collect()
                torch.cuda.empty_cache()
                report = evaluate_rdwm(
                    cfg, ckpt, log_root / f'rdwm_{updates}.json', device, args.eval_workers
                )
            else:
                run = train_lewm(
                    cfg,
                    seed=args.seed,
                    updates=updates,
                    stage=stage,
                    num_workers=args.num_workers,
                )
                gc.collect()
                torch.cuda.empty_cache()
                report = evaluate_lewm(
                    cfg, run, log_root / f'lewm_{updates}.json', device, args.eval_workers
                )
            notify_result(args.model, updates, report)
        except Exception as exc:
            from rdwm.exp1.exposure_eval import _notify

            print(f'FAILED {args.model} {updates}: {type(exc).__name__}: {exc}', flush=True)
            try:
                _notify(f'exposure {args.model} {updates} | failed | {type(exc).__name__}: {exc}')
            except Exception:
                print('slack unset or notify failed', flush=True)
            raise


if __name__ == '__main__':
    main()
