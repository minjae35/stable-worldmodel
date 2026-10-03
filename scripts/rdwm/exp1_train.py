"""One Experiment 1 training job.

    CUDA_VISIBLE_DEVICES=0 /venv/main/bin/python scripts/rdwm/exp1_train.py \
        --stage stage1_smoke --variant B --seed 0 --envs pusht,cube --updates 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rdwm.exp1.config import DEFAULT_CONFIG, load_config  # noqa: E402
from rdwm.exp1.train import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--stage', required=True)
    parser.add_argument('--variant', required=True, choices=['A', 'B', 'C', 'D'])
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--envs', required=True, help='comma-separated env names')
    parser.add_argument('--updates', type=int, required=True)
    parser.add_argument('--schedule-envs', default=None,
                        help='A only: generalist env set whose schedule is restricted')
    parser.add_argument('--ckpt-every', type=int, default=None)
    parser.add_argument('--ckpt-steps', default=None,
                        help='extra comma-separated updates to checkpoint at')
    parser.add_argument('--num-workers', type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.ckpt_every:
        cfg['train']['ckpt_every'] = args.ckpt_every
    if args.ckpt_steps:
        cfg['train']['ckpt_steps'] = {int(s) for s in args.ckpt_steps.split(',')}
    train(
        cfg,
        variant=args.variant,
        seed=args.seed,
        envs=args.envs.split(','),
        updates=args.updates,
        stage=args.stage,
        schedule_envs=args.schedule_envs.split(',') if args.schedule_envs else None,
        num_workers=args.num_workers,
    )


if __name__ == '__main__':
    main()
