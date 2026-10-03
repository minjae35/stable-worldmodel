"""Closed-loop CEM evaluation of one Experiment 1 checkpoint.

    CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl /venv/main/bin/python scripts/rdwm/exp1_eval.py \
        --ckpt /workspace/rdwm_runs/exp1/stage1_smoke/B/seed0/pusht+cube/ckpt_001000.pt \
        --split val --use-pairs 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
if os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'void':
    del os.environ['NVIDIA_VISIBLE_DEVICES']
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rdwm.exp1.config import DEFAULT_CONFIG, load_config  # noqa: E402
from rdwm.exp1.evaluate import evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--split', default='val', choices=['val', 'test'])
    parser.add_argument('--use-pairs', type=int, default=None,
                        help='first N pairs of the fixed list (default: whole list)')
    parser.add_argument('--envs', default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    size = cfg['eval']['pilot_val_pairs'] if args.split == 'val' else cfg['eval']['test_pairs']
    summary = evaluate(
        cfg,
        Path(args.ckpt),
        args.split,
        size,
        args.use_pairs or size,
        args.envs.split(',') if args.envs else None,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != 'ckpt'}, indent=2))


if __name__ == '__main__':
    main()
