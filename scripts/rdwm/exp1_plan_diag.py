"""PushT planning-failure diagnostic. Does not change the model or the loss.

    CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl /venv/main/bin/python scripts/rdwm/exp1_plan_diag.py
"""

from __future__ import annotations

import argparse
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

from rdwm.exp1.config import DEFAULT_CONFIG, load_config  # noqa: E402
from rdwm.exp1.plan_diag import diagnose  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument(
        '--ckpt',
        default='/workspace/rdwm_runs/exp1/diag_pusht/A/seed0/pusht/ckpt_005000.pt',
    )
    parser.add_argument(
        '--out',
        default='/workspace/rdwm_runs/exp1/logs/diag_pusht/plan_diag_5k.json',
    )
    parser.add_argument('--candidates', type=int, default=64)
    parser.add_argument('--pairs', type=int, default=10)
    args = parser.parse_args()
    cfg = load_config(args.config)
    diagnose(
        cfg,
        Path(args.ckpt),
        Path(args.out),
        args.candidates,
        args.pairs,
        torch.device('cuda'),
    )


if __name__ == '__main__':
    main()
