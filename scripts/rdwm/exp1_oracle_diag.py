"""Oracle comparison of model dynamics, latent cost, and true goal distance.

    CUDA_VISIBLE_DEVICES=0 /venv/main/bin/python scripts/rdwm/exp1_oracle_diag.py
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
from rdwm.exp1.oracle_diag import diagnose  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument(
        '--ckpt',
        default='/workspace/rdwm_runs/exp1/diag_pusht/A/seed0/pusht/ckpt_005000.pt',
    )
    parser.add_argument(
        '--out',
        default='/workspace/rdwm_runs/exp1/logs/diag_pusht/oracle_diag_5k.json',
    )
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    diagnose(
        load_config(args.config),
        Path(args.ckpt),
        Path(args.out),
        torch.device('cuda'),
        workers=args.workers,
    )


if __name__ == '__main__':
    main()
