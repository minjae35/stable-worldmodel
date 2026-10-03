"""LeWM PushT ranking and closed-loop parity against the RD-WM evaluator."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
if os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'void':
    del os.environ['NVIDIA_VISIBLE_DEVICES']
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from rdwm.exp1.config import DEFAULT_CONFIG, load_config  # noqa: E402
from rdwm.exp1.lewm_diag import diagnose  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument(
        '--out',
        default='/workspace/rdwm_runs/exp1/logs/diag_pusht/lewm_diag.json',
    )
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--candidates', type=int, default=512)
    parser.add_argument('--pairs', type=int, default=10)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--seed0', type=int, default=1000)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    diagnose(
        load_config(args.config),
        Path(args.out),
        torch.device(args.device),
        workers=args.workers,
        n_candidates=args.candidates,
        n_pairs=args.pairs,
        pair_offset=args.offset,
        seed0=args.seed0,
    )


if __name__ == '__main__':
    main()
