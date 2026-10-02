"""Run the RD-WM Phase 1 replay gate.

Example:
    MUJOCO_GL=egl /venv/main/bin/python scripts/rdwm/replay_validate.py
    MUJOCO_GL=egl /venv/main/bin/python scripts/rdwm/replay_validate.py --env pusht --num-starts 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Headless rendering. ``NVIDIA_VISIBLE_DEVICES=void`` hides the GPUs from
# EGL even when nvidia-smi can see them.
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
if os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'void':
    del os.environ['NVIDIA_VISIBLE_DEVICES']

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdwm.data.manifest import ENVS  # noqa: E402
from rdwm.replay import format_table, run_gate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--env',
        action='append',
        choices=sorted(ENVS),
        help='Environment to run. Repeat to select several. Default: all.',
    )
    parser.add_argument('--num-starts', type=int, default=None)
    parser.add_argument(
        '--report',
        type=Path,
        default=Path('/tmp/rdwm_phase1_report.json'),
    )
    args = parser.parse_args()
    results = run_gate(args.env, num_starts=args.num_starts)
    args.report.write_text(json.dumps(results, indent=2, default=str))
    print(format_table(results))
    print(f'report: {args.report}')
    failed = [row['env'] for row in results if row.get('status') != 'PASS']
    if failed:
        for row in results:
            if row.get('status') == 'PASS':
                continue
            print(f"\n[{row['env']}] {row.get('error') or row.get('status')}")
            if row.get('boundary_problems'):
                print(' boundary:', row['boundary_problems'][:5])
        raise SystemExit(1)


if __name__ == '__main__':
    main()
