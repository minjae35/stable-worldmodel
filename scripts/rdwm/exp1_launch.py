"""Run an Experiment 1 job queue on the configured GPUs.

    /venv/main/bin/python scripts/rdwm/exp1_launch.py scripts/rdwm/configs/launch_stage1.yaml --dry-run
    /venv/main/bin/python scripts/rdwm/exp1_launch.py scripts/rdwm/configs/launch_stage1.yaml

GPU ids come from the launcher YAML (``gpus: [0, 1]``) or, with
``gpus: auto``, from this process's CUDA_VISIBLE_DEVICES.
Private datasets need HF_TOKEN only for downloads; training reads local Lance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rdwm.exp1.config import REPO_ROOT  # noqa: E402
from rdwm.exp1.launcher import Launcher, plan  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('launch_config')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    lcfg, cfg, jobs, gpus = plan(args.launch_config)
    log_dir = Path(cfg['paths']['run_root']) / 'logs' / lcfg['stage']
    launcher = Launcher(cfg, str(REPO_ROOT / lcfg['config']), jobs, gpus, log_dir)
    print(f'GPUs: {gpus}')
    for index, job in enumerate(jobs):
        state = 'done' if launcher.is_done(job) else 'pending'
        print(f'{index:2d} {job.job_id(cfg):50s} updates={job.updates:6d} {state}')
    if args.dry_run:
        return
    status = launcher.run()
    failed = [k for k, v in status.items() if v.get('state') == 'failed']
    print(f'status: {log_dir / "status.json"}')
    if failed:
        print('failed:', failed)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
